import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch import autocast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMeanTeacher import nnUNetTrainerMeanTeacher
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = float(np.clip(current, 0.0, rampup_length))
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def sharpening(P, T=0.1):
    P_sharpen = P ** (1.0 / T) / (P ** (1.0 / T) + (1 - P) ** (1.0 / T) + 1e-8)
    return P_sharpen


class nnUNetTrainerMCNet(nnUNetTrainerMeanTeacher):
    """MC-Net+ (Wu et al., MedIA 2022) on nnU-Net. Shared encoder + two decoders
    (main + auxiliary with independent weights). Mutual consistency via sharpened
    soft pseudo-labels. No copy-paste / no mixed images -> stable."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.mc_temperature = 0.1
        self.mc_consistency_weight = 0.1
        self.mc_rampup = 40.0
        self._aux_decoder = None

    def initialize(self):
        if not self.was_initialized:
            # build network first (parent does this), but we need the aux decoder
            # BEFORE configure_optimizers is called. So replicate the order carefully.
            from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
            super().initialize()
            # after super().initialize(), network + optimizer exist; now build aux decoder
            base = self.network._orig_mod if hasattr(self.network, '_orig_mod') else self.network
            self._aux_decoder = copy.deepcopy(base.decoder)
            for m in self._aux_decoder.modules():
                if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                    nn.init.kaiming_normal_(m.weight, a=1e-2)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            self._aux_decoder = self._aux_decoder.to(self.device)
            # REBUILD optimizer to include aux decoder params
            import torch.optim as optim
            from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
            params = list(self.network.parameters()) + list(self._aux_decoder.parameters())
            self.optimizer = optim.SGD(params, self.initial_lr, weight_decay=self.weight_decay,
                                       momentum=0.99, nesterov=True)
            self.lr_scheduler = PolyLRScheduler(self.optimizer, self.initial_lr, self.num_epochs)
            self.print_to_log_file("[MC-Net+] auxiliary decoder built + optimizer rebuilt")

    def train_step(self, batch):
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        u = self._sample_unlabeled(self.unlabeled_batch_size).to(self.device, non_blocking=True)

        cw = self.mc_consistency_weight * sigmoid_rampup(self._mt_step / max(1, self.num_iterations_per_epoch), self.mc_rampup)

        self.optimizer.zero_grad(set_to_none=True)
        base = self.network._orig_mod if hasattr(self.network, '_orig_mod') else self.network

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            skips = base.encoder(data)
            out_main = base.decoder(skips)
            out_aux = self._aux_decoder(skips)
            sup = self.loss(out_main, target) + self.loss(out_aux, target)

            skips_u = base.encoder(u)
            um_main = base.decoder(skips_u)
            um_aux = self._aux_decoder(skips_u)
            m0 = um_main[0] if isinstance(um_main, (list, tuple)) else um_main
            a0 = um_aux[0] if isinstance(um_aux, (list, tuple)) else um_aux
            pm = torch.softmax(m0, 1)
            pa = torch.softmax(a0, 1)
            with torch.no_grad():
                pm_sharp = sharpening(pm, self.mc_temperature)
                pa_sharp = sharpening(pa, self.mc_temperature)
            consist = F.mse_loss(pm, pa_sharp) + F.mse_loss(pa, pm_sharp)

            loss = sup + cw * consist
            if not torch.isfinite(loss):
                loss = sup

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(list(self.network.parameters()) + list(self._aux_decoder.parameters()), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.network.parameters()) + list(self._aux_decoder.parameters()), 12)
            self.optimizer.step()

        self._mt_step += 1
        return {'loss': loss.detach().cpu().numpy()}

    def on_train_end(self):
        nnUNetTrainer.on_train_end(self)


class nnUNetTrainerMCNet_smoke(nnUNetTrainerMCNet):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 20
        self.num_val_iterations_per_epoch = 5
