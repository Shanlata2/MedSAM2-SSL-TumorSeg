import copy
import numpy as np
import torch
import torch.nn.functional as F
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


class nnUNetTrainerADMT(nnUNetTrainerMeanTeacher):
    """AD-MT (Zhao et al., ECCV 2024) on nnU-Net.
    One student + TWO non-trainable EMA teachers.
      * RPA (Random Periodic Alternate): only ONE teacher is EMA-updated at a time;
        which one alternates after a RANDOM period.
      * CCM (Conflict-Combating): the two teachers' predictions are ensembled with
        entropy-based weights (more confident teacher gets more weight per voxel).
    No copy-paste, no mixed images, no extra trainable params -> stable."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.admt_weight = 1.0
        self.admt_rampup = 40.0
        self.rpa_min = 40
        self.rpa_max = 160
        self.teacher2 = None
        self._active_t = 0
        self._rpa_left = 0

    def on_train_start(self):
        super().on_train_start()
        self.teacher2 = copy.deepcopy(self.teacher)
        for p in self.teacher2.parameters():
            p.detach_()
        self.teacher2 = self.teacher2.to(self.device)
        self._active_t = 0
        self._rpa_left = int(np.random.randint(self.rpa_min, self.rpa_max))
        self.print_to_log_file("[AD-MT] second teacher built | RPA period init=%d" % self._rpa_left)

    @torch.no_grad()
    def _ema_into(self, tgt, alpha):
        src = self.network._orig_mod if hasattr(self.network, '_orig_mod') else self.network
        for tp, sp in zip(tgt.parameters(), src.parameters()):
            tp.data.mul_(alpha).add_(sp.data, alpha=1 - alpha)
        for tb, sb in zip(tgt.buffers(), src.buffers()):
            tb.data.copy_(sb.data)

    def _rpa_update(self):
        alpha = min(1 - 1 / (self._mt_step + 1), 0.99)
        tgt = self.teacher if self._active_t == 0 else self.teacher2
        self._ema_into(tgt, alpha)
        self._rpa_left -= 1
        if self._rpa_left <= 0:
            self._active_t = 1 - self._active_t
            self._rpa_left = int(np.random.randint(self.rpa_min, self.rpa_max))

    def _ccm(self, u):
        with torch.no_grad():
            o1 = self.teacher(u)
            o2 = self.teacher2(u)
            o1 = o1[0] if isinstance(o1, (list, tuple)) else o1
            o2 = o2[0] if isinstance(o2, (list, tuple)) else o2
            p1 = torch.softmax(o1, 1)
            p2 = torch.softmax(o2, 1)
            e1 = -(p1 * torch.log(p1 + 1e-8)).sum(1)
            e2 = -(p2 * torch.log(p2 + 1e-8)).sum(1)
            w1 = (e2 / (e1 + e2 + 1e-8)).unsqueeze(1)
            w2 = (e1 / (e1 + e2 + 1e-8)).unsqueeze(1)
            p_ens = w1 * p1 + w2 * p2
            pseudo = torch.argmax(p_ens, dim=1, keepdim=True)
        return pseudo

    def train_step(self, batch):
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        u = self._sample_unlabeled(self.unlabeled_batch_size).to(self.device, non_blocking=True)

        epoch_f = self._mt_step / max(1, self.num_iterations_per_epoch)
        w = self.admt_weight * sigmoid_rampup(epoch_f, self.admt_rampup)

        pseudo = self._ccm(u)

        def ds_targets(full, ref):
            if isinstance(ref, list):
                return [F.interpolate(full.float(), size=t.shape[2:], mode='nearest').long() for t in ref]
            return F.interpolate(full.float(), size=ref.shape[2:], mode='nearest').long()

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            out_l = self.network(data)
            sup = self.loss(out_l, target)
            out_u = self.network(u)
            unsup = self.loss(out_u, ds_targets(pseudo, target))
            loss = sup + w * unsup
            if not torch.isfinite(loss):
                loss = sup

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        self._mt_step += 1
        self._rpa_update()
        return {'loss': loss.detach().cpu().numpy()}

    def on_train_end(self):
        nnUNetTrainer.on_train_end(self)


class nnUNetTrainerADMT_smoke(nnUNetTrainerADMT):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 20
        self.num_val_iterations_per_epoch = 5
        self.rpa_min = 5
        self.rpa_max = 15
