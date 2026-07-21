import os, json, glob, copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMeanTeacher import (
    nnUNetTrainerMeanTeacher, sigmoid_rampup)


class nnUNetTrainerUAMT(nnUNetTrainerMeanTeacher):
    """Uncertainty-Aware Mean Teacher (Yu et al., MICCAI 2019) on the nnU-Net pipeline.

    On top of Mean Teacher:
      (1) backbone carries dropout (injected in build_network_architecture) so the
          EMA teacher can be Monte-Carlo sampled;
      (2) consistency target = MEAN of T MC-dropout teacher passes; per-voxel
          predictive ENTROPY of those passes = uncertainty;
      (3) consistency loss is MASKED to the most-confident voxels via a PERCENTILE
          threshold (keep_frac ramps 0.5 -> 0.9). On tasks where the net becomes
          confident quickly, an absolute ln(C)-scaled threshold passes ~all voxels
          and the masking degenerates to plain Mean Teacher; the percentile form
          keeps the masking genuinely selective throughout training.
    EMA update, unlabeled pool, rampup weight, BN-buffer copy, teacher->student at
    on_train_end are inherited unchanged.
    """

    MC_DROPOUT_P = 0.1

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.mc_T = 8
        self.keep_frac_start = 0.5
        self.keep_frac_end = 0.9

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        arch = configuration_manager.configuration['architecture']
        arch['arch_kwargs']['dropout_op'] = 'torch.nn.Dropout3d'
        arch['arch_kwargs']['dropout_op_kwargs'] = {'p': nnUNetTrainerUAMT.MC_DROPOUT_P}
        if 'dropout_op' not in arch['_kw_requires_import']:
            arch['_kw_requires_import'].append('dropout_op')
        return nnUNetTrainer.build_network_architecture(
            plans_manager, configuration_manager,
            num_input_channels, num_output_channels, enable_deep_supervision)

    def _set_teacher_mc_mode(self, on: bool):
        if on:
            for m in self.teacher.modules():
                if isinstance(m, nn.Dropout3d):
                    m.train()
        else:
            self.teacher.eval()

    def _consistency_loss(self, u):
        s_out = self.network(u)
        s_out = s_out[0] if isinstance(s_out, (list, tuple)) else s_out
        s_prob = torch.softmax(s_out, 1).float()

        with torch.no_grad():
            self._set_teacher_mc_mode(True)
            acc = None
            for _ in range(self.mc_T):
                noise = torch.clamp(torch.randn_like(u) * 0.1, -0.2, 0.2)
                t_out = self.teacher(u + noise)
                t_out = t_out[0] if isinstance(t_out, (list, tuple)) else t_out
                p = torch.softmax(t_out, 1).float()
                acc = p if acc is None else acc + p
            self._set_teacher_mc_mode(False)
            t_prob = acc / float(self.mc_T)
            C = t_prob.shape[1]
            ent = -(t_prob * torch.log(t_prob + 1e-6)).sum(1, keepdim=True)   # (B,1,...)
            H_max = float(np.log(C))
            # selective percentile threshold: keep most-confident keep_frac of voxels
            r = sigmoid_rampup(self._mt_step, self._rampup_steps)
            keep_frac = self.keep_frac_start + (self.keep_frac_end - self.keep_frac_start) * r
            flat = ent.flatten()
            if flat.numel() > 200000:
                idx = torch.randint(0, flat.numel(), (200000,), device=flat.device)
                flat = flat[idx]
            thresh = torch.quantile(flat, keep_frac).item()
            mask = (ent <= thresh).float()

        se = ((s_prob - t_prob) ** 2).mean(1, keepdim=True)
        denom = mask.sum().clamp_min(1.0)
        cons = (se * mask).sum() / denom

        if self._mt_step % 50 == 0:
            self.print_to_log_file(
                f"[UA-MT] step={self._mt_step} keep_frac={keep_frac:.3f} "
                f"confident_frac={mask.mean().item():.3f} thresh={thresh:.4f} H_max={H_max:.4f} "
                f"mc_T={self.mc_T} p={self.MC_DROPOUT_P}")
        return cons


class nnUNetTrainerUAMT_smoke(nnUNetTrainerUAMT):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 20
        self.num_val_iterations_per_epoch = 5
        self.mc_T = 4
