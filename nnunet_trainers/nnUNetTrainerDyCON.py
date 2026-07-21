import os, json, glob, copy, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import blosc2
from torch import autocast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMeanTeacher import nnUNetTrainerMeanTeacher
from nnunetv2.utilities.helpers import dummy_context


def adaptive_beta(epoch, total_epochs, max_beta=5.0, min_beta=0.5):
    ratio = min_beta / max_beta
    return max_beta * (ratio ** (epoch / max(1, total_epochs)))


def sigmoid_rampup(t, T):
    if T <= 0:
        return 1.0
    p = float(np.clip(1.0 - t / T, 0.0, 1.0))
    return float(np.exp(-5.0 * p * p))


def sigmoid_rampup_thr(cur, total, lo, hi, steep=5.0):
    if total == 0:
        return hi
    cur = max(0.0, min(float(cur), total))
    phase = 1.0 - (cur / total)
    return lo + (hi - lo) * math.exp(-steep * (phase ** 2))


class UnCLoss(nn.Module):
    def forward(self, s_logits, t_logits, beta):
        EPS = 1e-6
        p_s = F.softmax(s_logits, dim=1)
        H_s = -torch.sum(p_s * torch.log(p_s + EPS), dim=1, keepdim=True)
        p_t = F.softmax(t_logits, dim=1)
        H_t = -torch.sum(p_t * torch.log(p_t + EPS), dim=1, keepdim=True)
        exp_H_s = torch.exp(beta * H_s)
        exp_H_t = torch.exp(beta * H_t)
        loss = (p_s - p_t) ** 2 / (exp_H_s + exp_H_t + EPS)
        loss = torch.mean(loss.sum(dim=1) + beta * (H_s + H_t))
        return loss.mean()


class FeCLoss(nn.Module):
    def __init__(self, device, temperature=0.6, gamma=2.0, use_focal=True, rampup_epochs=1500, lambda_cross=1.0):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.gamma = gamma
        self.use_focal = use_focal
        self.rampup_epochs = rampup_epochs
        self.lambda_cross = lambda_cross

    def forward(self, feat, mask, teacher_feat=None, epoch=0):
        B, N, _ = feat.shape
        mem_mask = torch.eq(mask, mask.transpose(1, 2)).float()
        mem_mask_neg = 1 - mem_mask
        feat_logits = torch.matmul(feat, feat.transpose(1, 2)) / self.temperature
        identity = torch.eye(N, device=self.device)
        neg_identity = 1 - identity
        feat_logits = feat_logits * neg_identity
        feat_logits = feat_logits - feat_logits.max(dim=1, keepdim=True)[0].detach()
        exp_logits = torch.exp(feat_logits)
        neg_sum = torch.sum(exp_logits * mem_mask_neg, dim=-1)
        division = exp_logits / (exp_logits + neg_sum.unsqueeze(-1) + 1e-18)
        loss_matrix = -torch.log(division + 1e-18) * mem_mask * neg_identity
        denom = (torch.sum(mem_mask, dim=-1) - 1 + 1e-18)
        loss_student = (torch.sum(loss_matrix, dim=-1) / denom).mean()
        if self.use_focal:
            similarity = division
            fw = torch.ones_like(similarity)
            pos_t = sigmoid_rampup_thr(epoch, self.rampup_epochs, 1.3, 1.5)
            neg_t = sigmoid_rampup_thr(epoch, self.rampup_epochs, 0.3, 0.5)
            hp = mem_mask.bool() & (similarity < pos_t)
            fw[hp] = (1 - similarity[hp]).pow(self.gamma)
            hn = mem_mask_neg.bool() & (similarity > neg_t)
            fw[hn] = similarity[hn].pow(self.gamma)
            loss_student = (torch.sum(loss_matrix * fw, dim=-1) / denom).mean()
        loss_cross = 0.0
        if teacher_feat is not None:
            cross_sim = torch.matmul(feat, teacher_feat.transpose(1, 2))
            neg_t = sigmoid_rampup_thr(epoch, self.rampup_epochs, 0.3, 0.5)
            chn = mem_mask_neg.bool() & (cross_sim > neg_t)
            if chn.sum() > 0:
                lc = -torch.log(1 - cross_sim + 1e-18) * chn.float()
                loss_cross = torch.sum(lc) / (torch.sum(chn.float()) + 1e-18)
        return loss_student + self.lambda_cross * loss_cross


class nnUNetTrainerDyCON(nnUNetTrainerMeanTeacher):

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.dycon_uncl_weight = 0.1      # was 0.5 (halved-and-more for stability)
        self.dycon_fecl_weight = 0.05     # contrastive loss is the unstable one -> smaller
        self.feature_scaler = 4
        self.fecl_num_points = 512
        self._uncl = UnCLoss()
        self._fecl = None
        self._stud_feat = None
        self._teach_feat = None

    def on_train_start(self):
        super().on_train_start()
        self._fecl = FeCLoss(device=self.device, temperature=0.6, gamma=2.0,
                             use_focal=True, rampup_epochs=int(0.6 * self.num_epochs))
        base = self.network._orig_mod if hasattr(self.network, '_orig_mod') else self.network
        self._stud_hook = base.decoder.stages[-1].register_forward_hook(self._save_stud)
        self._teach_hook = self.teacher.decoder.stages[-1].register_forward_hook(self._save_teach)
        self.print_to_log_file("[DyCON] hooks registered - stable version")

    def _save_stud(self, module, inp, out):
        self._stud_feat = out

    def _save_teach(self, module, inp, out):
        self._teach_feat = out

    def _embed(self, feat):
        B, C = feat.shape[0], feat.shape[1]
        e = feat.view(B, C, -1).transpose(1, 2)
        return F.normalize(e, dim=-1)

    def _downsample_mask(self, seg):
        m = F.interpolate(seg.unsqueeze(1).float(), scale_factor=1.0 / self.feature_scaler,
                          mode='trilinear', align_corners=False).squeeze(1)
        m = (m > 0.5).float()
        B = m.shape[0]
        return m.reshape(B, -1).unsqueeze(1)

    def train_step(self, batch):
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        u = self._sample_unlabeled(self.unlabeled_batch_size).to(self.device, non_blocking=True)

        epoch = self.current_epoch
        beta = adaptive_beta(epoch, self.num_epochs)
        # slow ramp-up of the DyCON losses (0 -> full over first 40% of training)
        ramp = self.consistency_max * sigmoid_rampup(self._mt_step, self._rampup_steps) / max(self.consistency_max, 1e-8)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            out_sup = self.network(data)
            sup = self.loss(out_sup, target)

            s_out = self.network(u)
            s_logits = s_out[0] if isinstance(s_out, (list, tuple)) else s_out
            s_feat = self._stud_feat
            with torch.no_grad():
                noise = torch.clamp(torch.randn_like(u) * 0.1, -0.2, 0.2)
                t_out = self.teacher(u + noise)
                t_logits = t_out[0] if isinstance(t_out, (list, tuple)) else t_out
                t_feat = self._teach_feat

            u_loss = self._uncl(s_logits, t_logits, beta)

            with torch.no_grad():
                pseudo = torch.argmax(t_logits, dim=1)
            s_emb = self._embed(s_feat)
            t_emb = self._embed(t_feat)
            mask_con = self._downsample_mask(pseudo)
            n = min(s_emb.shape[1], mask_con.shape[2])
            s_emb = s_emb[:, :n, :]
            t_emb = t_emb[:, :n, :]
            mask_con = mask_con[:, :, :n]
            if n > self.fecl_num_points:
                idx = torch.randperm(n, device=s_emb.device)[:self.fecl_num_points]
                s_emb = s_emb[:, idx, :]
                t_emb = t_emb[:, idx, :]
                mask_con = mask_con[:, :, idx]
            f_loss = self._fecl(feat=s_emb, mask=mask_con, teacher_feat=t_emb, epoch=epoch)

            # NaN guards: drop any loss term that went bad, so it can't poison the weights
            if not torch.isfinite(u_loss):
                u_loss = torch.zeros((), device=self.device)
            if not torch.isfinite(f_loss):
                f_loss = torch.zeros((), device=self.device)

            l = sup + ramp * (self.dycon_uncl_weight * u_loss + self.dycon_fecl_weight * f_loss)

            if not torch.isfinite(l):
                l = sup  # ultimate fallback: pure supervised this step

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        self._mt_step += 1
        self._update_teacher()
        return {'loss': l.detach().cpu().numpy()}

    def on_train_end(self):
        try:
            self._stud_hook.remove()
            self._teach_hook.remove()
        except Exception:
            pass
        super().on_train_end()


class nnUNetTrainerDyCON_smoke(nnUNetTrainerDyCON):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 20
        self.num_val_iterations_per_epoch = 5