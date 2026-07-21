import os, json, glob, copy
import numpy as np
import torch
import torch.nn.functional as F
import blosc2
from torch import autocast
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context


def sigmoid_rampup(t, T):
    if T <= 0:
        return 1.0
    p = float(np.clip(1.0 - t / T, 0.0, 1.0))
    return float(np.exp(-5.0 * p * p))


class nnUNetTrainerMeanTeacher(nnUNetTrainer):
    """3D Mean Teacher SSL baseline, fully inside the nnU-Net pipeline.
    Labeled cases come from the fold's train split (supervised loss).
    Unlabeled cases = all preprocessed cases NOT in train/val (consistency loss)."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250              # match V5.2 budget
        self.ema_alpha = 0.99
        self.consistency_max = 0.1
        self.unlabeled_batch_size = 2
        self._mt_step = 0
        self._rampup_steps = None


    def initialize(self):
        super().initialize()
        self._rampup_steps = int(0.4 * self.num_epochs * self.num_iterations_per_epoch)

    def on_train_start(self):
        super().on_train_start()
        if not hasattr(self, 'teacher'):
            base = self.network._orig_mod if hasattr(self.network, '_orig_mod') else self.network
            self.teacher = copy.deepcopy(base).to(self.device)
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self.teacher.eval()
            self._build_unlabeled_pool()

    def _build_unlabeled_pool(self):
        self._pre = self.preprocessed_dataset_folder
        splits = json.load(open(os.path.join(self.preprocessed_dataset_folder_base, "splits_final.json")))
        tr = set(splits[self.fold]["train"]); va = set(splits[self.fold]["val"])
        allc = set(os.path.basename(f)[:-5] for f in glob.glob(self._pre + "/*.b2nd")
                   if not f.endswith("_seg.b2nd"))
        self._unlab = sorted(allc - tr - va)
        self.print_to_log_file(f"[MeanTeacher] unlabeled pool = {len(self._unlab)} | "
                               f"labeled train = {len(tr)} | val = {len(va)}")
        self._ucache = {}
        for cid in self._unlab:
            self._ucache[cid] = np.asarray(blosc2.open(f"{self._pre}/{cid}.b2nd")[:], dtype=np.float32)
        self.print_to_log_file(f"[MeanTeacher] cached {len(self._ucache)} unlabeled volumes in RAM")

    def _sample_unlabeled(self, n):
        P = self.configuration_manager.patch_size
        px, py, pz = int(P[0]), int(P[1]), int(P[2])
        out = []
        for _ in range(n):
            cid = self._unlab[np.random.randint(len(self._unlab))]
            arr = self._ucache[cid]
            C, X, Y, Z = arr.shape
            x0 = np.random.randint(0, max(1, X - px + 1))
            y0 = np.random.randint(0, max(1, Y - py + 1))
            z0 = np.random.randint(0, max(1, Z - pz + 1))
            d = np.asarray(arr[:, x0:x0 + px, y0:y0 + py, z0:z0 + pz], dtype=np.float32)
            if d.shape[1:] != (px, py, pz):
                pad = np.zeros((C, px, py, pz), np.float32)
                pad[:, :d.shape[1], :d.shape[2], :d.shape[3]] = d
                d = pad
            for ax in (1, 2, 3):
                if np.random.rand() < 0.5:
                    d = np.flip(d, ax)
            out.append(np.ascontiguousarray(d))
        return torch.from_numpy(np.stack(out))

    def _consistency_loss(self, u):
        s_out = self.network(u)
        s_out = s_out[0] if isinstance(s_out, (list, tuple)) else s_out
        s_prob = torch.softmax(s_out, 1)
        with torch.no_grad():
            noise = torch.clamp(torch.randn_like(u) * 0.1, -0.2, 0.2)
            t_out = self.teacher(u + noise)
            t_out = t_out[0] if isinstance(t_out, (list, tuple)) else t_out
            t_prob = torch.softmax(t_out, 1)
        return F.mse_loss(s_prob, t_prob)

    def train_step(self, batch):
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
        u = self._sample_unlabeled(self.unlabeled_batch_size).to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            out_sup = self.network(data)
            sup = self.loss(out_sup, target)
            cons = self._consistency_loss(u)
            w = self.consistency_max * sigmoid_rampup(self._mt_step, self._rampup_steps)
            l = sup + w * cons

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

    def _update_teacher(self):
        a = min(1.0 - 1.0 / (self._mt_step + 1), self.ema_alpha)
        with torch.no_grad():
            for tp, sp in zip(self.teacher.parameters(), self.network.parameters()):
                tp.mul_(a).add_(sp.detach(), alpha=1.0 - a)
            for tb, sb in zip(self.teacher.buffers(), self.network.buffers()):
                tb.copy_(sb)

    def on_train_end(self):
        # save teacher as backup, then use it as the final model (standard MT)
        try:
            torch.save({'network_weights': self.teacher.state_dict()},
                       os.path.join(self.output_folder, 'teacher_final.pth'))
        except Exception as e:
            self.print_to_log_file(f"[MeanTeacher] teacher save failed: {e}")
        with torch.no_grad():
            for sp, tp in zip(self.network.parameters(), self.teacher.parameters()):
                sp.copy_(tp)
            for sb, tb in zip(self.network.buffers(), self.teacher.buffers()):
                sb.copy_(tb)
        super().on_train_end()
        # free the RAM cache of unlabeled volumes BEFORE final validation (prevents OOM)
        try:
            self._ucache = {}
        except Exception:
            pass
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class nnUNetTrainerMeanTeacher_smoke(nnUNetTrainerMeanTeacher):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 20
        self.num_val_iterations_per_epoch = 5
