# MedSAM2 Confidence-Gated Pseudo-Label Refinement for Semi-Supervised Tumor Segmentation

> **Paper:** Semi-Supervised Tumor Segmentation via MedSAM2 Confidence-Gated
> Pseudo-Label Refinement. Under review, IEEE Transactions on Biomedical
> Engineering (TBME), 2026.

Official implementation, evaluated on **KiTS21** (kidney tumor) and **LiTS** (liver tumor) at 5% and 10% label ratios.

## Method

A supervised seed model produces pseudo-labels on unlabeled scans, but systematically **under-segments** tumors. We address this in two stages:

1. **MedSAM2 tumor completion** — the under-segmented tumor is refined by prompting a frozen MedSAM2 video predictor at the densest tumor slice and propagating the mask bidirectionally along the z-axis. This recovers **+22% tumor volume** on average.

2. **Confidence gating** — each refined pseudo-label is scored by the fraction of predicted-tumor voxels whose tumor-class softmax probability exceeds 0.9. Only cases at or above tau = 0.80 are used for student retraining.

A student nnU-Net is then trained from scratch on the labeled ground truth combined with the gated pseudo-labels.

### Confidence score

```
c(y) = (1 / |Omega_t|) * SUM_{v in Omega_t}  1[ p_t(v) > 0.9 ],   Omega_t = { v : y(v) = tumor }
```

Both `p_t` and `Omega_t` are taken from the seed model f(.;theta_0); the resulting keep/discard decision is applied to the corresponding MedSAM2-refined label. A pseudo-label is kept if `c(y) >= tau = 0.80`.

This is a two-threshold criterion: a voxel counts as reliable if its tumor probability exceeds 0.9, and a case is kept if at least 80% of its predicted-tumor voxels clear that bar.

---

## Repository structure

```
nnunet_trainers/       SSL baseline trainers implemented on nnU-Net v2
  nnUNetTrainerMeanTeacher.py   Mean Teacher (NeurIPS 2017)
  nnUNetTrainerUAMT.py          UA-MT        (MICCAI 2019)
  nnUNetTrainerMCNet.py         MC-Net+      (MedIA 2022)
  nnUNetTrainerADMT.py          AD-MT        (ECCV 2024)
  nnUNetTrainerDyCON.py         DyCON        (CVPR 2025)
  train_launcher.py             training entry point

pipeline/
  1_seed_inference/      seed model inference on unlabeled scans
  2_medsam2_refinement/  MedSAM2 tumor completion (Contribution 1)
  3_confidence_gate/     confidence scoring and gating (Contribution 2)
  4_dataset_builders/    assemble nnU-Net datasets for student retraining

evaluation/              Dice, Jaccard, 95HD, ASD, significance tests
figures/                 scripts reproducing the paper figures
```

**Note on baselines.** Trainers for Mean Teacher, UA-MT, MC-Net+, DyCON, and AD-MT are included here. URPC and DTC were run using implementations following the original papers and the [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) reference code, under the same nnU-Net backbone, data splits, and training schedule as all other methods.

---

## Requirements

```bash
pip install -r requirements.txt
```

Also required: [MedSAM2](https://github.com/bowang-lab/MedSAM2) (SAM2.1-Tiny checkpoint, used frozen). Experiments were run with PyTorch 2.10 / CUDA 12.8 on a single NVIDIA RTX 5090.

---

## Usage

**1. Train the seed model**

```bash
python nnunet_trainers/train_launcher.py <DATASET_ID> 3d_fullres <FOLD> nnUNetTrainer_250epochs
```

**2. Generate pseudo-labels and refine with MedSAM2**

```bash
python pipeline/1_seed_inference/gen_v2_pseudo_labels_5pct.py
python pipeline/2_medsam2_refinement/gen_liver_medsam2_5pct.py
```

**3. Score and gate**

```bash
python pipeline/3_confidence_gate/predict_conf_sup10.py
```

**4. Build the student dataset and retrain**

```bash
python pipeline/4_dataset_builders/build_d511_liver_ours5.py
python nnunet_trainers/train_launcher.py 511 3d_fullres 0 nnUNetTrainer_250epochs
```

**5. Evaluate**

```bash
python evaluation/compute_all_metrics.py <TRAINER_NAME>
```

---

## Baselines

All nine methods were trained under **identical conditions** — same nnU-Net v2 backbone, same data splits, 250 epochs, same GPU — so that differences reflect the methods themselves rather than the training setup.

| Method | Venue |
|---|---|
| Supervised only | — |
| Mean Teacher | NeurIPS 2017 |
| UA-MT | MICCAI 2019 |
| DTC | AAAI 2021 |
| URPC | MedIA 2022 |
| MC-Net+ | MedIA 2022 |
| AD-MT | ECCV 2024 |
| DyCON | CVPR 2025 |
| **Ours** | — |

---

## Data

KiTS21 and LiTS are publicly available and are **not** redistributed here.

- KiTS21: https://github.com/neheller/kits21
- LiTS: https://competitions.codalab.org/competitions/17094

Both datasets were preprocessed with the standard nnU-Net v2 pipeline.

---

## Acknowledgements

Built on [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) and [MedSAM2](https://github.com/bowang-lab/MedSAM2). Baseline implementations follow the original papers and, where applicable, the [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) reference implementations.

---

## License

MIT — see [LICENSE](LICENSE).
