# MCG-PL: Confidence-Gated Foundation-Model Refinement for Semi-Supervised Tumor Segmentation

Official code for **MedSAM2 Confidence-Gated Pseudo-Label Refinement (MCG-PL)**, evaluated on
**KiTS21** (kidney tumor) and **LiTS** (liver tumor) at 5% and 10% labeling ratios.

> Paper under review.

## Method

In low-label tumor segmentation, the dominant pseudo-label error is systematic under-segmentation of the
tumor, not random noise. MCG-PL corrects it with a frozen medical foundation model and admits a corrected
pseudo-label only when it can be trusted.

1. **Seed model.** A 3D full-resolution nnU-Net is trained on the labeled scans and predicts organ and tumor
   (with softmax probabilities) on the unlabeled scans.
2. **MedSAM2 completion.** Seed tumor components are filtered (minimum size `max(200, 0.25 x largest)` voxels,
   +/-5-slice window). Each retained component is anchored at its largest cross-section, plus further
   anchors spread along its extent. The seed's own tumor mask on each anchor slice is used as a **mask
   prompt** for the frozen MedSAM2 CT-lesion checkpoint, which propagates it bidirectionally through the volume.
3. **Additive merge.** The pseudo-label keeps the seed organ. Its tumor is the union of the propagated mask and
   the seed tumor, so no detected lesion is lost.
4. **Confidence gate.** `c = fraction of seed-predicted tumor voxels with p_tumor > 0.9`. A pseudo-label is
   admitted only if `c >= 0.80`.
5. **Student.** A new nnU-Net is trained from scratch on the labeled scans and the admitted pseudo-labels.
6. **Round two.** The student becomes the new seed and steps 1-5 are repeated. In this round, tumor components
   that do not touch the predicted organ dilated by 3 voxels are removed from the admitted pseudo-labels
   (organ-consistency filter). The resulting student is the final model.

`delta = 0.9`, `tau = 0.80` and the filter are fixed for all datasets and ratios. Every design choice was
adopted on the training pool with a decision rule fixed in advance, never on the test set.

## Repository structure

```
mcgpl/                          final pipeline (produces the reported MCG-PL results)
  e2e_pseudo.py                 round 1: seed inference + confidence, MedSAM2 completion, gate
  build_union.py                round 1: additive merge (seed tumor OR propagated tumor), gate, nnU-Net dataset
  r2.py                         round 2: round-1 student as seed; same steps, nnU-Net dataset
  organ_check.py                pre-declared training-pool check that adopted the organ-consistency filter
  build_oc.py                   organ-consistency filter on the round-2 pseudo-labels -> final training sets
  build_ablation.py             KiTS21 5% ablation datasets (plain, gate only, refinement only, tau 0.70 / 0.90)

pipeline/
  2_medsam2_refinement/         MedSAM2 completion, called by the scripts above (KiTS21: gen_v2_pseudo_labels_medsam2_*,
                                LiTS: gen_liver_medsam2_*)
  3_confidence_gate/            confidence scoring of the original single-round runs (the seed-1 LiTS
                                pseudo-labels and confidences reused by mcgpl/build_union.py come from these runs)
  4_dataset_builders/           dataset builders of the original single-round runs

baselines/samburden/            SAM-burden baseline: same frozen MedSAM2, 2D per-slice box+point prompting
nnunet_trainers/                semi-supervised baselines implemented as nnU-Net v2 trainers
evaluation/
  metrics_all.py                the single metric pipeline (Dice, Jaccard, 95HD, ASD; raw and organ rule)
  seed1_tables.py, seed1_extra.py   comparison tables and significance tests of the paper
  significance.py               three-seed MCG-PL vs. supervised tests (Holm over the primary comparisons)
  compute_all_metrics.py, hd95_*.py, run_significance.py   earlier evaluation scripts, superseded by the above
figures/                        scripts for the paper figures
legacy/                         earlier SAM2 experiments, NOT used for the reported results
```

All scripts contain absolute paths from our server (`/root/autodl-tmp/...`). Set them to your own
`nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results` locations before running.

## Requirements

```bash
pip install -r requirements.txt
```

Also required: [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) and
[MedSAM2](https://github.com/bowang-lab/MedSAM2) (`MedSAM2_CTLesion.pt`, SAM2.1 Hiera-Tiny, 512x512 input,
used frozen without fine-tuning). Experiments were run on a single NVIDIA RTX 5090. Every model is trained for
250 epochs with identical nnU-Net plans per dataset.

## Usage (one setting, one seed)

```bash
# round 1
python mcgpl/e2e_pseudo.py <setting> <seed>           # seed predictions, confidence, MedSAM2 completion
python mcgpl/build_union.py <pool> <dataset_id>        # additive merge + gate -> nnU-Net dataset
nnUNetv2_train <dataset_id> 3d_fullres 0 -tr nnUNetTrainer_250epochs

# round 2 (+ organ-consistency filter)
python mcgpl/r2.py gpu <pool>
python mcgpl/r2.py build <pool>
python mcgpl/build_oc.py <pool>
nnUNetv2_train <final_dataset_id> 3d_fullres 0 -tr nnUNetTrainer_250epochs
```

`<pool>` is one of `kits5_s1..3`, `kits10_s1..3`, `lits5_s1..3`, `lits10_s1..3`. Seeds 2 and 3 use the
trainers `nnUNetTrainer_250epochs_seed2/3`.

## Compared methods

All methods use the same nnU-Net v2 backbone, plans, data splits, 250-epoch schedule and GPU.

| Method | Venue |
|---|---|
| Supervised only | - |
| Mean Teacher | NeurIPS 2017 |
| UA-MT | MICCAI 2019 |
| DTC | AAAI 2021 |
| URPC | MedIA 2022 |
| MC-Net+ | MedIA 2022 |
| AD-MT | ECCV 2024 |
| DyCON | CVPR 2025 |
| SAM-burden (same frozen MedSAM2, 2D per-slice prompting) | - |
| **MCG-PL (ours)** | - |

Trainers for Mean Teacher, UA-MT, MC-Net+, AD-MT and DyCON are in `nnunet_trainers/`. The trainer files for
DTC and URPC were not preserved and are not included; they followed the original papers and the
[SSL4MIS](https://github.com/HiLab-git/SSL4MIS) reference implementations.

## Data

KiTS21 and LiTS are public and are **not** redistributed here.

- KiTS21: https://github.com/neheller/kits21
- LiTS: https://competitions.codalab.org/competitions/17094

## Acknowledgements

Built on [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) and [MedSAM2](https://github.com/bowang-lab/MedSAM2).

## License

MIT, see [LICENSE](LICENSE).
