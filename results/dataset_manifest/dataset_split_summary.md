# Phyto Dataset Split Summary Report

## Overview
- **Total Physical Images**: 3,058
- **Random Seed**: 42
- **Stratification Standard**: 70% Train / 15% Validation / 15% Test

## Overall Split Distribution

| Split | Image Count | Percentage |
|---|---:|---:|
| Train | 2,140 | 69.98% |
| Validation | 458 | 14.98% |
| Test | 460 | 15.04% |
| **Total** | **3,058** | **100.00%** |

## Class-Wise Split Distribution

| Class Label | Total Images | Train Count ( % ) | Validation Count ( % ) | Test Count ( % ) |
|---|---:|---:|---:|---:|
| `early_leaf_spot` | 885 | 620 (70.06%) | 133 (15.03%) | 132 (14.92%) |
| `healthy_leaf` | 929 | 650 (69.97%) | 139 (14.96%) | 140 (15.07%) |
| `late_leaf_spot` | 689 | 482 (69.96%) | 103 (14.95%) | 104 (15.09%) |
| `nutrition_deficiency` | 329 | 230 (69.91%) | 49 (14.89%) | 50 (15.20%) |
| `rust` | 226 | 158 (69.91%) | 34 (15.04%) | 34 (15.04%) |

## Integrity & Validation Status
- [x] All 5 physical groundnut disease classes exist across all 3 splits.
- [x] Exactly 3,058 physical image records preserved (zero missing, zero duplicated).
- [x] Zero SHA256 hash collision/overlap across splits (100% hash isolation).
- [x] No physical image files were modified, moved, copied, or renamed.
