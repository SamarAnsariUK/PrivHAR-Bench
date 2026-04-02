# PrivHAR-Bench

A multi-tier benchmark dataset for evaluating the privacy-utility trade-off in video-based human activity recognition.

**Paper:** *PrivHAR-Bench: A Graduated Privacy Benchmark Dataset for Video-Based Action Recognition* (arXiv preprint, DOI: 
https://doi.org/10.48550/arXiv.2604.00761, April 2026)

**Dataset:** [Zenodo (DOI: 10.5281/zenodo.19352048)](https://doi.org/10.5281/zenodo.19352048)

## Overview

PrivHAR-Bench provides 1,932 source videos across 15 activity classes, each distributed in 9 parallel privacy tiers:

| Tier | Method | Privacy Level |
|------|--------|---------------|
| Original | Unmodified | None |
| Tier 1 | Gaussian Blur (sigma=15) | Low |
| Tier 2 | Canny Edge Extraction | Medium |
| Tier 3a | AES Block Permutation (B=16) | High |
| Tier 3b | AES Block Permutation (B=8) | High |
| Tier 3c | AES Block Permutation (B=4) | High |
| Tier 3a-NoBG | B=16 + Background Removed | High + Context Control |
| Tier 3b-NoBG | B=8 + Background Removed | High + Context Control |
| Tier 3c-NoBG | B=4 + Background Removed | High + Context Control |

The NoBG variants isolate the transformed human region from background context, controlling for the well-documented scene bias in HAR models.

## Repository Structure

```
PrivHAR-Bench/
├── generate_dataset.py          # Master pipeline script
├── config/
│   ├── ucf101.yaml              # UCF101 configuration
│   └── ntu120.yaml              # NTU RGB+D 120 placeholder
├── src/
│   ├── detect.py                # YOLOv8-Pose person detection
│   ├── tier1_blur.py            # Gaussian blur transformation
│   ├── tier2_edge.py            # Canny edge extraction
│   ├── tier3_scramble.py        # AES block permutation
│   ├── nobg.py                  # Background removal
│   ├── tiers.py                 # Tier generation orchestration
│   ├── metadata.py              # Annotations and split generation
│   ├── metrics.py               # Privacy metrics (SSIM, PSNR)
│   ├── arcface_eval.py          # ArcFace identity verification experiment
│   ├── baseline.py              # R3D-18 baseline training
│   ├── eval.py                  # Evaluation toolkit
│   ├── report.py                # Summary report generation
│   └── utils.py                 # Shared utilities
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Important:** `pycryptodome` is required to reproduce the canonical dataset. The pipeline includes a SHA-256 fallback, but it produces different permutations and was not used to generate the distributed dataset.

### 2. Configure

Edit `config/ucf101.yaml` and set the correct path for your UCF101 directory:

```yaml
source_dir: "/path/to/UCF-101"
```

### 3. Download YOLOv8-Pose weights

Place `yolov8n-pose.pt` in the `weights/` directory.

### 4. Run the full pipeline

```bash
python generate_dataset.py --config config/ucf101.yaml
```

Or run specific phases:

```bash
# Detection only
python generate_dataset.py --config config/ucf101.yaml --phases 1

# Detection + tier generation + metadata
python generate_dataset.py --config config/ucf101.yaml --phases 1 2 3

# Baseline training only (requires phases 1-3 completed)
python generate_dataset.py --config config/ucf101.yaml --phases 6
```

The pipeline is resume-safe: re-running skips already-completed work.

## Source-Agnostic Design

The pipeline reads all dataset-specific parameters from the YAML config file. Switching to a different source dataset (e.g., NTU RGB+D 120) requires only a new config file:

```bash
python generate_dataset.py --config config/ntu120.yaml
```

The `config/ntu120.yaml` template is included as a placeholder. NTU RGB+D 120 requires a license from ROSE Lab (NTU Singapore).

## Evaluation Toolkit

To evaluate model predictions against the benchmark:

```bash
python -m src.eval --predictions_dir results/ --dataset_dir PrivHAR-Bench_v1.0.0/
```

The predictions directory should contain one CSV per tier with columns `video_id` and `predicted_label`.

The toolkit computes: Top-1 accuracy (per-class and overall), cross-tier accuracy drop, ROI-SSIM, ROI-PSNR, and the composite Privacy-Utility score.

## Reproducibility

Deterministic execution is enforced:

- All random seeds are fixed (seed=42)
- PyTorch deterministic mode is enabled
- cuDNN benchmarking is disabled
- YOLOv8 weight file can be verified by SHA-256 hash
- All library versions are pinned in `requirements.txt`

Minor per-pixel variations may arise from CUDA atomic operations on different GPU architectures. These do not affect bounding boxes or encryption outputs.

## Dataset Access

The pre-generated dataset is available on Zenodo without running this pipeline:

**DOI:** [10.5281/zenodo.19352048](https://doi.org/10.5281/zenodo.19352048)

The dataset includes all 9 tiers as lossless PNG frame sequences, annotations, splits, estimated poses, and the evaluation toolkit.

## License

The pipeline **code** is released under the MIT License.

The PrivHAR-Bench **dataset** is released under CC-BY-NC-4.0 and inherits restrictions from UCF101, which is released for research purposes only. Commercial use of the dataset is not permitted.

## Citation

```bibtex
@article{ansari2026privharbench,
  title={PrivHAR-Bench: A Graduated Privacy Benchmark Dataset for Video-Based Action Recognition},
  author={Ansari, Samar},
  journal={arXiv preprint, DOI: https://doi.org/10.48550/arXiv.2604.00761},
  year={2026}
}
```

## Contact

Samar Ansari, University of Chester: m.ansari@chester.ac.uk
