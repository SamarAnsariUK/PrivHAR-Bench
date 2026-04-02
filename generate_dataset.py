#!/usr/bin/env python3
"""
PrivHAR-Bench Dataset Generation Pipeline

Generates the complete PrivHAR-Bench multi-tier privacy benchmark from source
video data. The pipeline is source-agnostic: switching between datasets (e.g.,
UCF101 vs NTU RGB+D 120) requires only a different config YAML file.

Pipeline phases:
  1. Detection     — YOLOv8-Pose person detection and pose estimation
  2. Tier generation — 9 parallel privacy-transformed frame sequences
  3. Metadata      — annotations.json, train/test splits, estimated poses
  4. Privacy metrics — ROI-SSIM and ROI-PSNR per tier
  5. ArcFace       — Face detection/verification under block scrambling
  6. Baseline      — R3D-18 training (Config A + Config B)
  7. Report        — Summary of all results

Usage:
    python generate_dataset.py --config config/ucf101.yaml
    python generate_dataset.py --config config/ntu120.yaml
    python generate_dataset.py --config config/ucf101.yaml --phases 1 2 3

All random seeds are fixed and deterministic execution is enforced.
Resume-safe: re-running skips already-completed work.
"""

import argparse
import sys
import time
import traceback

from src.utils import load_config, setup_determinism, init_logging, log, verify_weights
from src.utils import get_device
from src.tier3_scramble import is_aes_available


def parse_args():
    parser = argparse.ArgumentParser(
        description="PrivHAR-Bench Dataset Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with UCF101:
  python generate_dataset.py --config config/ucf101.yaml

  # Only run detection and tier generation:
  python generate_dataset.py --config config/ucf101.yaml --phases 1 2

  # Only run baseline training:
  python generate_dataset.py --config config/ucf101.yaml --phases 6
        """,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML configuration file (e.g., config/ucf101.yaml)",
    )
    parser.add_argument(
        "--phases", nargs="*", type=int, default=None,
        help="Specific phases to run (1-7). Default: all phases.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load and validate config
    cfg = load_config(args.config)

    # Set up determinism before any other initialization
    setup_determinism(cfg["seed"])

    # Initialize logging
    init_logging(cfg["output_dir"])

    # Verify weights if hash is specified
    if cfg.get("weight_hash"):
        verify_weights(cfg["weights_path"], cfg["weight_hash"])

    # Log startup information
    start = time.time()
    log(f"PrivHAR-Bench Pipeline starting at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Config:     {args.config}")
    log(f"Dataset:    {cfg['dataset_name']}")
    log(f"Source:     {cfg['source_dir']}")
    log(f"Output:     {cfg['output_dir']}")
    log(f"Device:     {get_device()}")
    log(f"AES lib:    {'pycryptodome (canonical)' if is_aes_available() else 'SHA-256 fallback (non-canonical)'}")
    log(f"Classes:    {len(cfg['classes'])}")
    log("")

    if not is_aes_available():
        log("WARNING: pycryptodome not installed. Using SHA-256 fallback.")
        log("  The fallback produces DIFFERENT permutations from the canonical AES-CTR path.")
        log("  Install pycryptodome for byte-identical output: pip install pycryptodome")
        log("")

    # Define phases
    from src.detect import run_detection
    from src.tiers import generate_tiers
    from src.metadata import generate_metadata
    from src.metrics import compute_privacy_metrics
    from src.arcface_eval import run_arcface_experiment
    from src.baseline import run_baseline_training
    from src.report import generate_report

    phases = [
        (1, "Detection",       lambda: run_detection(cfg)),
        (2, "Tier Generation", lambda: generate_tiers(cfg)),
        (3, "Metadata",        lambda: generate_metadata(cfg)),
        (4, "Privacy Metrics", lambda: compute_privacy_metrics(cfg)),
        (5, "ArcFace",         lambda: run_arcface_experiment(cfg)),
        (6, "Baseline",        lambda: run_baseline_training(cfg)),
        (7, "Report",          lambda: generate_report(cfg)),
    ]

    # Filter phases if specific ones requested
    if args.phases:
        phases = [(n, name, fn) for n, name, fn in phases if n in args.phases]

    # Execute phases
    for num, name, func in phases:
        try:
            func()
        except KeyboardInterrupt:
            log(f"\n  INTERRUPTED during Phase {num}: {name}. Progress is saved. Re-run to resume.")
            sys.exit(1)
        except Exception as e:
            log(f"\n  ERROR in Phase {num} ({name}): {e}")
            traceback.print_exc()
            log("  Continuing to next phase...\n")

    elapsed = time.time() - start
    log(f"\nTotal time: {elapsed / 3600:.1f} hours")
    log("Done.")


if __name__ == "__main__":
    main()
