"""Command-line entry point for the reproducible descriptor workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .descriptors import extract_descriptors, load_volume, parse_shape
from .predict import load_metadata, predict_checkpoint
from .reporting import write_mechanism_report


def _ids(text: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in text.split(",") if value.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract CEMHYD3D multiscale descriptors and mechanism reports.")
    parser.add_argument("--volume", required=True, type=Path, help="3D phase-label volume (.npy, .tif/.tiff, or raw CEMHYD3D file).")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--shape", default="100,100,100", help="Raw-volume shape; ignored for NPY/TIFF unless needed to reshape.")
    parser.add_argument("--pore-ids", default="0")
    parser.add_argument("--csh-ids", default="14,20,21,30")
    parser.add_argument("--topology-space", choices=("pore", "solid"), default="pore")
    parser.add_argument("--reference-csv", type=Path, help="Optional cohort CSV to create a standardized mechanism report.")
    parser.add_argument("--metadata-json", type=Path, help="Composition/age JSON. Required to run neural strength prediction.")
    parser.add_argument("--teacher-checkpoint", type=Path, help="Optional compact macro+micro teacher checkpoint.")
    parser.add_argument("--student-checkpoint", type=Path, help="Optional compact macro-only distilled-student checkpoint.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    volume = load_volume(args.volume, parse_shape(args.shape))
    descriptors = extract_descriptors(volume, pore_ids=_ids(args.pore_ids), csh_ids=_ids(args.csh_ids), topology_space=args.topology_space)
    pd.DataFrame([descriptors]).to_csv(args.out_dir / "descriptors.csv", index=False)
    (args.out_dir / "descriptors.json").write_text(json.dumps(descriptors, indent=2), encoding="utf-8")
    if args.teacher_checkpoint or args.student_checkpoint:
        if not args.metadata_json:
            raise ValueError("--metadata-json is required when a checkpoint is supplied.")
        values = load_metadata(args.metadata_json)
        values.update({name: float(value) for name, value in descriptors.items()})
        predictions: dict[str, float] = {}
        if args.teacher_checkpoint:
            predictions["teacher_macro_micro_strength_mpa"] = predict_checkpoint(args.teacher_checkpoint, values, "teacher")
        if args.student_checkpoint:
            predictions["distilled_student_macro_strength_mpa"] = predict_checkpoint(args.student_checkpoint, values, "student")
        if len(predictions) == 2:
            predictions["microstructure_conditioned_minus_student_mpa"] = (
                predictions["teacher_macro_micro_strength_mpa"] - predictions["distilled_student_macro_strength_mpa"]
            )
        (args.out_dir / "strength_prediction.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    if args.reference_csv:
        write_mechanism_report(descriptors, args.reference_csv, args.out_dir)
    print(f"Wrote descriptors to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
