"""A compact, transparent mechanism report for one microstructure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .descriptors import DESCRIPTOR_COLUMNS


_DIRECTION = {
    "Porosity": -1,
    "CSH_Fraction": 1,
    "Max_Pore_Connectivity": -1,
    "Pore_Component_Count": 1,
    "Beta_0": 1,
    "Beta_1": 1,
    "Beta_2": 1,
}


def write_mechanism_report(descriptors: dict[str, float | int], reference_csv: str | Path, output_dir: str | Path) -> None:
    """Compare descriptors against a reference cohort without causal overclaiming."""
    output_dir = Path(output_dir)
    frame = pd.read_csv(reference_csv)
    missing = [name for name in DESCRIPTOR_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Reference CSV is missing descriptor columns: {missing}")
    reference = frame.loc[:, DESCRIPTOR_COLUMNS].apply(pd.to_numeric, errors="coerce")
    means, stds = reference.mean(), reference.std(ddof=0).replace(0, 1.0)
    report = []
    for name in DESCRIPTOR_COLUMNS:
        value = float(descriptors[name])
        z = float((value - means[name]) / stds[name])
        report.append({"descriptor": name, "value": value, "reference_mean": float(means[name]), "z_score": z,
                       "signed_mechanism_index": z * _DIRECTION[name]})
    report_frame = pd.DataFrame(report)
    report_frame.to_csv(output_dir / "mechanism_attribution.csv", index=False)
    with (output_dir / "mechanism_attribution.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)

    ordered = report_frame.sort_values("signed_mechanism_index")
    fig, axis = plt.subplots(figsize=(8, 4.8))
    colors = ["#b04a4a" if value < 0 else "#247a6b" for value in ordered["signed_mechanism_index"]]
    axis.barh(ordered["descriptor"], ordered["signed_mechanism_index"], color=colors)
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.set_xlabel("Reference-standardized, direction-signed descriptor deviation")
    axis.set_title("Microstructure mechanism profile")
    fig.tight_layout()
    fig.savefig(output_dir / "mechanism_attribution.png", dpi=200)
    plt.close(fig)

    lines = ["# Microstructure mechanism report", "", "## Interpretation boundary", "",
             "This report describes how the supplied volume differs from the reference cohort. "
             "It is not a causal estimate and is not a strength prediction.", "",
             "## Descriptor profile", "", report_frame.to_markdown(index=False, floatfmt=".4f"), ""]
    (output_dir / "mechanism_report.md").write_text("\n".join(lines), encoding="utf-8")
