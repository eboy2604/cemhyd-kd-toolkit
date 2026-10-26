# CEMHYD-KD Toolkit

This repository package makes the manuscript's input-side analysis reproducible:

`labelled 3D CEMHYD3D volume -> seven multiscale descriptors -> optional strength predictions -> mechanism report`

It is deliberately a command-line tool, not a GUI. Its outputs are suitable for a supplementary-code release and are designed to make every input, phase convention and model checkpoint explicit.

## Scope and scientific boundary

The toolkit extracts the seven descriptors used by the core teacher/student study:

- porosity and C-S-H fraction;
- largest pore-cluster connectivity and pore-component count;
- pore-space Betti numbers `Beta_0`, `Beta_1` and `Beta_2` from cubical persistent homology.

The optional report standardizes one sample against a supplied reference cohort. It is a descriptive mechanism profile, **not a causal estimate**. Strength prediction requires both a validated checkpoint and the mixture/age metadata that match that checkpoint's feature schema. Do not use the bundled workflow to make claims beyond the training domain.

## Install

```powershell
cd CEMHYD-KD-Toolkit
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

For neural checkpoint inference, also install PyTorch:

```powershell
python -m pip install torch
```

## Minimal reproducible run

The following command accepts the project's raw CEMHYD3D snapshot format. Raw files require the voxel shape explicitly, so phase labels can never be reshaped silently.

```powershell
cemhyd-kd `
  --volume ..\Batch_Simulation_Results\Sample_0000\sample_0_dist.img.2300.20.100 `
  --shape 100,100,100 `
  --reference-csv ..\merged_macro_micro_strength_1030_csh_fixed.csv `
  --out-dir outputs\sample_0000
```

Outputs:

- `descriptors.csv` and `descriptors.json`: the model-facing descriptor record;
- `mechanism_attribution.csv` / `.png`: reference-standardized descriptor profile;
- `mechanism_report.md`: interpretation boundary and table.

`--volume` also accepts a labelled `*.npy` array and, with `tifffile` installed, a labelled 3D TIFF stack. The defaults assume CEMHYD3D pore phase `0` and C-S-H-related phases `14,20,21,30`; expose different phase mappings with `--pore-ids` and `--csh-ids`.

## Checkpoint prediction

The original compact neural model has a macro+micro **teacher** and a macro-only distilled **student**. A teacher prediction needs the volume-derived descriptors; a student prediction intentionally does not. Put the mixture composition and age in a JSON file using the *exact column names* recorded in the checkpoint, for example:

```json
{
  "Cement (component 1)(kg in a m^3 mixture)": 350.0,
  "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": 0.0,
  "Fly Ash (component 3)(kg in a m^3 mixture)": 0.0,
  "Water  (component 4)(kg in a m^3 mixture)": 180.0,
  "Superplasticizer (component 5)(kg in a m^3 mixture)": 5.0,
  "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": 1050.0,
  "Fine Aggregate (component 7)(kg in a m^3 mixture)": 760.0,
  "Age (day)": 28,
  "Water_Binder_Ratio": 0.5142857
}
```

Then run:

```powershell
cemhyd-kd --volume sample.npy --out-dir outputs\prediction `
  --metadata-json metadata.json `
  --teacher-checkpoint teacher_macro_micro_fold1.pt `
  --student-checkpoint student_distilled_macro_fold1.pt
```

The tool writes `strength_prediction.json`, including the teacher/student difference when both checkpoints are supplied. That difference is a model comparison diagnostic; it must not be labelled as a measured or causal microstructure effect.

The published release should include the selected compact MLP checkpoints, their `run_report.json`, and the reference cohort or its permitted access instructions. The higher-capacity final ensemble is intentionally rejected by this first adapter until its architecture is packaged and verified.

## Tests

```powershell
python -m pytest tests -q
```

## Recommended release contents

Keep the public repository small and auditable:

```text
CEMHYD-KD-Toolkit/
  cemhyd_kd/                 # package source
  tests/                     # format and descriptor tests
  checkpoints/               # only the validated release checkpoints
  examples/                  # one volume plus matching metadata
  environment.yml or requirements.txt
  README.md
  LICENSE
```

Avoid publishing raw simulation outputs, unpublished manuscript figures or large binaries unless their licence and provenance have been cleared.
