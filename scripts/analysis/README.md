# LAP in-depth analysis

Independent, additive analysis of LAP motion-language coverage and VLM representation
mixing. This does not import training datasets, alter training code, or construct WAN
and action-expert models.

## Run

Use the existing Python environment with torch, transformers (Qwen3-VL support),
PyAV, OpenCV, pyarrow, scipy, Pillow, matplotlib, ftfy, and the repository's vendored
UMT5 implementation. Install analysis-only dependencies without changing shared packages:

```bash
python -m pip install --no-deps \
  --target /root/nasbak/cjy/robot_raw/motus_in_depth_analysis/python_deps \
  -r scripts/analysis/requirements.txt

# 20-observation smoke (10 per domain):
ANALYSIS_OUTPUT="$PWD/outputs/in_depth_analysis/smoke" PER_DOMAIN=10 \
  bash scripts/analysis/run_in_depth_analysis.sh

# Formal run, 2,000 observations (1,000 per domain):
sbatch scripts/analysis/run_in_depth_analysis.sh

# GPU host without Slurm: same entrypoint, one selected idle GPU.
CUDA_VISIBLE_DEVICES=0 bash scripts/analysis/run_in_depth_analysis.sh
```

The job template follows `scripts/slurm/exmaple.sh`; it discovers the repository
from the script path rather than requiring `/apps`. `ANALYSIS_OUTPUT`,
`ANALYSIS_CACHE`, and `PER_DOMAIN` select output, large-cache storage, and size.
No pretrained weights are downloaded. Both checkpoint identities are fixed in the
Python entrypoint to the paths supplied by the user.

Stages can also be run separately:

```bash
python scripts/analysis/in_depth_analysis.py sample
python scripts/analysis/in_depth_analysis.py extract --model text
python scripts/analysis/in_depth_analysis.py extract --model without_lap
python scripts/analysis/in_depth_analysis.py extract --model with_lap
python scripts/analysis/in_depth_analysis.py analyze
python scripts/analysis/in_depth_analysis.py report
```

Sampling refuses to overwrite an existing manifest. Use a new output directory to
change the sample. Feature extraction resumes valid per-sample caches; cache identity
includes the manifest, extraction code, checkpoint path/size/mtime, model config,
software versions, and prompt. Cached observation bytes are verified before VLM use.
Existing checkpoint files must be treated as immutable; size/mtime identity is not a
full content digest of the 48 GB checkpoint.

## Protocol

- Four robot sources contribute 250 observations each (AgiBot, DROID, Fractal,
  Bridge), with task round-robin selection. EgoVerse contributes 1,000 observations,
  with a bounded per-task reservoir. Seed 42 fixes sampling. One observation per
  original episode; AgiBot uses original `h5_path`, EgoVerse uses `episode_key`.
- Prefer existing test/val splits, then train. Existing splits do **not** establish
  held-out status relative to these checkpoints. This is a balanced source study,
  not an estimate under the original training mixture.
- Reserve 16 subsequent frames for robot observations (the common action chunk)
  and 64 for ego (the sidecar generation window), or choose frame zero for shorter
  episodes. This excludes near-end frames; it does not standardize the different
  source-specific LAP prediction horizons. Missing media, empty labels, and decode
  failures are recorded in `sampling.json`; source quotas are not silently changed.
- Robot LAP lines are indexed by the video frame. Ego LAP lines are indexed by the
  sidecar generator's selected parquet rows, including episode filtering and sparse
  `frame_index` handling. Blank lines are never removed or shifted. AgiBot views are
  top-head above left/right wrist images, matching the loader; all observations are
  letterboxed to 384 x 320. PyAV decoding uses exact frame ordinal.
- Frozen local UMT5 encodes raw LAP text and a subject-normalized variant. Mean
  pooling includes nonpadding tokens/EOS; long descriptions fail rather than truncate.
  Normalization changes wrist/arm/hand to effector but retains side, number, unit,
  direction, and gripper text. UMT5 is an operational representation of motion
  language, not a calibrated metric of physical action equivalence.
- Strictly load `vlm_model.*` and `und_expert.vlm_adapter.*` from each checkpoint.
  Both have 626 VLM keys and 6 adapter keys. Input is the current observation plus
  the original task instruction and a fixed next-action question. No assistant
  answer, LAP label, state vector, or setup/control suffix enters the input.
- Extract the final valid prompt token from the last VLM layer (2048D), and its
  adapter output (512D). Run in eval/inference mode, then L2-normalize for analysis.
  Comparing hash digests of the actual processor tensors enforces identical input
  between checkpoints. Task instructions and camera layouts are still potential
  domain cues; the study does not isolate visual alignment.

## Statistics and figures

All quantitative statistics are computed in the normalized **original feature
space**, before PCA/t-SNE. k=20 excludes self; small smoke runs reduce k to N-1.
The cross-domain neighbor fraction measures domain mixing, while mean raw-LAP UMT5
cosine similarity of opposite-domain nearest neighbors measures semantic retention.
Mean feature variance and entropy effective rank diagnose collapse. These diagnostics
are reported, not collapsed into an arbitrary pass/fail definition of alignment.

Paired changes use 1,000 source-stratified episode-score bootstrap draws. They
condition on the fitted neighbor graphs: they do not refit neighborhoods, retrain
models, or measure training-seed uncertainty. Source quotas stay fixed in resampling.

For text coverage, report bidirectional nearest cross-domain distance quantiles,
five closest and five farthest queries per direction, and lexical motion-clause
frequencies. “Bilateral labels” describes annotation wording, not verified simultaneous
physical movement. Gripper clauses are absent in some human annotations by design.
Farther text neighborhoods therefore suggest annotation-space differences, not proven
novel physical capabilities or downstream training benefit.

Main t-SNE: L2 features -> PCA (up to 50 dimensions) -> t-SNE with PCA initialization,
perplexity 30, learning_rate=auto, 1,500 iterations, seed 42. Sensitivity runs show the
full Cartesian product of seeds 0/1/2 and perplexities 15/30/50. Smoke runs reduce the
main perplexity and omit invalid values. Domains share a fit within each panel;
checkpoints have separate fits and their 2D axes are not comparable. No checkpoint
alignment, plot-based cherry-picking, or quantitative 2D overlap measurement is used.

## Outputs and evidence boundaries

- `samples.jsonl`, `sampling.json`: exact sample/frame/LAP/image provenance and exclusions.
- `{text,without_lap,with_lap}_complete.json`: extraction completion, cache paths,
  dimensions encoded in NPZ arrays, and identities. Large arrays/images stay in cache.
- `metrics.json`, `metrics.csv`, `sample_scores.csv`: metrics, intervals, feature
  diagnostics, per-source breakdowns, per-sample contributions, and caveats.
- `nearest_examples.json`: deterministic coverage examples with traceable sample IDs.
- `figures/`: main, supplementary, high-dimensional metrics, and every sensitivity
  setting, in PDF and PNG. `projections/` stores reproducible t-SNE coordinates.
- `analysis_complete.json`: present only after statistics and all requested figures.
- `REPORT.md`, `in_depth_analysis.tex`: result-grounded English report and paper text.

The checkpoint without-LAP identity was explicitly confirmed by the user; its
current YAML has changed and no historical training YAML is archived there. The
two checkpoints differ in training duration (400k vs 200k), data mixture, action
dimension (24 vs 14), and potentially initialization. The generated text accordingly
describes an observational checkpoint comparison, not a controlled LAP-only ablation.
It never presupposes the desired positive result.

Validation:

```bash
python -m unittest discover -s tests -p test_in_depth_analysis.py -v
python -m py_compile scripts/analysis/in_depth_analysis.py scripts/analysis/analysis_report.py
bash -n scripts/analysis/run_in_depth_analysis.sh
```

Method references: [t-SNE](https://www.jmlr.org/papers/v9/vandermaaten08a.html),
[scikit-learn TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).
