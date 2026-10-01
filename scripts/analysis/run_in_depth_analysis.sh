#!/usr/bin/env bash
#SBATCH -p acd_u
#SBATCH -o output_%j.txt
#SBATCH -e err_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
# Based on scripts/slurm/exmaple.sh. Also runnable with bash on GPU hosts without Slurm.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
ANALYSIS_OUTPUT="${ANALYSIS_OUTPUT:-$PROJECT_ROOT/outputs/in_depth_analysis}"
ANALYSIS_CACHE="${ANALYSIS_CACHE:-/root/nasbak/cjy/robot_raw/motus_in_depth_analysis}"
PER_DOMAIN="${PER_DOMAIN:-1000}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ANALYSIS_OUTPUT"
exec > >(tee -a "$ANALYSIS_OUTPUT/run.log") 2>&1
echo "Analysis started: $(date -u +%FT%TZ)"
common=(--output "$ANALYSIS_OUTPUT" --cache "$ANALYSIS_CACHE" --per-domain "$PER_DOMAIN")
if [[ ! -f "$ANALYSIS_OUTPUT/samples.jsonl" ]]; then
  python scripts/analysis/in_depth_analysis.py sample "${common[@]}"
fi
for model in text without_lap with_lap; do
  python scripts/analysis/in_depth_analysis.py extract "${common[@]}" --model "$model"
done
python scripts/analysis/in_depth_analysis.py analyze "${common[@]}"
python scripts/analysis/in_depth_analysis.py report "${common[@]}"
echo "Analysis completed: $(date -u +%FT%TZ)"
