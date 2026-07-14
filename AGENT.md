# AGENT Instructions (Codex / Claudex)

## Scope
- This file defines project-level rules for coding agents working in this repository.

## Hard Rules
- Keep existing behavior intact when modifying code.
- Do not break original logic unless explicitly requested.

## Preferred Change Strategy
- Prefer additive and controlled changes.
- Add new logic branches instead of rewriting core paths.
- Add helper functions instead of inlining complex changes.
- Use parameters/flags to gate new behavior.

## GPU Job Submission
- Use this Slurm template for GPU jobs: ./scripts/slurm/exmaple.sh`

## 存储设置
大数据文件存放在/root/nasbak/cjy/robot_raw,训练用权重除外

