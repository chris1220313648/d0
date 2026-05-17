# Agent Runtime Guide

This document provides a minimal, executable guide for running GPU-required scripts on the cluster with SLURM.

## GPU 作业提交（SLURM）

### 1) 命令行提交（推荐）

```bash
sbatch -p acd_u -o output_%j.txt -e err_%j.txt -n 8 --gres=gpu:1 job_script.sh
```

说明：`--input` 仅用于标准输入重定向，不是常规提交脚本所必需参数。

### 2) 脚本式提交（推荐）

```bash
#!/bin/bash
#SBATCH -p acd_u
#SBATCH -o output_%j.txt
#SBATCH -e err_%j.txt
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH -D /apps

echo "Job started at $(date)"
python your_script.py
echo "Job ended at $(date)"
```

### 最小使用步骤

1. 保存脚本文件（例如 `job_script.sh`）。
2. 执行提交命令：`sbatch job_script.sh`。
3. 查看状态与日志：`squeue -u $USER` 或 `sacct -j <jobid>`。