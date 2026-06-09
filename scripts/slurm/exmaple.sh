#!/bin/bash
#SBATCH -p acd_u         # 指定GPU队列
#SBATCH -o output_%j.txt  # 指定作业标准输出文件，%j为作业号
#SBATCH -e err_%j.txt    # 指定作业标准错误输出文件
#SBATCH -n 8            # 指定CPU总核心数
#SBATCH --gres=gpu:1    # 指定GPU卡数
#SBATCH -D /apps        # 指定作业执行路径为/apps

# 以下是作业要执行的命令
echo "Job started at $(date)"
python your_script.py  # 假设要运行一个Python脚本
echo "Job ended at $(date)"