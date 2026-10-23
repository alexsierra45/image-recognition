#!/bin/bash -l
# Ejecuta un experimento en una GPU (CHG-029). Los argumentos son el comando, que corre dentro del .venv:
#   sbatch --job-name=EXP-040 slurm/run.sbatch tools/run_notebook.py xview_recognition.ipynb --python .venv/bin/python
#   sbatch --job-name=sweep_d3 slurm/run.sbatch sweep.py phase_d3_6
# Para cambiar de GPU o de tiempo, las opciones en la línea de sbatch pisan a las de aquí (--gres=gpu:a100:1, --time=...).
#SBATCH --job-name=xview
#SBATCH --partition=standard-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/y404/y404034/image-recognition
#SBATCH --output=experiments/logs/%x-%j.out

module purge
module load apps/2021 Python/3.13.5-GCCcore-14.3.0
source .venv/bin/activate

echo "host=$(hostname) job=$SLURM_JOB_ID gres=$SLURM_JOB_GPUS commit=$(git rev-parse --short HEAD) start=$(date -Is)"
echo "cmd: python $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python "$@"
status=$?
echo "end=$(date -Is) exit=$status"
exit $status
