#!/bin/bash -l
# Comprueba que el .venv ve la GPU desde un nodo de cómputo (CHG-029). Uso: sbatch slurm/gpu_check.sbatch
#SBATCH --job-name=gpu_check
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/y404/y404034/image-recognition
#SBATCH --output=experiments/logs/%x-%j.out

module purge
module load apps/2021 Python/3.13.5-GCCcore-14.3.0
source .venv/bin/activate

echo "host=$(hostname) job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES"
nvidia-smi
srun python - <<'EOF'
import sys, time
import tensorflow as tf, keras
print('python', sys.version.split()[0], '| tf', tf.__version__, '| keras', keras.__version__, '| backend', keras.backend.backend())
print('build', {k: v for k, v in tf.sysconfig.get_build_info().items() if 'cud' in k})
gpus = tf.config.list_physical_devices('GPU')
print('GPUs:', gpus)
assert gpus, 'TensorFlow no ve ninguna GPU'
x = tf.random.normal((64, 224, 224, 3))
conv = keras.layers.Conv2D(64, 3)
conv(x)
t = time.time()
for _ in range(20):
    y = conv(x)
_ = y.numpy()
print('20 conv 3x3 (64x224x224x3 -> 64): %.3f s en %s' % (time.time() - t, y.device))
EOF
