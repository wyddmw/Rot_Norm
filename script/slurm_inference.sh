#!/usr/bin/env bash

set -x

JOB=$1
TRAINING_MODE=$2
GPUS=$3
PY_ARGS=${@:4}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}

# PRETRAINED_MODEL='./output/baseline/ckpt/checkpoint_29.pth'
# MODEL_NAME=point2_rot

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

if [ "${TRAINING_MODE}" == "Test" ]; then
    echo "using srun for training"
    srun -p ${TRAINING_MODE} --job-name=${JOB_NAME} -n 1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=1 \
    python -u inference.py --launcher slurm --tcp_port $PORT ${PY_ARGS}
    # --model_name ${MODEL_NAME} --width_threshold 0
else
    echo "using srun for training"
    srun -p ${TRAINING_MODE} --job-name=${JOB_NAME} -n ${GPUS} --gres=gpu:${GPUS_PER_NODE} --ntasks-per-node=${GPUS_PER_NODE} --cpus-per-task=${CPUS_PER_TASK} \
    python -u inference.py --launcher slurm --tcp_port $PORT ${PY_ARGS}
fi


