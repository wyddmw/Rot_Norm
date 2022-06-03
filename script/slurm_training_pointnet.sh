#!/usr/bin/env bash

set -x

JOB_NAME=$1
GPUS=$3
TRAINING_MODE=$2
PY_ARGS=${@:4}

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-4}

# training config
EPOCHS=30
PRETRAINED_MODEL=None
LR=1e-3
MODEL_NAME=pointnet_rot_dir
SCHEDULE_PARAM='{"gamma":0.9}'
LR_SCHEDULE='Exponential'

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
    srun -p ${TRAINING_MODE} --job-name=${JOB_NAME} -n 1 --gres=gpu:1 \
    --ntasks-per-node=1 --cpus-per-task=${CPUS_PER_TASK} \
    python -u train.py --launcher slurm --tcp_port $PORT --pretrained_model ${PRETRAINED_MODEL} \
    --LR ${LR} --model_name ${MODEL_NAME} --lr_schedule $LR_SCHEDULE \
    --scheduler_param ${SCHEDULE_PARAM} --epochs ${EPOCHS} --use_planes --width_threshold 40

else
    echo "using srun for training"
    srun -p ${TRAINING_MODE} --job-name=${JOB_NAME} -n ${GPUS} --gres=gpu:${GPUS} \
    --ntasks-per-node=${GPUS} --cpus-per-task=${CPUS_PER_TASK} \
    python -u train.py --launcher slurm --tcp_port $PORT --pretrained_model ${PRETRAINED_MODEL} \
    --LR ${LR} --model_name ${MODEL_NAME} --lr_schedule $LR_SCHEDULE \
    --scheduler_param ${SCHEDULE_PARAM} ${PY_ARGS}

fi


