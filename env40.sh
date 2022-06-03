ENV_HOME=/mnt/lustre/share/polaris
ENV_NAME=pt1.5-share
 
GCC_ROOT=${ENV_HOME}/dep/gcc-5.4.0
CONDA_ROOT=/mnt/lustre/zhangsongyan/anaconda3
#CUDA_ROOT=${ENV_HOME}/dep/cuda-9.0-cudnn7.6.5
CUDA_ROOT=/mnt/lustre/share/cuda-10.0
MPI_ROOT=${ENV_HOME}/dep/openmpi-4.0.3-cuda9.0-ucx1.7.0
UCX_ROOT=${ENV_HOME}/dep/ucx-1.7.0
NCCL_ROOT=${ENV_HOME}/dep/nccl_2.5.6-1-cuda9.0

export PATH=~/.local/bin/:$PATH 
export CUDA_HOME=${CUDA_ROOT}
export MPI_ROOT=${MPI_ROOT}
export NCCL_ROOT=${NCCL_ROOT}
export LD_LIBRARY_PATH=${GCC_ROOT}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CONDA_ROOT}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${CUDA_ROOT}/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${MPI_ROOT}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${UCX_ROOT}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${NCCL_ROOT}/lib/:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/mnt/lustre/share/shishaoshuai/anaconda3/lib/python3.7/site-packages/spconv/:$LD_LIBRARY_PATH


export PIP_CONFIG_FILE=${CONDA_ROOT}/envs/${ENV_NAME}/.pip/pip.conf
export LD_PRELOAD=${MPI_ROOT}/lib/libmpi.so
 
# export PYTHONUSERBASE=${HOME}/.local.${ENV_NAME}
export PATH=${GCC_ROOT}/bin:${CONDA_ROOT}/bin:${MPI_ROOT}/bin:${CUDA_ROOT}/bin:$PATH
 
source /mnt/lustre/zhangsongyan/anaconda3/bin/activate ${ENV_NAME}
