ENV_HOME=/mnt/lustre/share/polaris
ENV_NAME=pt1.5-share

CONDA_ROOT=/mnt/cache/mig_lod/anaconda3
GCC_ROOT=${ENV_HOME}/dep/gcc-5.4.0
CUDA_ROOT=${ENV_HOME}/dep/cuda-9.0-cudnn7.6.5
MPI_ROOT=${ENV_HOME}/dep/openmpi-4.0.3-cuda9.0-ucx1.7.0
UCX_ROOT=${ENV_HOME}/dep/ucx-1.7.0
NCCL_ROOT=${ENV_HOME}/dep/nccl_2.5.6-1-cuda9.0

export CUDA_HOME=${CUDA_ROOT}
export MPI_ROOT=${MPI_ROOT}
export NCCL_ROOT=${NCCL_ROOT}
export LD_LIBRARY_PATH=${GCC_ROOT}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CONDA_ROOT}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${CUDA_ROOT}/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${MPI_ROOT}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${UCX_ROOT}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${NCCL_ROOT}/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CONDA_ROOT}/envs/pt1.5-share/lib/python3.7/site-packages/spconv:$LD_LIBRARY_PATH


export PIP_CONFIG_FILE=${CONDA_ROOT}/envs/${ENV_NAME}/.pip/pip.conf
export LD_PRELOAD=${MPI_ROOT}/lib/libmpi.so

# export PYTHONUSERBASE=${HOME}/.local.${ENV_NAME}
export PATH=${GCC_ROOT}/bin:${CONDA_ROOT}/bin:${MPI_ROOT}/bin:${CUDA_ROOT}/bin:$PATH
export PYTHONPATH=./:./spconv/:$PYTHONPATH
export NUMBAPRO_NVVM=/mnt/lustre/share/polaris/dep/cuda-9.0-cudnn7.6.5/nvvm/lib64/libnvvm.so
export NUMBAPRO_LIBDEVICE=/mnt/lustre/share/polaris/dep/cuda-9.0-cudnn7.6.5/nvvm/libdevice

source /mnt/cache/mig_lod/anaconda3/bin/activate ${ENV_NAME}
