import logging
import subprocess
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:
    
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank

def create_optimizer(model, opt):
    if opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.LR)
    elif opt.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr)
    return optimizer

def create_schedule(optimizer, opt):
    if opt.lr_schedule == 'None':
        return None
    elif opt.lr_schedule == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.scheduler_param['T_max'])
    elif opt.lr_schedule == 'stepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.scheduler_param['step_size'], gamma=opt.scheduler_param['gamma'])
    elif opt.lr_schedule == 'Exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=opt.scheduler_param['gamma'])
    else:
        raise NotImplemented('wrog lr schedule')
    return scheduler

def save_model(model, save_dir, optimizer=None, epoch=None):
    state = {}
    # model_state = {}
    ckpt_name = save_dir + '/checkpoint_%d.pth' % epoch
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_state = model_state_to_cpu(model.module.state_dict())
    else:
        model_state = model.module.state_dict()
    state['model_state'] = model_state
    optim_state = optimizer.state_dict() if optimizer is not None else None
    state['optimizer_state'] = optim_state
    state['epoch'] = epoch
    torch.save(state, ckpt_name)

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu

