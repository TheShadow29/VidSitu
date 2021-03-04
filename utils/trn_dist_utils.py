import torch
from torch import distributed as dist


def run_job(proc_rank, num_proc, func, init_method, cfg):
    print("world_size", num_proc)
    print("rank", proc_rank)

    torch.cuda.set_device(proc_rank)
    torch.distributed.init_process_group(
        backend=cfg.DIST_BACKEND,
        init_method=init_method,
        world_size=num_proc,
        rank=proc_rank,
    )
    print("dist_rank", dist.get_rank())
    func(cfg)


def launch_job(cfg, init_method, func, daemon=False):
    """
    Run 'func' on one or more GPUs, specified in cfg
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        init_method (str): initialization method to launch the job with multiple
            devices.
        func (function): job to run on GPU(s)
        daemon (bool): The spawned processes’ daemon flag. If set to True,
            daemonic processes will be created
    """
    if cfg.num_gpus > 1:
        assert cfg.do_dist
        torch.multiprocessing.spawn(
            run_job,
            nprocs=cfg.num_gpus,
            args=(cfg.num_gpus, func, init_method, cfg,),
            daemon=daemon,
        )
    else:
        assert not cfg.do_dist
        func(cfg=cfg)
