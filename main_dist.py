"""
Main file for distributed training
"""
import torch
import fire
from functools import partial
import sys
from yacs.config import CfgNode as CN
import os

os.environ["PYTHONWARNINGS"] = "ignore"


from utils.trn_utils import Learner
from utils.trn_dist_utils import launch_job
from vidsitu_code.extended_config import CfgProcessor
from vidsitu_code.mdl_selector import get_mdl_loss_eval
from vidsitu_code.dat_loader import get_data
import resource
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)


rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def get_name_from_inst(inst):
    return inst.__class__.__name__


def learner_init(uid: str, cfg: CN) -> Learner:
    mdl_loss_eval = get_mdl_loss_eval(cfg)
    get_default_net = mdl_loss_eval["mdl"]
    get_default_loss = mdl_loss_eval["loss"]
    get_default_eval = mdl_loss_eval["evl"]

    device = torch.device("cuda")
    data = get_data(cfg)
    comm = data.train_dl.dataset.comm
    mdl = get_default_net(cfg=cfg, comm=comm)

    loss_fn = get_default_loss(cfg, comm)
    loss_fn.to(device)

    comm2 = data.valid_dl.dataset.comm
    eval_fn = get_default_eval(cfg, comm2, device)
    eval_fn.to(device)
    opt_fn = partial(torch.optim.Adam, betas=(0.9, 0.99))

    # unfreeze cfg to save the names
    cfg.defrost()
    module_name = mdl
    cfg.mdl_data_names = CN(
        {
            "trn_data": get_name_from_inst(data.train_dl.dataset),
            "val_data": get_name_from_inst(data.valid_dl.dataset),
            "trn_collator": get_name_from_inst(data.train_dl.collate_fn),
            "val_collator": get_name_from_inst(data.valid_dl.collate_fn),
            "mdl_name": get_name_from_inst(module_name),
            "loss_name": get_name_from_inst(loss_fn),
            "eval_name": get_name_from_inst(eval_fn),
            "opt_name": opt_fn.func.__name__,
        }
    )
    cfg.freeze()
    if cfg.num_gpus > 0:
        cur_device = torch.cuda.current_device()
        mdl = mdl.to(device=cur_device)
        if cfg.num_gpus > 1:
            assert cfg.do_dist
            mdl = torch.nn.parallel.DistributedDataParallel(
                module=mdl,
                device_ids=[cur_device],
                output_device=cur_device,
                broadcast_buffers=True,
                find_unused_parameters=True,
            )

    learn = Learner(
        uid=uid,
        data=data,
        mdl=mdl,
        loss_fn=loss_fn,
        opt_fn=opt_fn,
        eval_fn=eval_fn,
        device=device,
        cfg=cfg,
    )
    return learn


def main_fn(cfg):
    uid = cfg.uid
    learn = learner_init(uid, cfg)

    # Train or Test
    if not (cfg.only_val or cfg.only_test or cfg.overfit_batch):
        learn.fit(epochs=cfg.train.epochs, lr=cfg.train.lr)
        if cfg.run_final_val:
            print("Running Final Validation using best model")
            torch.cuda.empty_cache()
            learn.load_model_dict(resume_path=learn.model_file, load_opt=False)
            val_loss, val_acc, _ = learn.validate(
                db={"valid": learn.data.valid_dl}, write_to_file=True
            )
            print(val_loss)
            print(val_acc)
        else:
            pass
    else:
        if cfg.overfit_batch:
            learn.overfit_batch(cfg.train.epochs, 1e-4)
        if cfg.only_val:
            val_loss, val_acc, _ = learn.validate(
                db={cfg.val_dl_name: learn.data.valid_dl}, write_to_file=True
            )
            print(val_loss)
            print(val_acc)
        if cfg.only_test:
            test_loss, test_acc, _ = learn.validate(
                db={cfg.test_dl_name: learn.data.test_dl}, write_to_file=True
            )
            print(test_loss)
            print(test_acc)
    if hasattr(learn, "mlf_logger"):
        learn.mlf_logger.end_run()
    return


def main_dist(uid: str, **kwargs):
    """
    uid is a unique identifier for the experiment name
    Can be kept same as a previous run, by default will start executing
    from latest saved model
    **kwargs: allows arbit arguments of cfg to be changed
    """
    CFP = CfgProcessor("./configs/vsitu_cfg.yml")
    cfg = CFP.get_vsitu_default_cfg()
    num_gpus = torch.cuda.device_count()
    cfg.num_gpus = num_gpus
    cfg.uid = uid
    argv = sys.argv
    cfg.cmd = argv
    cfg.cmd_str = " ".join(argv)
    if num_gpus > 1:
        # We are doing distributed parallel
        cfg.do_dist = True
    else:
        # We are doing data parallel
        cfg.do_dist = False
    # Update the config file depending on the command line args
    key_maps = CFP.get_key_maps()
    cfg = CFP.pre_proc_config(cfg, kwargs)
    cfg = CFP.update_from_dict(cfg, kwargs, key_maps)
    cfg = CFP.post_proc_config(cfg)
    cfg.freeze()
    print(cfg)
    launch_job(cfg, init_method="tcp://localhost:9997", func=main_fn)

    return


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    num_gpus = torch.cuda.device_count()
    fire.Fire(main_dist)
