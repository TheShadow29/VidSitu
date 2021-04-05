"""
Utility functions
"""

from typing import Dict, List, Optional, Union, Callable
import torch
from torch import nn
from pathlib import Path
import sys
from tqdm import tqdm
import time
import json
import datetime
from dataclasses import dataclass
from fastprogress.fastprogress import master_bar, progress_bar
import logging
import pickle
from torch import distributed as dist
from torch.utils.data import DataLoader
import mlflow
from utils.dat_utils import DataWrap
from yacs.config import CfgNode as CN
from vidsitu_code.extended_config import CfgProcessor
from slowfast.utils.checkpoint import load_checkpoint


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        raise TypeError("Invalid type for move_to")


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def reduce_dict(input_dict, average=False):
    """
    Args:
    input_dict (dict): all the values will be reduced
    average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
            values = torch.stack(values, dim=0)
            dist.reduce(values, dst=0)
            # if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            # values /= world_size
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def reduce_dict_corr(input_dict, nums):
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    new_inp_dict = {k: v * nums for k, v in input_dict.items()}
    out_dict = reduce_dict(new_inp_dict)
    dist.reduce(nums, dst=0)
    if not is_main_process():
        return out_dict
    out_dict_avg = {k: v / nums.item() for k, v in out_dict.items()}
    return out_dict_avg


def exec_func_if_main_proc(func: Callable):
    def wrapper(*args, **kwargs):
        if is_main_process():
            func(*args, **kwargs)

    return wrapper


class SmoothenValue:
    """
    Create a smooth moving average for a value(loss, etc) using `beta`.
    Adapted from fastai(https://github.com/fastai/fastai)
    """

    def __init__(self, beta: float):
        self.beta, self.n, self.mov_avg = beta, 0, 0
        self.smooth = 0

    def add_value(self, val: float) -> None:
        "Add `val` to calculate updated smoothed value."
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)


class SmoothenDict:
    "Converts list to dicts"

    def __init__(self, keys: List[str], val: int):
        self.keys = keys
        self.smooth_vals = {k: SmoothenValue(val) for k in keys}

    def add_value(self, val: Dict[str, torch.tensor]):
        for k in self.keys:
            self.smooth_vals[k].add_value(val[k].detach())

    @property
    def smooth(self):
        return {k: self.smooth_vals[k].smooth for k in self.keys}

    @property
    def smooth1(self):
        return self.smooth_vals[self.keys[0]].smooth

    def tostring(self):
        out_str = ""
        for k in self.keys:
            out_str += f"{k}: {self.smooth_vals[k].smooth.item():.4f} "
        out_str += "\n"
        return out_str

    def to_dict(self, prefix=""):
        dct1 = {k: self.smooth_vals[k].smooth.item() for k in self.keys}
        dct2 = dct_tensor_to_float(dct1)
        return get_dct_with_prefix(dct2, prefix=prefix)


SD = SmoothenDict


def compute_avg(inp: List, nums: torch.tensor) -> float:
    "Computes average given list of torch.tensor and numbers corresponding to them"
    return (torch.stack(inp) * nums).sum() / nums.sum()


def compute_avg_dict(inp: Dict[str, List], nums: torch.tensor) -> Dict[str, float]:
    "Takes dict as input"
    out_dict = {}
    for k in inp:
        out_dict[k] = compute_avg(inp[k], nums)

    return out_dict


def good_format_stats(names, stats) -> str:
    "Format stats before printing."
    str_stats = []
    for name, stat in zip(names, stats):
        t = str(stat) if isinstance(stat, int) else f"{stat.item():.4f}"
        t += " " * (len(name) - len(t))
        str_stats.append(t)
    return "  ".join(str_stats)


def get_dct_with_prefix(dct: Dict, prefix: str = ""):
    return {prefix + k: v for k, v in dct.items()}


def dct_tensor_to_float(dct):
    return {k: float(v) for k, v in dct.items()}


@dataclass
class MLFlowTracker:
    cfg: CN
    loss_keys: List[str]
    met_keys: List[str]
    txt_log_file: str

    @exec_func_if_main_proc
    def __post_init__(self):
        cfg_exp_id = self.cfg.expm.exp_name
        task_type = self.cfg.task_type
        exp_name = cfg_exp_id + "_" + task_type
        cfg_uid = self.cfg.uid
        exp_exist = mlflow.get_experiment_by_name(exp_name)
        if not exp_exist:
            mlflow.create_experiment(exp_name)
            exp_exist = mlflow.get_experiment_by_name(exp_name)
        exp_id = exp_exist.experiment_id
        self.active_run = mlflow.start_run(experiment_id=exp_id, run_name=cfg_uid)

        self.cfg.defrost()
        run_id = self.active_run.info.run_id
        st_time = datetime.datetime.fromtimestamp(
            int(str(self.active_run.info.start_time)[:10])
        )
        st_time_str = st_time.strftime("%Y-%m-%d %H:%M:%S")
        self.cfg.expm.run_id = run_id
        self.cfg.expm.st_time = st_time_str
        self.cfg.freeze()

    @exec_func_if_main_proc
    def save_cfg_file(self, inp_cfg: CN, cfg_fpath_dir: str, uid: str):
        run_id = self.active_run.info.run_id
        cfg_fpath = Path(cfg_fpath_dir) / f"cfg_file_{uid}_{run_id}.yml"
        with open(cfg_fpath, "w") as g:
            inp_cfg_str = CfgProcessor.to_str(inp_cfg)
            g.write(inp_cfg_str)
        mlflow.log_artifact(cfg_fpath)

        cfg_dct_flat = CfgProcessor.cfg_to_flat_dct(inp_cfg)
        cfg_dct_flat_keys = list(cfg_dct_flat.keys())
        for ix in range(0, len(cfg_dct_flat_keys), 100):
            cf_keys = cfg_dct_flat_keys[ix : ix + 100]
            cfg_params1 = {
                k: cfg_dct_flat[k] for k in cf_keys if k not in set(["cmd", "cmd_str"])
            }
            mlflow.log_params(cfg_params1)
        return

    @exec_func_if_main_proc
    def log_loss_batch(self, sm_loss_dct: SD, num_it: int):
        loss_dct = sm_loss_dct.to_dict(prefix="trn_batch_")
        mlflow.log_metrics(loss_dct, step=num_it)
        return

    @staticmethod
    def add_met(met_sd: Dict, prefix: str, step: int):
        if is_main_process():
            if met_sd is not None:
                if isinstance(met_sd, SmoothenDict):
                    met_dct = met_sd.to_dict(prefix=prefix)
                elif isinstance(met_sd, dict):
                    met_dct1 = dct_tensor_to_float(met_sd)
                    met_dct = get_dct_with_prefix(met_dct1, prefix=prefix)
                else:
                    import pdb

                    pdb.set_trace()
                    raise NotImplementedError
                mlflow.log_metrics(met_dct, step=step)
        return

    @exec_func_if_main_proc
    def log_met_loss_epoch(
        self,
        num_epoch: int,
        trn_loss: SD = None,
        trn_acc: SD = None,
        val_loss: SD = None,
        val_acc: SD = None,
    ):

        MLFlowTracker.add_met(trn_loss, "trn_", num_epoch)
        MLFlowTracker.add_met(trn_acc, "trn_", num_epoch)
        MLFlowTracker.add_met(val_loss, "val_", num_epoch)
        MLFlowTracker.add_met(val_acc, "val_", num_epoch)
        return

    @exec_func_if_main_proc
    def log_validation_epoch(self, num_epoch: int, val_loss: SD, val_acc: SD):
        MLFlowTracker.add_met(val_loss, "best_val_", num_epoch)
        MLFlowTracker.add_met(val_acc, "best_val_", num_epoch)

    @exec_func_if_main_proc
    def end_run(self):
        mlflow.log_artifact(str(self.txt_log_file))
        mlflow.end_run()


@dataclass
class Learner:
    uid: str
    data: DataWrap
    mdl: nn.Module
    loss_fn: nn.Module
    cfg: Dict
    eval_fn: nn.Module
    opt_fn: Callable
    device: torch.device = torch.device("cuda")

    def __post_init__(self):
        "Setup log file, load model if required"

        # Get rank
        self.rank = get_rank()

        self.init_log_dirs()

        self.prepare_log_keys()

        self.logger = self.init_logger()

        self.mlf_logger = MLFlowTracker(
            self.cfg,
            loss_keys=self.loss_keys,
            met_keys=self.met_keys,
            txt_log_file=self.txt_log_file,
        )
        self.prepare_log_file()

        # Set the number of iterations, epochs, best_met to 0.
        # Updated in loading if required
        self.num_it = 0
        self.num_epoch = 0
        self.best_met = 0

        # Resume if given a path
        if self.cfg.train["resume"]:
            self.load_model_dict(
                resume_path=self.cfg.train["resume_path"],
                load_opt=self.cfg.train["load_opt"],
            )
        elif self.cfg.mdl["load_sf_pretrained"]:
            if self.cfg.task_type == "vb":
                convert_from_caffe2 = self.cfg.sf_mdl.TRAIN.CHECKPOINT_TYPE == "caffe2"
                ckpt_pth = self.cfg.sf_mdl.TRAIN.CHECKPOINT_FILE_PATH
                assert Path(ckpt_pth).exists()
                if self.cfg.do_dist:
                    mdl1 = self.mdl.module.sf_mdl
                else:
                    mdl1 = self.mdl.sf_mdl
                load_checkpoint(
                    ckpt_pth,
                    model=mdl1,
                    data_parallel=False,
                    convert_from_caffe2=convert_from_caffe2,
                )
                self.logger.info(
                    f"Loaded model from Pretrained Weights from {ckpt_pth} Correctly"
                )
            elif self.cfg.task_type == "vb_arg":
                if is_main_process():
                    ckpt_pth = self.cfg.train.sfbase_pret_path
                    if self.cfg.do_dist:
                        mdl1 = self.mdl.module.sf_mdl
                    else:
                        mdl1 = self.mdl.sf_mdl
                    ckpt_data = torch.load(ckpt_pth)
                    ckpt_mdl_sd = ckpt_data["model_state_dict"]
                    # ckpt_mdl_sd = ckpt_mdl.state_dict()

                    def strip_module_from_key(key):
                        assert "module" == key.split(".")[0]
                        return ".".join(key.split(".")[1:])

                    if "module" == list(ckpt_mdl_sd.keys())[0].split(".")[0]:
                        ckpt_mdl_sd = {
                            strip_module_from_key(k): v for k, v in ckpt_mdl_sd.items()
                        }

                    sf_mdl_ckpt_dct = {
                        k.split(".", 1)[1]: v
                        for k, v in ckpt_mdl_sd.items()
                        if k.split(".", 1)[0] == "sf_mdl"
                    }
                    mdl1.load_state_dict(sf_mdl_ckpt_dct)
                    self.logger.info(
                        f"Loaded model from Pretrained SFBase from {ckpt_pth} Correctly"
                    )

                if self.cfg.train.freeze_sfbase:
                    if self.cfg.do_dist:
                        mdl1 = self.mdl.module.sf_mdl
                    else:
                        mdl1 = self.mdl.sf_mdl
                    for param in mdl1.parameters():
                        param.requires_grad = False
                    self.logger.info("Freezing SFBase")

    def init_logger(self):
        logger = logging.getLogger(__name__)
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        if not is_main_process():
            return logger
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(str(self.extra_logger_file))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def init_log_dirs(self):
        """
        Convenience function to create the following:
        1. Log dir to store log file in txt format
        2. Extra Log dir to store the logger output
        3. Tb log dir to store tensorboard files
        4. Model dir to save the model files
        5. Predictions dir to store the predictions of the saved model
        6. [Optional] Can add some 3rd party logger
        """
        # Saves the text logs
        self.txt_log_file = Path(self.data.path) / "txt_logs" / f"{self.uid}.txt"

        # Saves the output of self.logger
        self.extra_logger_file = Path(self.data.path) / "ext_logs" / f"{self.uid}.txt"

        # Saves SummaryWriter outputs
        self.tb_log_dir = Path(self.data.path) / "tb_logs" / f"{self.uid}"

        # Saves the trained model
        self.model_file = Path(self.data.path) / "models" / f"{self.uid}.pth"

        # Train Model All Epochs
        self.model_epoch_dir = Path(self.data.path) / "model_epochs" / f"{self.uid}"

        # Saves the output predictions
        self.predictions_dir = Path(self.data.path) / "predictions" / f"{self.uid}"

        # Saves the currently used cfg
        self.saved_cfgs_dir = Path(self.data.path) / "cfgs_logs" / f"{self.uid}"

        self.create_log_dirs()

    @exec_func_if_main_proc
    def create_log_dirs(self):
        """
        Creates the directories initialized in init_log_dirs
        """
        self.txt_log_file.parent.mkdir(exist_ok=True, parents=True)
        self.extra_logger_file.parent.mkdir(exist_ok=True)
        self.tb_log_dir.mkdir(exist_ok=True, parents=True)
        self.model_file.parent.mkdir(exist_ok=True)
        self.model_epoch_dir.parent.mkdir(exist_ok=True)
        self.model_epoch_dir.mkdir(exist_ok=True)
        self.predictions_dir.mkdir(exist_ok=True, parents=True)
        self.saved_cfgs_dir.mkdir(exist_ok=True, parents=True)

    def prepare_log_keys(self):
        """
        Creates the relevant keys to be logged.
        Mainly used by the txt logger to output in a good format
        """

        def _prepare_log_keys(
            keys_list: List[List[str]], prefix: List[str]
        ) -> List[str]:
            """
            Convenience function to create log keys
            keys_list: List[loss_keys, met_keys]
            prefix: List['trn', 'val']
            """
            log_keys = []
            for keys in keys_list:
                for key in keys:
                    log_keys += [f"{p}_{key}" for p in prefix]
            return log_keys

        self.loss_keys = self.loss_fn.loss_keys
        self.met_keys = self.eval_fn.met_keys

        # When writing Training and Validation together
        self.trn_met = False
        self.log_keys = ["epochs"] + _prepare_log_keys([self.loss_keys], ["trn"])

        self.val_log_keys = ["epochs"] + _prepare_log_keys(
            [self.loss_keys, self.met_keys], ["val"]
        )
        self.log_keys += self.val_log_keys[1:]
        self.test_log_keys = ["epochs"] + _prepare_log_keys([self.met_keys], ["test"])

    @exec_func_if_main_proc
    def prepare_log_file(self):
        "Prepares the log files depending on arguments"
        f = self.txt_log_file.open("a")
        cfgtxt = json.dumps(self.cfg)
        f.write(cfgtxt)
        f.write("\n\n")
        f.close()
        self.mlf_logger.save_cfg_file(self.cfg, self.saved_cfgs_dir, self.uid)

    @exec_func_if_main_proc
    def update_log_file(self, towrite: str):
        "Updates the log files as and when required"
        with self.txt_log_file.open("a") as f:
            f.write(towrite + "\n")

    def get_predictions_list(self, predictions: Dict[str, List]) -> List[Dict]:
        "Converts dictionary of lists to list of dictionary"
        keys = list(predictions.keys())
        num_preds = len(predictions[keys[0]])
        out_list = [{k: predictions[k][ind] for k in keys} for ind in range(num_preds)]
        return out_list

    def validate(
        self,
        db: Union[DataLoader, Dict[str, DataLoader]] = None,
        mb=None,
        write_to_file=False,
    ) -> List[torch.tensor]:
        "Validation loop, done after every epoch"

        torch.cuda.empty_cache()
        self.mdl.eval()
        if db is None:
            dl = self.data.valid_dl
            dl_name = "valid"
        elif isinstance(db, DataLoader):
            dl = db
            dl_name = "valid"
        else:
            assert len(db) == 1
            dl_name = list(db.keys())[0]
            dl = db[dl_name]
        # if is_main_process():
        with torch.no_grad():
            out_loss, out_acc = self.eval_fn(
                self.mdl,
                self.loss_fn,
                dl,
                dl_name,
                rank=get_rank(),
                pred_path=self.predictions_dir,
                mb=mb,
            )

        synchronize()
        if is_main_process():
            if write_to_file:
                self.logger.debug(out_loss)
                self.logger.debug(out_acc)
                out_str = "  ".join(self.val_log_keys) + "\n"
                out_list = [self.num_epoch]
                out_list += [out_loss[k] for k in self.loss_keys]
                out_list += [out_acc[k] for k in self.met_keys]
                out_str += good_format_stats(self.val_log_keys, out_list)
                self.update_log_file(out_str)
                self.mlf_logger.log_validation_epoch(self.num_epoch, out_loss, out_acc)

        return out_loss, out_acc, {}

    def train_epoch(self, mb) -> List[torch.tensor]:
        "One epoch used for training"
        self.mdl.train()
        # trn_loss = SmoothenValue(0.9)
        trn_loss = SmoothenDict(self.loss_keys, 0.9)
        trn_acc = SmoothenDict(self.met_keys, 0.9)
        # synchronize()
        for batch_id, batch in enumerate(progress_bar(self.data.train_dl, parent=mb)):

            # Increment number of iterations
            self.num_it += 1
            batch = move_to(batch, self.device)
            self.optimizer.zero_grad()
            out = self.mdl(batch)
            out_loss = self.loss_fn(out, batch)
            loss = out_loss[self.loss_keys[0]]
            loss = loss.mean()
            if torch.isnan(loss).any():
                print("Pain In", batch["vseg_idx"])
            loss.backward()
            self.optimizer.step()

            # Returns original dictionary if not distributed parallel
            # loss_reduced = reduce_dict(out_loss, average=True)
            # metric_reduced = reduce_dict(metric, average=True)
            trn_loss.add_value(out_loss)

            comment_to_print = f"LossB {loss: .4f} | SmLossB {trn_loss.smooth1: .4f}"
            if self.trn_met:
                metric = self.eval_fn(out, batch)
                trn_acc.add_value(metric)
                comment_to_print += f" | AccB {trn_acc.smooth1: .4f}"
            mb.child.comment = comment_to_print
            if self.num_it % self.cfg.log.deb_it == 0:
                self.logger.debug(f"Num_it {self.num_it} {trn_loss.tostring()}")
                self.mlf_logger.log_loss_batch(trn_loss, self.num_it)
            del out_loss
            del loss
            del batch
        self.optimizer.zero_grad()
        out_loss = reduce_dict(trn_loss.smooth, average=True)
        if self.trn_met:
            out_met = reduce_dict(trn_acc.smooth, average=True)
        else:
            out_met = trn_acc.smooth
        return out_loss, out_met

    # @exec_func_if_main_proc
    def load_model_dict(
        self, resume_path: Optional[str] = None, load_opt: bool = False
    ):
        "Load the model and/or optimizer"

        def check_if_mgpu_state_dict(state_dict):
            return "module" == list(state_dict.keys())[0].split(".")[0]

        def strip_module_from_key(key):
            assert "module" == key.split(".")[0]
            return ".".join(key.split(".")[1:])

        if resume_path == "":
            mfile = self.model_file
        else:
            mfile = Path(resume_path)

        if not mfile.exists():
            self.logger.info(f"No existing model in {mfile}, starting from scratch")
            return
        try:
            checkpoint = torch.load(open(mfile, "rb"))
            self.logger.info(f"Loaded model from {mfile} Correctly")
        except OSError as e:
            self.logger.error(
                f"Some problem with resume path: {resume_path}. Exception raised {e}"
            )
            raise e
        if self.cfg.train["load_normally"]:
            mdl_state_dict_to_load = checkpoint["model_state_dict"]
            curr_mdl_state_dict = self.mdl.state_dict()
            checkp_mdl_mgpu = check_if_mgpu_state_dict(mdl_state_dict_to_load)
            curr_mdl_mgpu = check_if_mgpu_state_dict(curr_mdl_state_dict)

            if curr_mdl_mgpu != checkp_mdl_mgpu:
                if curr_mdl_mgpu:
                    self.mdl.module.load_state_dict(
                        mdl_state_dict_to_load, strict=self.cfg.train["strict_load"]
                    )
                if checkp_mdl_mgpu:
                    mdl_state_dict_to_load = {
                        strip_module_from_key(k): v
                        for k, v in mdl_state_dict_to_load.items()
                    }
                    self.mdl.load_state_dict(
                        mdl_state_dict_to_load, strict=self.cfg.train["strict_load"]
                    )
            else:
                if curr_mdl_mgpu:
                    mdl_state_dict_to_load = {
                        strip_module_from_key(k): v
                        for k, v in mdl_state_dict_to_load.items()
                    }
                    self.mdl.module.load_state_dict(
                        mdl_state_dict_to_load, strict=self.cfg.train["strict_load"]
                    )
                else:
                    self.mdl.load_state_dict(
                        mdl_state_dict_to_load, strict=self.cfg.train["strict_load"]
                    )

        if "num_it" in checkpoint.keys():
            self.num_it = checkpoint["num_it"]

        if "num_epoch" in checkpoint.keys():
            self.num_epoch = checkpoint["num_epoch"]

        if "best_met" in checkpoint.keys():
            self.best_met = checkpoint["best_met"]

        if load_opt:
            self.optimizer = self.prepare_optimizer()
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                self.lr_scheduler = self.prepare_scheduler(self.optimizer)
                self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    @exec_func_if_main_proc
    def save_model_dict(self, mdl_epocher: bool = False):
        "Save the model and optimizer"
        checkpoint = {
            "model_state_dict": self.mdl.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "num_it": self.num_it,
            "num_epoch": self.num_epoch,
            "cfgtxt": json.dumps(self.cfg),
            "best_met": self.best_met,
        }
        if not mdl_epocher:
            torch.save(checkpoint, self.model_file.open("wb"))
        else:
            mdl_file = self.model_epoch_dir / f"mdl_ep_{self.num_epoch}.pth"
            torch.save(checkpoint, mdl_file.open("wb"))

    def update_prediction_file(self, predictions, pred_file):
        rank = self.rank
        if self.cfg.do_dist:
            pred_file_to_use = pred_file.parent / f"{rank}_{pred_file.name}"
            pickle.dump(predictions, pred_file_to_use.open("wb"))
            if is_main_process() and self.cfg.do_dist:
                if pred_file.exists():
                    pred_file.unlink()
        else:
            pickle.dump(predictions, pred_file.open("wb"))

    @exec_func_if_main_proc
    def rectify_predictions(self, pred_file):
        world_size = get_world_size()
        pred_files_to_use = [
            pred_file.parent / f"{r}_{pred_file.name}" for r in range(world_size)
        ]
        assert all([p.exists() for p in pred_files_to_use])
        out_preds = []
        for pf in pred_files_to_use:
            tmp = pickle.load(open(pf, "rb"))
            assert isinstance(tmp, list)
            out_preds += tmp
        pickle.dump(out_preds, pred_file.open("wb"))

    def prepare_to_write(
        self,
        train_loss: Dict[str, torch.tensor],
        train_acc: Dict[str, torch.tensor],
        val_loss: Dict[str, torch.tensor] = None,
        val_acc: Dict[str, torch.tensor] = None,
        key_list: List[str] = None,
    ) -> List[torch.tensor]:
        if key_list is None:
            key_list = self.log_keys

        epoch = self.num_epoch
        out_list = [epoch]

        out_list += [train_loss[k] for k in self.loss_keys]
        if val_loss is not None:
            out_list += [val_loss[k] for k in self.loss_keys]
        if train_acc is not None:
            out_list += [train_acc[k] for k in self.met_keys]
        if val_acc is not None:
            out_list += [val_acc[k] for k in self.met_keys]

        assert len(out_list) == len(key_list)
        return out_list

    @property
    def lr(self):
        return self.cfg.train["lr"]

    @property
    def epoch(self):
        return self.cfg.train["epochs"]

    @exec_func_if_main_proc
    def master_bar_write(self, mb, **kwargs):
        mb.write(**kwargs)

    def fit(self, epochs: int, lr: float, params_opt_dict: Optional[Dict] = None):
        self.update_log_file("  ".join(self.log_keys) + "\n")
        "Main training loop"
        # Print logger at the start of the training loop
        self.logger.info(self.cfg)
        # Initialize the progress_bar
        mb = master_bar(range(epochs))
        # Initialize optimizer
        # Prepare Optimizer may need to be re-written as per use
        self.optimizer = self.prepare_optimizer(params_opt_dict)
        # Initialize scheduler
        # Prepare scheduler may need to re-written as per use
        self.lr_scheduler = self.prepare_scheduler(self.optimizer)

        # Write the top row display
        # mb.write(self.log_keys, table=True)
        self.master_bar_write(mb, line=self.log_keys, table=True)
        exception = False
        met_to_use = None
        # Keep record of time until exit
        if is_main_process():
            st_time = time.time()
        try:
            # Loop over epochs
            for epoch in mb:
                self.num_epoch += 1
                train_loss, train_acc = self.train_epoch(mb)
                synchronize()
                valid_loss, valid_acc, _ = self.validate(self.data.valid_dl, mb)
                synchronize()
                valid_acc_to_use = valid_acc[self.met_keys[0]]
                # Depending on type
                self.scheduler_step(valid_acc_to_use)

                # Now only need main process
                # Decide to save or not
                met_to_use = valid_acc[self.met_keys[0]].cpu()
                if self.best_met < met_to_use:
                    self.best_met = met_to_use
                    self.save_model_dict()
                if self.cfg.train.save_mdl_epochs:
                    self.save_model_dict(mdl_epocher=True)

                synchronize()
                # Prepare what all to write
                to_write = self.prepare_to_write(
                    train_loss, None, valid_loss, valid_acc
                )
                synchronize()
                # Display on terminal
                assert to_write is not None
                mb_write = [
                    str(stat) if isinstance(stat, int) else f"{stat:.4f}"
                    for stat in to_write
                ]
                self.master_bar_write(mb, line=mb_write, table=True)

                # Update in the log file
                self.update_log_file(good_format_stats(self.log_keys, to_write))
                self.mlf_logger.log_met_loss_epoch(
                    self.num_epoch, train_loss, None, valid_loss, valid_acc
                )
                synchronize()
        except (Exception, KeyboardInterrupt, RuntimeError) as e:
            exception = e
            self.mlf_logger.end_run()
            raise e
        finally:
            if is_main_process():
                end_time = time.time()
                self.update_log_file(
                    f"epochs done {epoch}. Exited due to exception {exception}. "
                    f"Total time taken {end_time - st_time: 0.4f}\n\n"
                )
                # Decide to save finally or not
                if met_to_use:
                    if self.best_met < met_to_use:
                        self.save_model_dict()

            synchronize()

    def testing(self, db: Dict[str, DataLoader]):
        if isinstance(db, DataLoader):
            db = {"dl0": db}
        for dl_name, dl in tqdm(db.items(), total=len(db)):
            out_loss, out_acc, preds = self.validate(dl)

            log_keys = self.val_log_keys

            to_write = self.prepare_to_write(out_loss, out_acc, key_list=log_keys)
            header = "  ".join(log_keys) + "\n"
            self.update_log_file(header)
            self.update_log_file(good_format_stats(log_keys, to_write))

            self.logger.info(header)
            self.logger.info(good_format_stats(log_keys, to_write))

            self.update_prediction_file(
                preds, self.predictions_dir / f"{dl_name}_preds.pkl"
            )

    def prepare_optimizer(self, params=None):
        "Prepare a normal optimizer"
        if not params:
            params = self.mdl.parameters()
        opt = self.opt_fn(params, lr=self.lr)
        return opt

    def prepare_scheduler(self, opt: torch.optim):
        "Prepares a LR scheduler on top of optimizer"
        self.sched_using_val_metric = self.cfg.train.use_reduce_lr_plateau
        if self.sched_using_val_metric:
            lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, factor=self.cfg.reduce_factor, patience=self.cfg.patience
            )
        else:
            lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: 1)

        return lr_sched

    def scheduler_step(self, val_metric):
        if self.sched_using_val_metric:
            self.lr_scheduler.step(val_metric)
        else:
            self.lr_scheduler.step()
        return

    def overfit_batch(self, epochs: int, lr: float):
        "Sanity check to see if model overfits on a batch"
        # guess = True
        # idx = 0
        # diter = iter(self.data.valid_dl)
        diter = iter(self.data.train_dl)
        batch = next(diter)
        batch = move_to(batch, self.device)
        self.mdl.train()
        opt = self.prepare_optimizer()

        for i in range(epochs):
            opt.zero_grad()
            out = self.mdl(batch)
            out_loss = self.loss_fn(out, batch)
            loss = out_loss[self.loss_keys[0]]
            loss = loss.mean()
            loss.backward()
            opt.step()
            # met = self.eval_f n(out, batch)
            out_str = f"Iter {i} | loss {loss: 0.4f}"
            out_str += " | ".join([f"{k}: {v.mean()}" for k, v in out_loss.items()])
            self.logger.debug(out_str)
            print(out_str)
            # print(f'Iter {i} | loss {loss: 0.4f} | acc {met: 0.4f}')
