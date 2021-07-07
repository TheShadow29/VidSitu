"""
Extract Visual Features for further processing
"""
from vidsitu_code.dat_loader import VsituDS, BatchCollator
from yacs.config import CfgNode as CN
from typing import Dict
from munch import Munch
from pathlib import Path
from utils.dat_utils import read_file_with_assertion, get_dataloader
from vidsitu_code.extended_config import CfgProcessor
import fire
from vidsitu_code.mdl_selector import get_mdl_loss_eval
import torch
from tqdm import tqdm
import numpy as np
from utils.trn_utils import move_to
from slowfast.utils.checkpoint import load_checkpoint


class VsituDS_All(VsituDS):
    def __init__(self, cfg: CN, comm: Dict, split_type: str):
        self.full_cfg = cfg
        self.cfg = cfg.ds.vsitu
        self.sf_cfg = cfg.sf_mdl
        self.task_type = self.full_cfg.task_type
        self.split_type = split_type
        # self.mdl_cfg = cfg.
        self.comm = Munch(comm)

        if len(comm) == 0:
            self.set_comm_args()

        self.read_files(split_type=split_type)
        self.itemgetter = getattr(self, "all_itemgetter")

    def set_comm_args(self):
        frm_seq_len = self.sf_cfg.DATA.NUM_FRAMES * self.sf_cfg.DATA.SAMPLING_RATE
        fps = self.sf_cfg.DATA.TARGET_FPS
        cent_frm_per_ev = {f"Ev{ix+1}": int((ix + 1 / 2) * fps * 2) for ix in range(5)}

        self.comm.num_frms = self.sf_cfg.DATA.NUM_FRAMES
        self.comm.sampling_rate = self.sf_cfg.DATA.SAMPLING_RATE
        self.comm.frm_seq_len = frm_seq_len
        self.comm.fps = fps
        self.comm.cent_frm_per_ev = cent_frm_per_ev
        self.comm.max_frms = 300
        self.comm.vb_id_vocab = read_file_with_assertion(
            self.cfg.vocab_files.verb_id_vocab, reader="pickle"
        )

        if self.sf_cfg.MODEL.ARCH in self.sf_cfg.MODEL.MULTI_PATHWAY_ARCH:
            self.comm.path_type = "multi"
        elif self.sf_cfg.MODEL.ARCH in self.sf_cfg.MODEL.SINGLE_PATHWAY_ARCH:
            self.comm.path_type = "single"
        else:
            raise NotImplementedError

    def read_files(self, split_type: str):
        self.vsitu_frm_dir = Path(self.cfg.video_frms_tdir)
        vseg_lst = read_file_with_assertion(self.cfg.split_files_lb[split_type])
        self.vseg_lst = vseg_lst

    def __getitem__(self, index: int) -> Dict:
        return self.itemgetter(index)

    def all_itemgetter(self, idx):
        frms_out_dct = self.get_frms_all(idx)
        frms_out_dct["vseg_idx"] = torch.tensor(idx).long()
        return frms_out_dct

    def __len__(self) -> int:
        if self.full_cfg.debug_mode:
            return 30
        return len(self.vseg_lst)


class FeatExtract:
    def __init__(self, cfg: CN):
        self.cfg = cfg

    def set_mdl_dl(self, mdl, dl, mdl_name: str, split_name: str):
        self.mdl = mdl
        self.dl = dl
        self.mdl_name = mdl_name
        self.split_name = split_name
        out_tdir = Path(self.cfg.ds.vsitu.vsitu_frm_feats) / f"{mdl_name}"
        out_tdir.mkdir(exist_ok=True)
        self.out_tdir = out_tdir

    @torch.no_grad()
    def forward_all(self):
        vseg_lst = self.dl.dataset.vseg_lst
        for batch in tqdm(self.dl):
            batch_gpu = move_to(batch, torch.device("cuda"))
            feat_out = self.mdl.forward_encoder(batch_gpu)
            head_out = self.mdl.head(feat_out)
            # (B, C, T, H, W) -> (B, T, H, W, C).
            head_out = head_out.permute((0, 2, 3, 4, 1))
            B = len(batch["vseg_idx"])
            assert head_out.size(1) == 1
            assert head_out.size(2) == 1
            assert head_out.size(3) == 1
            out = head_out.view(B, 5, -1)
            out_np = out.cpu().numpy()

            for vix in range(B):
                vseg_ix = batch["vseg_idx"][vix]
                vseg_name = vseg_lst[vseg_ix]
                out_np_one = out_np[vix]
                out_np_name = self.out_tdir / f"{vseg_name}_feats.npy"
                np.save(out_np_name, out_np_one)
        return


def rem_mdl(k):
    k1 = k.split("module.", 1)[1]
    return k1


def main(mdl_resume_path: str, mdl_name_used: str, is_cu: bool = False, **kwargs):
    CFP = CfgProcessor("./configs/vsitu_cfg.yml")
    cfg = CFP.get_vsitu_default_cfg()
    num_gpus = 1
    cfg.num_gpus = num_gpus
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
    comm = None
    feat_ext = FeatExtract(cfg)
    mdl_loss_eval = get_mdl_loss_eval(cfg)
    get_default_net = mdl_loss_eval["mdl"]

    # for split_type in ["train", "valid", "test"]:
    ds = VsituDS_All(cfg, {}, split_type="train")
    comm = ds.comm
    mdl = get_default_net(cfg=cfg, comm=comm)
    if not is_cu:
        mdl.load_state_dict(
            {
                rem_mdl(k): v
                for k, v in torch.load(mdl_resume_path)["model_state_dict"].items()
            }
        )
    else:
        print("Using Caffe2 checkpoint")
        load_checkpoint(
            mdl_resume_path,
            model=mdl.sf_mdl,
            data_parallel=False,
            convert_from_caffe2=True,
        )

    mdl.to(torch.device("cuda"))

    for split_type in ["valid", "train", "test_verb", "test_srl", "test_evrel"]:
        if comm is None:
            comm = {}
        ds = VsituDS_All(cfg, comm, split_type=split_type)
        comm = ds.comm
        batch_collator = BatchCollator(cfg, ds.comm)
        dl = get_dataloader(cfg, ds, is_train=False, collate_fn=batch_collator)

        mdl.eval()
        feat_ext.set_mdl_dl(mdl, dl, mdl_name=mdl_name_used, split_name=split_type)
        feat_ext.forward_all()


if __name__ == "__main__":
    fire.Fire(main)
