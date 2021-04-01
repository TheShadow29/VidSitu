"""
Download Youtube
"""

from tqdm import tqdm
from pathlib import Path
from utils.dat_utils import read_file_with_assertion
from yacs.config import CfgNode as CN
import subprocess
import os
import fire


class YTDown:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_all_yt_ids(self, split_dir: str):
        split_dir = Path(split_dir)
        assert split_dir.exists()
        self.split_data = {
            "train_split": read_file_with_assertion(
                split_dir / "vseg_split_train_lb.json"
            ),
            "val_split": read_file_with_assertion(
                split_dir / "vseg_split_valid_lb.json"
            ),
            "test_vb_split": read_file_with_assertion(
                split_dir / "vseg_split_testvb_lb.json"
            ),
            "test_srl_split": read_file_with_assertion(
                split_dir / "vseg_split_testsrl_lb.json"
            ),
            "test_evrel_split": read_file_with_assertion(
                split_dir / "vseg_split_testevrel_lb.json"
            ),
        }

        def proc_vname(vname):
            vid_st_en = vname.split("v_", 1)[1]
            vid_id, ste = vid_st_en.rsplit("_seg_", 1)
            st, en = ste.split("_")
            return {
                "vid_seg_id": vname,
                "vid_id": vid_id,
                "start": int(st),
                "end": int(en),
            }

        self.combined_split = [
            proc_vname(x) for y in list(self.split_data.values()) for x in y
        ]

        return

    def download_yt_vids(self):
        video_dir = Path(self.cfg.video_trimmed_dir)
        assert video_dir.parent.exists()
        video_dir.mkdir(exist_ok=True)

        processes = set()
        max_process = self.cfg.max_process

        for yt_id in tqdm(self.combined_split[:10]):
            cmd = f"ffmpeg -ss {yt_id['start']} -i $(youtube-dl -f 22 --get-url https://www.youtube.com/watch?v={yt_id['vid_id']}) -to 10 {video_dir / yt_id['vid_seg_id']}.mp4"
            processes.add(subprocess.Popen(cmd, shell=True))
            if len(processes) >= max_process:
                os.wait()
                processes.difference_update(
                    [p for p in processes if p.poll() is not None]
                )

    def extract_frames_fast(self):
        in_dir = Path(self.cfg.video_trimmed_dir)
        assert in_dir.exists()

        out_dir = Path(self.cfg.video_frm_dir)
        assert out_dir.parent.exists()
        out_dir.mkdir(exist_ok=True)

        processes = set()
        max_process = self.cfg.max_process

        in_dir_files = [y for y in in_dir.iterdir() if y.suffix == ".mp4"]

        for in_file in tqdm(in_dir_files):
            out_vid_dir = out_dir / f"{in_file.stem}"
            if out_vid_dir.exists():
                continue
            else:
                out_vid_dir.mkdir()

            out_name = str(out_vid_dir / f"{in_file.stem}_%06d.jpg")
            cmd = f"ffmpeg -i {in_file} -r 30 -q:v 1 {out_name}"
            processes.add(subprocess.Popen(cmd, shell=True))
            if len(processes) >= max_process:
                os.wait()
                processes.difference_update(
                    [p for p in processes if p.poll() is not None]
                )

        return


def main(task_type: str):
    cfg = CN(
        {
            "video_trimmed_dir": "./data/video_trimmed_dir",
            "video_frm_dir": "./data/vsitu_video_frames_dir",
            "max_process": 10,
            "split_dir": "./data/vidsitu_data/vidsitu_s3_upload/split_files",
        }
    )

    ytd = YTDown(cfg)
    ytd.get_all_yt_ids(cfg.split_dir)

    if task_type == "dwn_vids":
        ytd.download_yt_vids()
    elif task_type == "extract_frames":
        ytd.extract_frames_fast()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    fire.Fire(main)
