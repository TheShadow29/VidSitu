"""
Download Youtube
"""

from tqdm import tqdm
from pathlib import Path
from yacs.config import CfgNode as CN
from typing import Generator
import subprocess
import yaml
import os
import logging
import json
import argparse

logging.basicConfig(
    filename="./prep_data/problematic_vidsegs.txt",
    filemode="w",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)
import signal
import shutil

logger = logging.getLogger(__name__)


def read_file_with_assertion(fpath: str, read_type: str = "r", reader: str = "json"):
    fpath1 = Path(fpath)
    if read_type == "r":
        assert fpath1.exists(), f"{fpath1} doesn't exist"
        if reader == "json":
            with open(fpath1, "r") as f:
                file_data = json.load(f)
            return file_data
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def process_wrapper(
    iter_gen: Generator, max_processes: int, total=None, suppress_stdout: bool = False
):
    """Parallelizes the process
    Parameters
    ----------
    iter_gen : Generator
        Should output a dict with cmd key which contains the command to be executed
    max_processes : int
        Max number of processes to use
    """
    try:
        processes = set()
        std_out = None
        std_err = None
        if suppress_stdout:
            std_out = subprocess.PIPE
            std_err = subprocess.PIPE

        for elm in tqdm(iter_gen, total=total):
            cmd = elm["cmd"]
            processes.add(
                subprocess.Popen(
                    cmd,
                    shell=True,
                    preexec_fn=os.setsid,
                    stdout=std_out,
                    stderr=std_err,
                )
            )
            if len(processes) >= max_processes:
                os.wait()
                processes.difference_update(
                    [p for p in processes if p.poll() is not None]
                )
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        for p in processes:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    finally:
        for p in processes:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        print("Loop DONE")
    return


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
            (k, proc_vname(x))
            for k in self.split_data
            for x in list(self.split_data[k])
        ]

        return

    def download_yt_vids(self):
        if self.cfg.cookies_file == "":
            ytdl_cookies_str = ""
        else:
            assert Path(self.cfg.cookies_file).exists()
            ytdl_cookies_str = f"--cookies {self.cfg.cookies_file}"

        video_dir = Path(self.cfg.video_trimmed_dir)
        assert video_dir.parent.exists()
        video_dir.mkdir(exist_ok=True)

        max_process = self.cfg.max_processes

        def get_out_file(vid_seg_id):
            return Path(f"{video_dir / vid_seg_id}.mp4")

        def get_generator(remaining_ids, format=None) -> Generator:
            for _, yt_id in remaining_ids:
                out_file = get_out_file(yt_id["vid_seg_id"])
                if format is None:
                    format = "22/best"
                cmd = (
                    f"ffmpeg -ss {yt_id['start']} -i $(yt-dlp {ytdl_cookies_str} -f {format}"
                    + f" --get-url https://www.youtube.com/watch\?v\={yt_id['vid_id']}) -to 10 {out_file}"
                )
                yield {"cmd": cmd}

        retry_count = self.cfg.retry_count
        orig_retry_count = retry_count

        def check_exists(fpath, strict_check: bool = False):
            if not fpath.exists():
                return False
            if strict_check:
                # FILE SHOULD BE > 50K
                # If Images have already been extracted, they should be > 290
                if fpath.stat().st_size < 50000:
                    fpath.unlink()
                    return False
                fdir = Path(self.cfg.video_frm_tdir) / f"{fpath.stem}"
                if fdir.exists():
                    nfiles = len([i for i in fdir.iterdir()])
                    if nfiles < 290:
                        fpath.unlink()
                        return False

                return True
            return True

        while retry_count >= 0:

            remaining_vid_seg_ids = [
                x
                for x in self.combined_split
                if not check_exists(
                    get_out_file(x[1]["vid_seg_id"]), strict_check=self.cfg.hard_check
                )
            ]

            if len(remaining_vid_seg_ids) == 0:
                break

            print("Retry Count", orig_retry_count - retry_count)
            format = None
            # ONLY Try at last retry
            if retry_count == 0:
                format = "webm"
            remaining_vid_gen = get_generator(
                remaining_ids=remaining_vid_seg_ids, format=format
            )

            process_wrapper(
                iter_gen=remaining_vid_gen,
                max_processes=max_process,
                total=len(remaining_vid_seg_ids),
                suppress_stdout=self.cfg.suppress_ffmpeg_outputs,
            )
            retry_count -= 1

        not_found = 0
        for split_name, yt_id in tqdm(self.combined_split):
            vid_seg_id = yt_id["vid_seg_id"]
            out_file = get_out_file(vid_seg_id)
            if not out_file.exists():
                logger.info(
                    f"File {vid_seg_id}.mp4 from {split_name} not found in expected location {out_file}"
                )
                not_found += 1

        out_str = f"Not Found {not_found} of total {len(self.combined_split)}"
        print(out_str)
        logger.info(out_str)

        print("Exit")

    def extract_frames_fast(self):
        in_dir = Path(self.cfg.video_trimmed_dir)
        assert in_dir.exists()

        out_dir = Path(self.cfg.video_frm_tdir)
        assert out_dir.parent.exists()
        out_dir.mkdir(exist_ok=True)
        max_process = self.cfg.max_processes

        in_dir_files = [y for y in in_dir.iterdir() if y.suffix == ".mp4"]

        def get_out_vid_dir(vid_seg_id):
            return out_dir / f"{vid_seg_id}"

        def get_in_file_gen(in_dir_files) -> Generator:
            for in_file in in_dir_files:
                vid_seg_id = in_file.stem.replace("_trimmed", "")
                out_vid_dir = get_out_vid_dir(vid_seg_id)
                out_vid_dir.mkdir()
                out_name = str(out_vid_dir / f"{vid_seg_id}_%06d.jpg")
                cmd = f"ffmpeg -i {in_file} -r 30 -q:v 1 {out_name}"
                yield {"cmd": cmd}

        def check_exist(dir_path, strict=False):
            if not dir_path.exists():
                return False
            if strict:
                nfiles = len([i for i in dir_path.iterdir()])
                if nfiles < 290:
                    shutil.rmtree(dir_path)
                    return False
                return True
            return True

        remaining_in_dir_files = [
            y
            for y in in_dir_files
            if not check_exist(
                get_out_vid_dir(y.stem.replace("_trimmed", "")),
                strict=self.cfg.hard_check,
            )
        ]

        file_gen = get_in_file_gen(in_dir_files=remaining_in_dir_files)

        process_wrapper(
            file_gen,
            max_process,
            total=len(remaining_in_dir_files),
            suppress_stdout=self.cfg.suppress_ffmpeg_outputs,
        )
        return


def main(task_type: str, **kwargs):

    cfg = CN(yaml.safe_load(open("./configs/vsitu_setup_cfg.yml")))
    cfg.update(**kwargs)

    ytd = YTDown(cfg)
    ytd.get_all_yt_ids(cfg.split_dir)

    if task_type == "dwn_vids":
        ytd.download_yt_vids()
    elif task_type == "extract_frames":
        ytd.extract_frames_fast()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    # fire.Fire(main)
    parser = argparse.ArgumentParser(
        description="Specify Download Videos or Extract Frames"
    )
    parser.add_argument(
        "--task_type", help="Task Type is either `dwn_vids` or `extract_frames`"
    )
    parser.add_argument(
        "--max_processes",
        type=int,
        help="Max number of parallel processes to run.",
        default=10,
    )
    parser.add_argument("--cookies_file", help="Cookies for YTDL", default="")

    parser.add_argument("--hard_check", action="store_true", help="check strict")

    parser.add_argument(
        "--retry_count",
        type=int,
        help="How many times to retry youtube-download. NOTE: Should be >= 1, recommended 3",
        default=3,
    )
    parser.add_argument(
        "--suppress_ffmpeg_outputs",
        action="store_true",
        help="Dont print ffmpeg logs.",
    )

    args_inp = parser.parse_args()
    main(**vars(args_inp))
