from pathlib import Path
import torch
import re
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from typing import Dict, Optional, Union, List, Tuple

# from fairseq.data import Dictionary
import numpy as np
import json
import pickle


@dataclass
class DataWrap:
    path: Union[str, Path]
    train_dl: DataLoader
    valid_dl: DataLoader
    test_dl: Optional[Union[DataLoader, Dict]] = None


def make_data_sampler(dataset: Dataset, shuffle: bool, distributed: bool) -> Sampler:
    if distributed:
        # return NewDistributedSampler(dataset, shuffle=shuffle)
        return DistributedSampler(dataset=dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def get_dataloader(cfg, dataset: Dataset, is_train: bool, collate_fn) -> DataLoader:
    is_distributed = cfg.do_dist
    batch_size_inp = cfg.train.bs if is_train else cfg.train.bsv
    nw = cfg.train.nw if is_train else cfg.train.nwv
    if is_distributed:
        # DistributedDataParallel
        assert batch_size_inp % cfg.num_gpus == 0
        batch_size = batch_size_inp // cfg.num_gpus
        num_workers = nw
    elif cfg.do_dp:
        # DataParallel
        batch_size = batch_size_inp * cfg.num_gpus
        num_workers = nw * cfg.num_gpus
    else:
        batch_size = batch_size_inp
        num_workers = nw

    if is_train:
        shuffle = True and cfg.ds.trn_shuffle
    else:
        shuffle = False
        # shuffle = False

    sampler = make_data_sampler(dataset, shuffle, is_distributed)

    collator = collate_fn

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=is_train,
        num_workers=num_workers,
        collate_fn=collator,
    )


def collate_dct_lst_naive(batch: List[Dict]):
    all_keys = list(batch[0].keys())
    out_dict = {}
    for k in all_keys:
        out_dict[k] = [b[k] for b in batch]
    return out_dict


def simple_collate_dct_list(
    batch: List[Dict], stack_or_cat: str = "stack", cat_dim: int = None
) -> Dict[str, List]:
    """
    Convert List[Dict[k, tensor]] -> Dict[k, Stacked Tensor]
    """
    assert stack_or_cat in ["stack", "cat"]
    if stack_or_cat == "cat":
        assert cat_dim is not None
    out_dict = {}
    # nothing needs to be done
    all_keys = list(batch[0].keys())
    if stack_or_cat == "stack":
        batch_size = len(batch)
    else:
        batch_size = len(batch) * batch[0][all_keys[0]].shape[0]
    for k in all_keys:
        shape = batch[0][k].shape
        if not all([b[k].shape == shape for b in batch]):
            raise NotImplementedError
            # ForkedPdb().set_trace()
        if stack_or_cat == "stack":
            out_dict[k] = torch.stack([b[k] for b in batch])
        elif stack_or_cat == "cat":
            out_dict[k] = torch.cat([b[k] for b in batch], cat_dim)
        else:
            raise NotImplementedError
    assert all([len(v) == batch_size for k, v in out_dict.items()])
    return out_dict


def coalesce_dicts(dct_list: List[Dict]) -> Dict:
    """
    Convert list of dicts with different keys
    to a single dict
    """
    out_dict = {}
    for dct in dct_list:
        for k in dct:
            if k in out_dict:
                assert torch.all(out_dict[k] == dct[k])

        out_dict.update(dct)
    return out_dict


def arg_mapper(arg_inp, argm_re=None):
    if argm_re is None:
        argm_re = re.compile(r"ArgM (.*)")
    arg_name = arg_inp.split(" ")[0]
    if arg_name in set(["Arg0", "Arg1", "Arg2", "Arg3", "Arg4", "Arg5"]):
        return arg_name
    elif arg_inp == "Scene of the Event":
        return "AScn"
    else:
        assert arg_name == "ArgM"
        y2 = argm_re.findall(arg_inp)[0].strip()
        if "direction" in y2:
            return "ADir"
        elif "purpose" in y2:
            return "APrp"
        elif "manner" in y2:
            return "AMnr"
        elif "location" in y2:
            return "ALoc"
        elif "goal" in y2:
            return "AGol"
        else:
            raise NotImplementedError


def truncate_batch(
    inp_dict: Dict[str, torch.tensor], key: str, max_len: int, dim: int
) -> Dict[str, torch.tensor]:
    """
    Truncate the value for the dictionary key
    with max len and wrt dim
    """
    assert len(inp_dict[key].shape) > dim
    if dim == 1:
        inp_dict[key] = inp_dict[key][:, :max_len].contiguous()
    elif dim == 2:
        inp_dict[key] = inp_dict[key][:, :, :max_len].contiguous()
    elif dim == 3:
        inp_dict[key] = inp_dict[key][:, :, :, :max_len].contiguous()
    else:
        raise NotImplementedError

    return


def pad_words(
    word_list: List, max_len: int, pad_index, eos_index=None, append_eos=False
) -> Tuple[List, int]:
    if append_eos:
        assert eos_index is not None
        cur_len = len(word_list)
        if cur_len >= max_len:
            return word_list[: max_len - 1] + [eos_index], max_len
        out_word_list = word_list + [eos_index] + [pad_index] * (max_len - 1 - cur_len)
        return out_word_list, cur_len + 1
    else:
        cur_len = len(word_list)
        if cur_len > max_len:
            return word_list[:max_len], max_len
        out_word_list = word_list + [pad_index] * (max_len - cur_len)
        return out_word_list, cur_len


def pad_tokens(
    lst: List[int],
    pad_index: int,
    pad_side: str,
    append_eos: bool,
    eos_index: int,
    max_len: int,
):
    curr_len = len(lst)
    if isinstance(lst, list):
        lst = torch.tensor(lst, dtype=torch.long)
    sent_out_enc = lst.new_full((max_len,), pad_index, dtype=torch.long)

    if append_eos:
        if curr_len >= max_len:
            sent_out_enc[:max_len] = lst[:max_len]
            sent_out_enc[max_len - 1] = eos_index
            out_len = max_len
        else:
            if pad_side == "right":
                sent_out_enc[:curr_len] = lst
            else:
                sent_out_enc[-curr_len:] = lst
            sent_out_enc[curr_len] = eos_index
            out_len = curr_len + 1
    else:
        if curr_len >= max_len:
            sent_out_enc[:max_len] = lst[:max_len]
            out_len = max_len
        else:
            if pad_side == "right":
                sent_out_enc[:curr_len] = lst
            else:
                sent_out_enc[-curr_len:] = lst
            out_len = curr_len
    if pad_side == "right":
        attn_mask = [1] * out_len + [0] * (max_len - out_len)
    else:
        attn_mask = [0] * (max_len - out_len) + [1] * out_len
    assert len(attn_mask) == max_len
    return sent_out_enc, attn_mask


def pad_words_new(
    sent: str,
    max_len: int,
    wvoc,
    append_eos=False,
    use_hf: bool = False,
    pad_side: str = "right",
    prefix_lst: List[int] = None,
) -> Tuple[List, int]:
    assert pad_side in ["left", "right"]
    if use_hf:
        sent_enc = wvoc(sent)["input_ids"]
        pad_index = wvoc.pad_token_id
        eos_index = wvoc.eos_token_id
    else:
        sent_enc = wvoc.encode_line(sent, add_if_not_exist=False, append_eos=False)
        pad_index = wvoc.pad_index
        eos_index = wvoc.eos_index
    if prefix_lst is not None:
        sent_enc = prefix_lst + sent_enc
    sent_out_enc, attn_mask = pad_tokens(
        sent_enc,
        pad_index=pad_index,
        pad_side=pad_side,
        append_eos=append_eos,
        eos_index=eos_index,
        max_len=max_len,
    )
    return sent_out_enc, attn_mask


def add_prev_tokens(
    inp_dict: Dict[str, torch.tensor], key: str, pad_token: int, bos_token: int
) -> Dict[str, torch.tensor]:
    """
    Create prev tokens for the given dictionary key
    """
    src_toks = inp_dict[key]
    # prev_output_tokens = src_toks.new_full(src_toks.shape, fill_value=pad_token)
    # prev_output_tokens[..., 0] = bos_token
    # prev_output_tokens[..., 1:] = src_toks[..., :-1].clone()
    prev_output_tokens = add_prev_tokens_tensor(
        src_tensor=src_toks, pad_token=pad_token, bos_token=bos_token
    )
    out_key = f"prev_out_{key}"
    inp_dict[out_key] = prev_output_tokens
    return


def add_prev_tokens_tensor(
    src_tensor: torch.tensor, pad_token: int, bos_token: int
) -> torch.tensor:
    """
    Create prev tokens for the given dictionary key
    """
    prev_output_tokens = src_tensor.new_full(src_tensor.shape, fill_value=pad_token)
    prev_output_tokens[..., 0] = bos_token
    prev_output_tokens[..., 1:] = src_tensor[..., :-1].clone()
    return prev_output_tokens


def read_file_with_assertion(fpath: str, read_type: str = "r", reader: str = "json"):
    fpath1 = Path(fpath)
    if read_type == "r":
        assert fpath1.exists(), f"{fpath1} doesn't exist"
        if reader == "json":
            with open(fpath1, "r") as f:
                file_data = json.load(f)
            return file_data
        elif reader == "pickle":
            with open(fpath1, "rb") as f:
                file_data = pickle.load(f)
            return file_data
        elif reader == "numpy":
            return np.load(fpath1)
    elif read_type == "w":
        assert fpath1.parent.exists()
    else:
        raise NotImplementedError
