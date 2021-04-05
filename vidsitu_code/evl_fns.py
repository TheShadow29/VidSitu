"""
Evaluating IVD
Use eval metrics from Pycocoevalcap
"""
import fire
import pickle
from typing import Dict, List
import numpy as np
from collections import namedtuple, Counter
import re
import yaml
from yacs.config import CfgNode as CN
from coval.conll.reader import get_mention_assignments
from coval.eval import evaluator
from coval.eval.evaluator import Evaluator
from pathlib import Path
import json
import sys

sys.path.insert(0, "./coco-caption")


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


def collate_dct_lst_naive(batch: List[Dict]):
    all_keys = list(batch[0].keys())
    out_dict = {}
    for k in all_keys:
        out_dict[k] = [b[k] for b in batch]
    return out_dict


def get_avg(lst):
    if len(lst) == 0:
        return 0
    return sum(lst) / len(lst)


def remove_nonascii(text):
    return "".join([i if ord(i) < 128 else " " for i in text])


def read_gt_file(full_cfg, task_type, split_type):
    ds_cfg = full_cfg.ds.vsitu
    split_files_cfg = ds_cfg.split_files_lb
    vsitu_ann_files_cfg = ds_cfg.vsitu_ann_files_lb

    vinfo_files_cfg = ds_cfg.vinfo_files_lb

    # ann_idx is with respect to this.
    vseg_lst = read_file_with_assertion(split_files_cfg[split_type])

    vseg_ann_lst = read_file_with_assertion(vsitu_ann_files_cfg[split_type])

    vsitu_ann_dct = {}
    for vseg_ann in vseg_ann_lst:
        vseg = vseg_ann["Ev1"]["vid_seg_int"]
        if vseg not in vsitu_ann_dct:
            vsitu_ann_dct[vseg] = []
        vsitu_ann_dct[vseg].append(vseg_ann)

    out_dct = {
        "vseg_lst": vseg_lst,
        "vsitu_ann_dct": vsitu_ann_dct,
    }
    if task_type == "vb":
        assert "valid" in split_type or "test" in split_type
        vseg_info_lst = read_file_with_assertion(vinfo_files_cfg[split_type])
        vsitu_vinfo_dct = {}
        for vseg_info in vseg_info_lst:
            vseg = vseg_info["vid_seg_int"]
            assert vseg not in vsitu_vinfo_dct
            assert len(vseg_info["vbid_lst"]["Ev1"]) >= 9
            vid_seg_ann_lst = [vseg_info["vbid_lst"][f"Ev{eix}"] for eix in range(1, 6)]
            vseg_info["vb_id_lst_eval"] = vid_seg_ann_lst
            vsitu_vinfo_dct[vseg] = vseg_info

        out_dct["vsitu_vinfo_dct"] = vsitu_vinfo_dct
    elif task_type == "vb_arg":
        pass
    elif task_type == "evrel":
        pass
    else:
        raise NotImplementedError

    return out_dct


class EvlFn_EvRel:
    def __init__(
        self, cfg, comm, met_keys, read_val_file: bool = False, get_gt_dct: bool = False
    ):
        self.cfg = cfg
        self.comm = comm
        self.met_keys = met_keys

    def read_gt_file(self, split_type):
        files_out = read_gt_file(self.cfg, "evrel", split_type=split_type)
        self.vseg_lst = files_out["vseg_lst"]
        vsitu_ann_dct = files_out["vsitu_ann_dct"]
        self.gts_dct = {
            ix: vsitu_ann_dct[self.vseg_lst[ix]] for ix in range(len(self.vseg_lst))
        }

    def simple_acc_evrel(self, pred_file: str, split_type: str = "valid"):
        hypos_gts_mask = self.prepare_hyp_gts(
            pred_file=pred_file, split_type=split_type
        )
        hypos = hypos_gts_mask["hypos"]
        gts = hypos_gts_mask["gts"]
        mask = hypos_gts_mask["mask"]
        hypos_ids = sorted(list(hypos.keys()))
        corr_lst = []
        gt_corr_lst = []
        msk_lst = []
        for hid in hypos_ids:
            hyp = hypos[hid]
            gt1 = gts[hid]
            msk1 = mask[hid]

            for ev_ix in [1, 2, 4, 5]:
                hyp_evi = hyp[f"Ev{ev_ix}"]
                gt_evi = gt1[f"Ev{ev_ix}"]
                msk_evi = msk1[f"Ev{ev_ix}"]
                assert len(hyp_evi) == len(gt_evi)

                gt_max = Counter(gt_evi).most_common()[0][0]
                gt_evi_ix = [i for i in range(len(gt_evi)) if gt_evi[i] == gt_max]
                hyp_evii = [hyp_evi[i] for i in gt_evi_ix]
                gt_evii = [gt_evi[i] for i in gt_evi_ix]
                for h, g in zip(hyp_evii, gt_evii):
                    corr_lst.append(h == g)
                    gt_corr_lst.append(g)
                    msk_lst.append(msk_evi)
        out_corr_lst = []
        assert len(msk_lst) == len(corr_lst)
        for cor1, cor_msk1 in zip(corr_lst, msk_lst):
            if cor_msk1:
                out_corr_lst.append(cor1)
        mac_dct = {}
        for gix, g in enumerate(gt_corr_lst):
            if g not in mac_dct:
                mac_dct[g] = []
            if msk_lst[gix]:
                mac_dct[g].append(corr_lst[gix])
        mac_dct2 = {k: sum(v) / len(v) for k, v in mac_dct.items()}

        return {
            "Top_1": sum(out_corr_lst) / len(out_corr_lst),
            "Len": len(out_corr_lst),
            "Macro_Top_1": sum(mac_dct2.values()) / len(mac_dct2),
            "Macro_Top_Dct": mac_dct2,
        }

    def prepare_hyp_gts(self, pred_file: str, split_type: str = "valid"):
        with open(pred_file, "rb") as f:
            pred_data = pickle.load(f)

        self.read_gt_file(split_type=split_type)

        hypo_dct = {}
        for pred in pred_data:
            ann_idx = pred["ann_idx"]
            # assert ann_idx not in hypo_dct
            if ann_idx not in hypo_dct:
                hypo_dct[ann_idx] = pred

        hypos = {}
        gts = {}
        mask = {}
        ev_lst = [f"Ev{ix}" for ix in [1, 2, 4, 5]]

        if self.cfg.debug_mode:
            pass
        else:
            assert len(hypo_dct) == len(self.vseg_lst), "Missing Elements in Prediction"

        for ann_idx in hypo_dct:
            if ann_idx not in hypos:
                pred_one = hypo_dct[ann_idx]
                preds = pred_one["pred_evrels_ev"]

                gt_vbs_lst = self.gts_dct[pred_one["ann_idx"]]
                gt_vbs = [
                    [gt_i[f"Ev{ev_i}"]["EvRel"] for gt_i in gt_vbs_lst]
                    for ev_i in [1, 2, 4, 5]
                ]

                hypos[pred_one["ann_idx"]] = {
                    ev_i: preds[ev_ix] for ev_ix, ev_i in enumerate(ev_lst)
                }
                gts[pred_one["ann_idx"]] = {
                    ev_i: gt_vbs[ev_ix][:3] for ev_ix, ev_i in enumerate(ev_lst)
                }

                mask[pred_one["ann_idx"]] = {
                    ev_i: 1
                    if Counter(gt_vbs[ev_ix][:3]).most_common()[0][1] >= 2
                    else 0
                    for ev_ix, ev_i in enumerate(ev_lst)
                }

        return {"hypos": hypos, "gts": gts, "mask": mask}


class EvlFn_Vb:
    def __init__(self, cfg, comm, met_keys):
        self.cfg = cfg
        self.comm = comm
        self.met_keys = met_keys

        evix_lst = [eix for eix in range(1, 6)]
        evlst = [f"Ev{eix}" for eix in evix_lst]
        self.evlst = evlst
        self.evix_lst = evix_lst

    def read_gt_file(self, split_type):
        files_out = read_gt_file(self.cfg, task_type="vb", split_type=split_type)

        self.vseg_lst = files_out["vseg_lst"]
        self.vsitu_ann_dct = files_out["vsitu_ann_dct"]
        self.vsitu_vinfo_dct = files_out["vsitu_vinfo_dct"]
        return

    def vb_classf_metrics_all(self, hyps: Dict, gts: Dict):
        """
        Assumes hyps, gts dicts with keys as video ids (10-sec)
        """
        assert set(hyps.keys()) == set(gts.keys())
        vid_key_lst = sorted(list(hyps.keys()))
        ev_lst = [f"Ev{ix}" for ix in self.evix_lst]
        corr_dct = {f"Top_{k}": [] for k in range(1, 6)}
        corr_dct_by_vid = {f"Top_{k}": [] for k in range(1, 6)}
        corr_dct_by_vb = {}
        corr_dct_by_vb_wt = {}

        for vid_key in vid_key_lst:
            hypos1 = hyps[vid_key]
            gts1 = gts[vid_key]
            assert len(hypos1) == len(ev_lst)
            assert len(gts1) == len(ev_lst)
            corr_ev_lst = {f"Top_{k}": [] for k in range(1, 6)}
            for ev_i in ev_lst:
                hy1 = hypos1[ev_i]
                gt1 = gts1[ev_i]
                for topk in range(1, 6):
                    corr_one = int(len(set(hy1[:topk]).intersection(gt1)) > 0)
                    corr_dct[f"Top_{topk}"].append(corr_one)
                    corr_ev_lst[f"Top_{topk}"].append(corr_one)
                gt1_counts = [y for y in Counter(gt1).most_common() if y[1] >= 2]
                for gtvb, gtvc in gt1_counts:
                    if gtvb not in corr_dct_by_vb:
                        corr_dct_by_vb[gtvb] = []
                    if gtvb not in corr_dct_by_vb_wt:
                        corr_dct_by_vb_wt[gtvb] = []
                    if gtvb in set(hy1):
                        corr_dct_by_vb[gtvb].append(1)
                        corr_dct_by_vb_wt[gtvb].append((gtvc, gtvc, len(gt1)))
                    else:
                        corr_dct_by_vb[gtvb].append(0)
                        corr_dct_by_vb_wt[gtvb].append((0, gtvc, len(gt1)))
            for topk in range(1, 6):
                corr_dct_by_vid[f"Top_{topk}"].append(
                    int(all([y == 1 for y in corr_ev_lst[f"Top_{topk}"]]))
                )
        out_dct = {}

        for k in corr_dct:
            out_dct[f"Per_Ev_{k}"] = get_avg(corr_dct[k])
        for k in corr_dct_by_vid:
            out_dct[f"Per_Vid_{k}"] = get_avg(corr_dct_by_vid[k])
        out_dct["acc"] = out_dct["Per_Ev_Top_5"]
        corr_lst_by_vb = sorted(
            [(k, get_avg(v), len(v)) for k, v in corr_dct_by_vb.items()],
            key=lambda x: x[1],
            reverse=True,
        )
        for thresh in range(0, 10):
            lst_thresh = [y[1] for y in corr_lst_by_vb if y[2] > thresh]
            out_dct[f"recall_macro_1_th_{thresh}"] = get_avg(lst_thresh)
            out_dct[f"num_vbs_thresh_{thresh}"] = len(lst_thresh)
        # out_dct["acc"] = out_dct["recall_macro_1_th_5"]
        return out_dct
        # return (corr_dct, corr_dct_by_vb, corr_dct_by_vb_wt, corr_dct_by_vid)

    def prepare_hyp_gts(self, pred_file: str, split_type: str = "valid"):
        with open(pred_file, "rb") as f:
            pred_data = pickle.load(f)

        self.read_gt_file(split_type=split_type)

        hypo_dct = {}
        for pred in pred_data:
            ann_idx = pred["ann_idx"]
            # assert ann_idx not in hypo_dct
            if ann_idx not in hypo_dct:
                hypo_dct[ann_idx] = pred

        hypos = {}
        gts = {}

        ev_lst = [f"Ev{ix}" for ix in self.evix_lst]
        if self.cfg.debug_mode:
            pass
        else:
            assert len(hypo_dct) == len(self.vseg_lst), "Missing Elements in Prediction"
        # for pix, pred_one in enumerate(pred_data):
        for ann_idx in hypo_dct:
            if ann_idx not in hypos:
                pred_one = hypo_dct[ann_idx]
                preds = pred_one["pred_vbs_ev"]
                vseg_name = self.vseg_lst[pred_one["ann_idx"]]
                gt_vbs = self.vsitu_vinfo_dct[vseg_name]["vb_id_lst_eval"]

                hypos[pred_one["ann_idx"]] = {
                    ev_i: preds[ev_ix][:5] for ev_ix, ev_i in enumerate(ev_lst)
                }

                gts[pred_one["ann_idx"]] = {
                    ev_i: gt_vbs[ev_ix][:10] for ev_ix, ev_i in enumerate(ev_lst)
                }

        return hypos, gts

    def simple_acc(self, pred_file: str, split_type: str = "valid"):
        hypos, gts = self.prepare_hyp_gts(pred_file=pred_file, split_type=split_type)

        out_dct = self.vb_classf_metrics_all(hyps=hypos, gts=gts)
        return out_dct


class EvalFnCap:
    def __init__(self, cfg, comm, met_keys, read_val_file: bool = True):
        self.cfg = cfg
        self.comm = comm
        self.met_keys = met_keys
        self.get_scorers()
        self.scorers = {}
        self.args_used = ["Arg0", "Arg1", "Arg2", "ALoc", "AScn"]
        self.ngt = 3
        ScorerE = namedtuple("ScorerE", ["fn", "out_str"])
        for k in self.met_keys:
            scorer_tuple = self.scorer_dict[k]
            if scorer_tuple.to_init:
                scorer = scorer_tuple.cls_fn()
            else:
                scorer = scorer_tuple.cls_fn
            self.scorers[k] = ScorerE(scorer, scorer_tuple.out_str)

    def read_gt_file(self, split_type):
        files_out = read_gt_file(self.cfg, "vb_arg", split_type=split_type)

        self.vseg_lst = files_out["vseg_lst"]
        vsitu_ann_dct = files_out["vsitu_ann_dct"]

        self.gts_dct = {
            ix: vsitu_ann_dct[self.vseg_lst[ix]] for ix in range(len(self.vseg_lst))
        }
        np.random.seed(5)
        self.gts_dct = {
            ix: [v[rix] for rix in np.random.permutation(len(v))]
            for ix, v in self.gts_dct.items()
        }

        return

    def get_scorers(self):
        # from pycoco_scorers_vizseq import BLEUScorerAll
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge

        from pycocoevalcap.meteor.meteor import Meteor

        # from pycocoevalcap.spice.spice import Spice

        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

        Scorer_ = namedtuple("Scorer_", ["cls_fn", "to_init", "out_str"])
        self.scorer_dict = {
            "bleu": Scorer_(
                Bleu(4, verbose=0), False, ["bleu_1", "bleu_2", "bleu_3", "bleu_4"]
            ),
            "meteor": Scorer_(Meteor(), False, ["meteor"]),
            "cider": Scorer_(Cider("corpus"), False, ["cider"]),
            "rouge": Scorer_(Rouge(), False, ["rouge"]),
            # "spice": Scorer_(Spice(), False, ["spice"]),
        }
        self.tokenizer = PTBTokenizer()

        self.coval_all_metrics = [
            ("mentions", evaluator.mentions),
            ("muc", evaluator.muc),
            ("bcub", evaluator.b_cubed),
            ("ceafe", evaluator.ceafe),
            ("lea", evaluator.lea),
            ("lea_soft", evaluator.lea_soft),
        ]
        self.reset_coval_scorer_dict()

    def reset_coval_scorer_dict(self):
        self.coval_scorer_dict = {}
        for metric in self.coval_all_metrics:
            cov_met_name = f"{metric[0]}"
            self.coval_scorer_dict[cov_met_name] = Evaluator(metric[1])

        return

    def prepare_hyp_gts(
        self, pred_file: str, split_type: str = "valid", ix_gt: int = 3
    ):
        ngt = self.ngt
        pred_outs = read_file_with_assertion(pred_file, reader="pickle")
        hypo_dct = {}
        for pred in pred_outs:
            ann_idx = pred["ann_idx"]
            if ann_idx not in hypo_dct:
                hypo_dct[ann_idx] = pred["vb_output"]

        if self.cfg.debug_mode:
            pass
        else:
            assert sorted(list(hypo_dct.keys())) == sorted(
                (list(self.gts_dct.keys()))
            ), "Missing Elements from Prediction"

        ann_idx_keys = sorted(list(hypo_dct.keys()))
        gto_dct = {
            an_ix: [y for yix, y in enumerate(self.gts_dct[an_ix]) if yix != ix_gt][
                :ngt
            ]
            for an_ix in ann_idx_keys
        }
        aix = 0

        hypo_str_dct = {}
        gts_str_dct = {}
        aix_vb_dct = {}
        aix_arg_dct = {}
        aix_encoder_dct = {}
        ev_lst = [f"Ev{eix}" for eix in range(1, 6)]
        for ann_idx in ann_idx_keys:
            hypo_vb_dct = hypo_dct[ann_idx]
            gt_vseg_assgns = [
                y for yix, y in enumerate(self.gts_dct[ann_idx]) if yix != ix_gt
            ][:ngt]
            for ev_i in ev_lst:
                gt_args = gt_vseg_assgns[0][ev_i]["Args"]
                vb_id = gt_vseg_assgns[0][ev_i]["VerbID"]
                for gt_ag in gt_args:
                    gt_ag_name = arg_mapper(gt_ag)
                    if gt_ag_name in self.args_used:

                        gt_lst = [gtva[ev_i]["Args"][gt_ag] for gtva in gt_vseg_assgns]
                        gts_str_dct[aix] = gt_lst
                        if ev_i in hypo_vb_dct and gt_ag_name in hypo_vb_dct[ev_i]:
                            hypo_lst = [hypo_vb_dct[ev_i][gt_ag_name]]
                        else:
                            hypo_lst = [""]
                        hypo_str_dct[aix] = hypo_lst
                        aix_vb_dct[aix] = vb_id
                        aix_arg_dct[aix] = gt_ag_name
                        aix_encoder_dct[aix] = {
                            "aix": aix,
                            "ann_idx": ann_idx,
                            "ev_ix": ev_i,
                            "agname": gt_ag_name,
                            "ev_agname": f"{ev_i}_{gt_ag_name}",
                            "agname_real": gt_ag,
                        }
                        aix += 1

        return {
            "hypos": hypo_str_dct,
            "gts": gts_str_dct,
            "hypos_orig": hypo_dct,
            "gts_orig": gto_dct,
            "ix_to_vb_map": aix_vb_dct,
            "ix_to_arg_map": aix_arg_dct,
            "ix_to_all_map": aix_encoder_dct,
        }

    def vb_arg_metrics_all(self, hypos, gts, return_sent=False):
        out_met_dct = {}
        for met in self.met_keys:
            scorer_met_corp, scorer_met_sent = self.scorers[met].fn.compute_score(
                gts=gts, res=hypos
            )
            if isinstance(scorer_met_corp, float):
                scorer_met_corp = [scorer_met_corp]
                scorer_met_sent = [scorer_met_sent]
            met_out_str_lst = self.scorers[met].out_str
            for mix, met_out_str in enumerate(met_out_str_lst):
                out_met_dct[met_out_str] = scorer_met_corp[mix]
                if return_sent:
                    out_met_dct[f"{met_out_str}_sent"] = scorer_met_sent[mix]
        return out_met_dct

    def vb_arg_compute_macro(self, hypo_str_dct, gts_str_dct, ix_to_vb_map):
        vb_to_ix_dct = {}
        for ix, vb in ix_to_vb_map.items():
            if vb not in vb_to_ix_dct:
                vb_to_ix_dct[vb] = []
            vb_to_ix_dct[vb].append(ix)
        out_met_dct_vb_lst = {}
        for vb, ix_lst in vb_to_ix_dct.items():
            hypos_vb = {k: hypo_str_dct[k] for k in ix_lst}
            gts_vb = {k: gts_str_dct[k] for k in ix_lst}
            out_met_dct_vb = self.vb_arg_metrics_all(hypos=hypos_vb, gts=gts_vb)
            out_met_dct_vb_lst[vb] = out_met_dct_vb
        collated_out_met_dct_vb = collate_dct_lst_naive(
            list(out_met_dct_vb_lst.values())
        )
        out_met_macro = {k: get_avg(v) for k, v in collated_out_met_dct_vb.items()}

        return out_met_macro, out_met_dct_vb_lst

    def get_coref_from_orig_hyp_gts_dcts(
        self, hyp_orig_dct, gts_orig_dct, met_inp=None, conv_dct=None,
    ):
        self.reset_coval_scorer_dict()

        def get_coref_dct_for_gt1(gt1):
            coref_dct = {}
            for evix, ev_i in enumerate(ev_lst, 1):
                gt_args = gt1[ev_i]["Args"]
                for gt_ag in gt_args:
                    gt_ag_name = arg_mapper(gt_ag)
                    if gt_ag_name in self.args_used:
                        gtv1 = gt_args[gt_ag]
                        if gtv1 not in coref_dct:
                            coref_dct[gtv1] = []
                        coref_dct[gtv1].append(f"{ev_i}_{gt_ag_name}")
            return coref_dct

        def get_coref_dct_for_pred(pred, gt1):
            coref_dct = {}
            for evix, ev_i in enumerate(ev_lst, 1):
                # gt_args = gt1[ev_i]["Args"]
                gt_args = list(gt1[ev_i]["Args"].keys())
                # pred_set1 = set()
                for gt_ag in gt_args:
                    gt_ag_name = arg_mapper(gt_ag)

                    if gt_ag_name in self.args_used:
                        if gt_ag_name in pred[ev_i]:
                            pred_v1 = pred[ev_i][gt_ag_name]
                            if pred_v1 not in coref_dct:
                                coref_dct[pred_v1] = []
                            coref_dct[pred_v1].append(f"{ev_i}_{gt_ag_name}")
            return coref_dct

        def preproc_dct(dct1):
            out_lst = list(dct1.values())
            return out_lst

        ev_lst = [f"Ev{ix}" for ix in range(1, 6)]
        ann_idx_keys = sorted(list(hyp_orig_dct.keys()))
        coval_mets = ["mentions", "muc", "bcub", "ceafe", "lea", "lea_soft"]
        out_f1_scores = {cmet: [] for cmet in coval_mets}

        is_lea_soft = False
        if conv_dct is not None:
            is_lea_soft = True
        if is_lea_soft:
            conv_dct2 = {}
            for ck, c in conv_dct.items():
                if c["ann_idx"] not in conv_dct2:
                    conv_dct2[c["ann_idx"]] = []
                conv_dct2[c["ann_idx"]].append(c)

        gt_max = len(gts_orig_dct[list(gts_orig_dct.keys())[0]])
        for gtix in range(gt_max):
            self.reset_coval_scorer_dict()
            for ann_idx in ann_idx_keys:
                gts1 = gts_orig_dct[ann_idx][gtix]
                hypo_1 = hyp_orig_dct[ann_idx]
                if is_lea_soft:
                    conv1 = conv_dct2[ann_idx]
                    conv11 = {v["ev_agname"]: v for v in conv1}
                if "Ev1" in hypo_1:
                    if "Args" in hypo_1["Ev1"]:
                        sys_dct = preproc_dct(get_coref_dct_for_gt1(hypo_1))
                    else:
                        sys_dct = preproc_dct(get_coref_dct_for_pred(hypo_1, gts1))
                    if is_lea_soft:
                        cid_sc_lst = []
                        for cls1 in sys_dct:
                            cid_sc_lst1 = []
                            for cls11 in cls1:
                                cid_sc_idx = conv11[cls11]
                                cid_sc = met_inp["cider_sent"][cid_sc_idx["aix"]]
                                cid_sc_lst1.append(cid_sc)
                            cid_sc_lst.append(cid_sc_lst1)

                    key_dct = preproc_dct(get_coref_dct_for_gt1(gts1))
                    key_to_sys_dct = get_mention_assignments(key_dct, sys_dct)
                    sys_to_key_dct = get_mention_assignments(sys_dct, key_dct)
                    tup = (key_dct, sys_dct, key_to_sys_dct, sys_to_key_dct)
                    for cmet in coval_mets:
                        if cmet != "lea_soft":
                            self.coval_scorer_dict[cmet].update(tup)
                        else:
                            self.coval_scorer_dict[cmet].update(
                                tup, cider_for_sys=cid_sc_lst
                            )

            for cmt in coval_mets:
                out_f1_scores[cmt].append(self.coval_scorer_dict[cmt].get_f1())
        return {cmt: sum(v) / len(v) for cmt, v in out_f1_scores.items()}

    def get_evals_from_hyp_gts_dcts(self, hyp_gts_dicts):
        hypo_str_dct = hyp_gts_dicts["hypos"]
        gts_str_dct = hyp_gts_dicts["gts"]
        ix_to_vb_map = hyp_gts_dicts["ix_to_vb_map"]
        ix_to_arg_map = hyp_gts_dicts["ix_to_arg_map"]
        out_met_dct = self.vb_arg_metrics_all(
            hypos=hypo_str_dct, gts=gts_str_dct, return_sent=True
        )
        out_met_macro_vb, _ = self.vb_arg_compute_macro(
            hypo_str_dct=hypo_str_dct,
            gts_str_dct=gts_str_dct,
            ix_to_vb_map=ix_to_vb_map,
        )
        out_met_macro_arg, out_met_dct_arg_lst = self.vb_arg_compute_macro(
            hypo_str_dct=hypo_str_dct,
            gts_str_dct=gts_str_dct,
            ix_to_vb_map=ix_to_arg_map,
        )
        for k in out_met_macro_vb:
            out_met_dct[f"MacroVb_{k}"] = out_met_macro_vb[k]
        for k in out_met_macro_arg:
            out_met_dct[f"MacroArg_{k}"] = out_met_macro_arg[k]
        for k in out_met_dct_arg_lst:
            for k1 in out_met_dct_arg_lst[k]:
                out_met_dct[f"{k}_{k1}"] = out_met_dct_arg_lst[k][k1]

        hypo_orig_dct = hyp_gts_dicts["hypos_orig"]
        gts_orig_dct = hyp_gts_dicts["gts_orig"]
        coval_mets = self.get_coref_from_orig_hyp_gts_dcts(
            hyp_orig_dct=hypo_orig_dct,
            gts_orig_dct=gts_orig_dct,
            met_inp=out_met_dct,
            conv_dct=hyp_gts_dicts["ix_to_all_map"],
        )
        out_met_dct.update(coval_mets)
        return out_met_dct

    def eval_cap_mets(
        self, pred_file: str, split_type="valid",
    ):
        self.read_gt_file(split_type=split_type)

        hyp_gts_dicts = self.prepare_hyp_gts(
            pred_file=pred_file, split_type=split_type,
        )

        return self.get_evals_from_hyp_gts_dcts(hyp_gts_dicts=hyp_gts_dicts)


def get_fname_key(task_type: str) -> str:
    fname_key_dct = {"vb": "test_verb", "vb_arg": "test_srl", "evrel": "test_evrel"}
    return fname_key_dct[task_type]


def main(
    pred_file,
    task_type: str,
    split_file_path: str,
    vinfo_file_path: str,
    vsitu_ann_file_path: str,
    split_type: str,
    out_file: str = "./results/results.json",
    **kwargs,
):

    cfg = CN(yaml.safe_load(open("./eval_files/vsitu_cfg.yml")))

    assert "valid" in split_type or "test" in split_type

    if split_type == "valid":
        fname_key = "valid"
    else:
        fname_key = get_fname_key(task_type)

    assert Path(split_file_path).exists()
    assert Path(vinfo_file_path).exists()
    assert Path(vsitu_ann_file_path).exists()

    cfg.ds.vsitu.split_files_lb[fname_key] = split_file_path
    cfg.ds.vsitu.vinfo_files_lb[fname_key] = vinfo_file_path
    cfg.ds.vsitu.vsitu_ann_files_lb[fname_key] = vsitu_ann_file_path

    cfg.freeze()

    if task_type == "vb_arg":
        evl_cap = EvalFnCap(cfg, None, met_keys=["cider", "bleu", "rouge"])
        out_met = evl_cap.eval_cap_mets(pred_file=pred_file, split_type=split_type)

        out_results = {k: float(v) for k, v in out_met.items() if "sent" not in k}

    elif task_type == "vb":
        evl_vb = EvlFn_Vb(cfg, {}, ["acc"])
        out_met = evl_vb.simple_acc(pred_file=pred_file, split_type=split_type)
        out_results = {k: float(v) for k, v in out_met.items()}
    elif task_type == "evrel":
        evl_rel_fn = EvlFn_EvRel(cfg, {}, ["Top_1"])
        out_met = evl_rel_fn.simple_acc_evrel(
            pred_file=pred_file, split_type=split_type
        )
        out_results = out_met

    with open(out_file, "w") as g:
        json.dump(out_results, g)


if __name__ == "__main__":
    fire.Fire(main)
