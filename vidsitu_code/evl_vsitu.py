"""
Evalution for Vsitu
"""
import torch
from torch import nn
from torch.nn import functional as F
import pickle
from pathlib import Path
from utils.trn_utils import (
    progress_bar,
    move_to,
    synchronize,
    is_main_process,
    compute_avg_dict,
    get_world_size,
)
from vidsitu_code.evl_fns import EvlFn_Vb, EvalFnCap, EvlFn_EvRel
from vidsitu_code.seq_gen import SeqGenCustom


class EvalB(nn.Module):
    def __init__(self, cfg, comm, device):
        super().__init__()
        self.cfg = cfg
        self.full_cfg = cfg
        self.comm = comm
        self.device = device
        self.met_keys = ["Per_Ev_Top_1", "Per_Ev_Top_5", "recall_macro_1_th_9"]
        self.after_init()
        return

    def after_init(self):
        self.evl_met = EvlFn_Vb(self.cfg, self.comm, self.met_keys)
        self.evl_fn = self.evl_met.simple_acc
        self.compute_loss = False

        return

    def forward_one_batch(self, mdl, inp):
        mdl_out = mdl(inp)["mdl_out"]
        mdl_out_probs = F.softmax(mdl_out, dim=-1)
        mdl_probs_sorted, mdl_ixs_sorted = mdl_out_probs.sort(dim=-1, descending=True)
        # label_lst10 = inp["label_tensor10"]
        ann_lst = inp["vseg_idx"]
        topk_save = 5

        def get_dct(pred_vbs, pred_scores, ann_idx):
            pred_vbs_out = []
            pred_scores_out = []
            assert len(pred_vbs) == 5
            assert len(pred_scores) == 5
            # assert len(tgt_vbs10) == 5

            # iterate over Ev1-5
            for pvb, pvs in zip(pred_vbs, pred_scores):
                pvb_used = pvb[:topk_save]
                pvb_str = [self.comm.vb_id_vocab.symbols[pv] for pv in pvb_used]
                pred_vbs_out.append(pvb_str)

                pvb_score = pvs[:topk_save]
                pred_scores_out.append(pvb_score)

            return {
                "pred_vbs_ev": pred_vbs_out,
                "pred_scores_ev": pred_scores_out,
                "ann_idx": ann_idx,
            }

        out_dct_lst = [
            get_dct(pred_vbs, pred_scores, ann_idx)
            for pred_vbs, pred_scores, ann_idx in zip(
                mdl_ixs_sorted.tolist(), mdl_probs_sorted.tolist(), ann_lst.tolist(),
            )
        ]
        return out_dct_lst

    def forward(self, model, loss_fn, dl, dl_name, rank=0, pred_path=None, mb=None):

        fname = Path(pred_path) / f"{dl_name}_{rank}.pkl"
        model.eval()
        model.to(self.device)
        loss_keys = loss_fn.loss_keys
        val_losses = {k: [] for k in loss_keys}
        nums = []
        results = []
        for batch in progress_bar(dl, parent=mb):
            batch = move_to(batch, self.device)
            b = next(iter(batch.keys()))
            nums.append(batch[b].size(0))
            torch.cuda.empty_cache()
            if self.compute_loss:
                with torch.no_grad():
                    out = model(batch)
                    out_loss = loss_fn(out, batch)

                for k in out_loss:
                    val_losses[k].append(out_loss[k].detach().cpu())
            results += self.forward_one_batch(model, batch)
        pickle.dump(results, open(fname, "wb"))
        nums = torch.tensor(nums).float()
        if self.compute_loss:
            val_loss = compute_avg_dict(val_losses, nums)

        synchronize()
        if is_main_process():
            curr_results = results
            world_size = get_world_size()
            for w in range(1, world_size):
                tmp_file = Path(pred_path) / f"{dl_name}_{w}.pkl"
                with open(tmp_file, "rb") as f:
                    tmp_results = pickle.load(f)
                curr_results += tmp_results
                tmp_file.unlink
            with open(fname, "wb") as f:
                pickle.dump(curr_results, f)
            if self.full_cfg.only_test:
                task_type = self.full_cfg.task_type
                if task_type == "vb":
                    spl = "test_verb"
                elif task_type == "vb_arg":
                    spl = "test_srl"
                elif task_type == "evrel":
                    spl = "test_evrel"
                else:
                    raise NotImplementedError
            else:
                spl = "valid"
            out_acc = self.evl_fn(fname, split_type=spl)
            val_acc = {
                k: torch.tensor(v).to(self.device)
                for k, v in out_acc.items()
                if k in self.met_keys
            }
        synchronize()
        if is_main_process():
            if self.compute_loss:
                return val_loss, val_acc
            else:
                dummy_loss = {k: torch.tensor(0.0).to(self.device) for k in loss_keys}
                return dummy_loss, val_acc
        else:
            return (
                {k: torch.tensor(0.0).to(self.device) for k in loss_keys},
                {k: torch.tensor(0.0).to(self.device) for k in self.met_keys},
            )


class EvalB_Gen(EvalB):
    def after_init(self):
        self.in_met_keys = ["cider", "bleu", "rouge"]
        self.met_keys = ["cider", "rouge", "lea", "MacroVb_cider", "MacroArg_cider"]

        self.evl_met = EvalFnCap(
            self.cfg, self.comm, self.in_met_keys, read_val_file=True
        )
        self.evl_fn = self.evl_met.eval_cap_mets
        self.compute_loss = False

    def forward_one_batch(self, mdl, inp):

        if self.cfg.num_gpus > 1:
            seq_gen = SeqGenCustom(
                [mdl.module], tgt_dict=self.comm.gpt2_hf_tok, **self.cfg.gen
            )
            out_sents = mdl.module.forward_gen(inp, seq_gen)
        else:
            seq_gen = SeqGenCustom(
                [mdl], tgt_dict=self.comm.gpt2_hf_tok, **self.cfg.gen
            )
            out_sents = mdl.forward_gen(inp, seq_gen)
        ann_lst = inp["vseg_idx"]
        wvoc = self.comm.gpt2_hf_tok

        def conv_seq_to_srl(inp_seq: str, ann_idx):
            inp_tok_lst = inp_seq.split(" ")
            if "." not in inp_tok_lst[0]:
                return {}
            vb = inp_tok_lst[0]
            ix = 1
            vb_dct = {"vb_id": vb}
            curr_str_lst = []
            curr_arg_name = ""
            while ix < len(inp_tok_lst):
                if inp_tok_lst[ix] not in self.comm.ag_name_dct.ag_dct_start.values():
                    curr_str_lst.append(inp_tok_lst[ix])
                else:
                    if ix > 1:
                        vb_dct[curr_arg_name] = " ".join(curr_str_lst)
                    curr_arg_name = inp_tok_lst[ix].split("<", 1)[1].rsplit(">", 1)[0]
                    curr_str_lst = []
                ix += 1
            vb_dct[curr_arg_name] = " ".join(curr_str_lst)

            return vb_dct

        ev_lst = [f"Ev{ix}" for ix in range(1, 6)]

        def get_dct(out_sent, ann_idx):
            out_vb_dct = {}
            for ev_ix, ev_in in enumerate(ev_lst):

                assert len(out_sent[ev_ix]) == 1
                out_sent_toks = wvoc.decode(
                    out_sent[ev_ix][0], skip_special_tokens=True
                )
                out_vb_dct[ev_in] = conv_seq_to_srl(out_sent_toks, ann_idx)
            out_dct = {"ann_idx": ann_idx, "vb_output": out_vb_dct}
            return out_dct

        out_dct_lst = [
            get_dct(pred_sent, ann_idx)
            for pred_sent, ann_idx in zip(out_sents.tolist(), ann_lst.tolist(),)
        ]
        return out_dct_lst


class EvalB_Acc(EvalB):
    def after_init(self):
        self.met_keys = ["Macro_Top_1", "Top_1"]
        self.evl_met = EvlFn_EvRel(self.cfg, self.comm, self.met_keys)
        self.evl_fn = self.evl_met.simple_acc_evrel
        self.compute_loss = True

    def forward_one_batch(self, mdl, inp):

        mdl_out = mdl(inp)["mdl_out"]
        mdl_out_probs = F.softmax(mdl_out, dim=-1)
        mdl_probs_sorted, mdl_ixs_sorted = mdl_out_probs.sort(dim=-1, descending=True)
        ann_lst = inp["vseg_idx"]

        def get_dct(pred_vbs, pred_scores, ann_idx):
            pred_vbs_out = []
            pred_scores_out = []

            assert len(pred_vbs) == 4
            assert len(pred_scores) == 4

            # iterate over Ev1-5
            for pvb, pvs in zip(pred_vbs, pred_scores):

                pvb_used = [pvb_i[0] for pvb_i in pvb]
                pvb_str = [self.comm.evrel_dct_opp[pv] for pv in pvb_used]

                pred_vbs_out.append(pvb_str)

                pvb_score = [pvs_i[0] for pvs_i in pvs]
                pred_scores_out.append(pvb_score)

            return {
                "pred_evrels_ev": pred_vbs_out,
                "pred_scores_ev": pred_scores_out,
                "ann_idx": ann_idx,
            }

        out_dct_lst = [
            get_dct(pred_vbs, pred_scores, ann_idx)
            for pred_vbs, pred_scores, ann_idx in zip(
                mdl_ixs_sorted.tolist(), mdl_probs_sorted.tolist(), ann_lst.tolist(),
            )
        ]
        return out_dct_lst
