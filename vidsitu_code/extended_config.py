import json
from pathlib import Path
from yacs.config import CfgNode as CN
from utils._init_stuff import yaml
from typing import Dict, Any
from slowfast.config.defaults import get_cfg
import argparse
from fairseq.models import ARCH_CONFIG_REGISTRY, ARCH_MODEL_REGISTRY
from fairseq.models.transformer import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
)

sf_mdl_to_cfg_fpath_dct = {
    "slow_fast_nl_r50_8x8": "./configs/vsitu_mdl_cfgs/Kinetics_c2_SLOWFAST_8x8_R50.yaml",
    "slow_nl_r50_8x8": "./configs/vsitu_mdl_cfgs/Kinetics_c2_SLOW_8x8_R50.yaml",
    "c2d_r50_8x8": "./configs/vsitu_mdl_cfgs/Kinetics_C2D_8x8_R50.yaml",
    "i3d_r50_8x8": "./configs/vsitu_mdl_cfgs/Kinetics_c2_I3D_8x8_R50.yaml",
    "i3d_r50_nl_8x8": "./configs/vsitu_mdl_cfgs/Kinetics_c2_I3D_NLN_8x8_R50.yaml",
}

tx_to_cfg_fpath_dct = {
    "transformer": "./configs/vsitu_tx_cfgs/transformer.yaml",
}


def get_default_tx_dec_cfg():
    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS, allow_abbrev=False
    )
    ARCH_MODEL_REGISTRY["transformer"].add_args(parser)
    args1 = parser.parse_known_args()[0]
    ARCH_CONFIG_REGISTRY["transformer"](args1)
    args1_dct = vars(args1)
    args1_dct["max_source_positions"] = DEFAULT_MAX_SOURCE_POSITIONS
    args1_dct["max_target_positions"] = DEFAULT_MAX_TARGET_POSITIONS
    return CN(args1_dct)


class CfgProcessor:
    def __init__(self, cfg_pth):
        assert Path(cfg_pth).exists()
        self.cfg_pth = cfg_pth

    def get_vsitu_default_cfg(self):
        with open(self.cfg_pth) as f:
            c4 = yaml.safe_load(f)
        cfg_dct = c4.copy()
        return CN(cfg_dct)

    def get_key_maps(self):
        key_maps = {}
        return key_maps

    @staticmethod
    def get_val_from_cfg(cfg, key_str):
        key_split = key_str.split(".")
        d = cfg
        for k in key_split[:-1]:
            d = d[k]

        return d[key_split[-1]]

    def create_from_dict(self, dct: Dict[str, Any], prefix: str, cfg: CN):
        """
        Helper function to create yacs config from dictionary
        """
        dct_cfg = CN(dct, new_allowed=True)
        prefix_list = prefix.split(".")
        d = cfg
        for pref in prefix_list[:-1]:
            assert isinstance(d, CN)
            if pref not in d:
                setattr(d, pref, CN())
            d = d[pref]
        if hasattr(d, prefix_list[-1]):
            old_dct_cfg = d[prefix_list[-1]]
            dct_cfg.merge_from_other_cfg(old_dct_cfg)

        setattr(d, prefix_list[-1], dct_cfg)
        return cfg

    @staticmethod
    def update_one_full_key(cfg: CN, dct, full_key, val=None):
        if cfg.key_is_deprecated(full_key):
            return
        if cfg.key_is_renamed(full_key):
            cfg.raise_key_rename_error(full_key)

        if val is None:
            assert full_key in dct
            v = dct[full_key]
        else:
            v = val
        key_list = full_key.split(".")
        d = cfg
        for subkey in key_list[:-1]:
            # Most important statement
            assert subkey in d, f"key {full_key} doesnot exist"
            d = d[subkey]

        subkey = key_list[-1]
        # Most important statement
        assert subkey in d, f"key {full_key} doesnot exist"

        value = cfg._decode_cfg_value(v)

        assert isinstance(value, type(d[subkey]))
        d[subkey] = value

        return

    def update_from_dict(
        self, cfg: CN, dct: Dict[str, Any], key_maps: Dict[str, str] = None
    ) -> CN:
        """
        Given original CfgNode (cfg) and input dictionary allows changing
        the cfg with the updated dictionary values
        Optional key_maps argument which defines a mapping between
        same keys of the cfg node. Only used for convenience
        Adapted from:
        https://github.com/rbgirshick/yacs/blob/master/yacs/config.py#L219
        """
        # Original cfg
        # root = cfg
        if key_maps is None:
            key_maps = []
        # Change the input dictionary using keymaps
        # Now it is aligned with the cfg
        full_key_list = list(dct.keys())
        for full_key in full_key_list:
            if full_key in key_maps:
                # cfg[full_key] = dct[full_key]
                self.update_one_full_key(cfg, dct, full_key)
                new_key = key_maps[full_key]
                # dct[new_key] = dct.pop(full_key)
                self.update_one_full_key(cfg, dct, new_key, val=dct[full_key])

        # Convert the cfg using dictionary input
        # for full_key, v in dct.items():
        for full_key in dct.keys():
            self.update_one_full_key(cfg, dct, full_key)
        return cfg

    @staticmethod
    def pre_proc_config(cfg: CN, dct: Dict = None):
        """
        Add any pre processing based on cfg
        """

        def upd_sub_mdl(
            cfg: CN,
            sub_mdl_default_cfg: CN,
            sub_mdl_name_key: str,
            sub_mdl_file_key: str,
            sub_mdl_mapper: Dict,
            new_dct: Dict,
        ):
            if new_dct is not None and sub_mdl_name_key in new_dct:
                sub_mdl_name = new_dct[sub_mdl_name_key]
            else:
                sub_mdl_name = CfgProcessor.get_val_from_cfg(cfg, sub_mdl_name_key)

            assert sub_mdl_name in sub_mdl_mapper
            sub_mdl_file = sub_mdl_mapper[sub_mdl_name]
            assert Path(sub_mdl_file).exists()
            CfgProcessor.update_one_full_key(
                cfg, {sub_mdl_file_key: sub_mdl_file}, full_key=sub_mdl_file_key
            )

            sub_mdl_default_cfg.merge_from_file(sub_mdl_file)
            sub_mdl_cfg = yaml.safe_load(sub_mdl_default_cfg.dump())
            sub_mdl_cfg_dct_keep = {k: v for k, v in sub_mdl_cfg.items()}

            return CN(sub_mdl_cfg_dct_keep)

        sf_mdl_cfg_default = get_cfg()
        cfg.sf_mdl = upd_sub_mdl(
            cfg,
            sf_mdl_cfg_default,
            "mdl.sf_mdl_name",
            "mdl.sf_mdl_cfg_file",
            sf_mdl_to_cfg_fpath_dct,
            dct,
        )
        tx_dec_default = get_default_tx_dec_cfg()
        cfg.tx_dec = upd_sub_mdl(
            cfg,
            tx_dec_default,
            "mdl.tx_dec_mdl_name",
            "mdl.tx_dec_cfg_file",
            tx_to_cfg_fpath_dct,
            dct,
        )
        return cfg

    @staticmethod
    def post_proc_config(cfg: CN):
        """
        Add any post processing based on cfg
        """
        return cfg

    @staticmethod
    def cfg_to_flat_dct(cfg: CN):
        def to_flat_dct(dct, prefix_key: str):
            def get_new_key(prefix_key, curr_key):
                if prefix_key == "":
                    return curr_key
                return prefix_key + "." + curr_key

            out_dct = {}
            for k, v in dct.items():
                if isinstance(v, dict):
                    out_dct1 = to_flat_dct(v, prefix_key=get_new_key(prefix_key, k))
                else:
                    out_dct1 = {get_new_key(prefix_key, k): v}
                out_dct.update(out_dct1)
            return out_dct

        cfg_dct = json.loads(json.dumps(cfg))
        return to_flat_dct(cfg_dct, prefix_key="")

    @staticmethod
    def to_str(cfg: CN):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(cfg.items()):
            # seperator = "\n" if isinstance(v, CN) else " "
            if isinstance(v, CN):
                seperator = "\n"
                str_v = CfgProcessor.to_str(v)
            else:
                seperator = " "
                str_v = str(v)
                if str_v == "" or str_v == "":
                    str_v = "''"
            attr_str = "{}:{}{}".format(str(k), seperator, str_v)
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r
