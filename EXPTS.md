# Reproducing Results in the Paper

This document presents how to run all experiments provided in the paper and reproducing the same results.

Pre-trained models are uploaded on google drive.

To run a particular model see `cmd` under log. 

# Verb Prediction

|    | Name        | Model Tag             | Model                                                                       | Log                                                                       |
|---:|:------------|:----------------------|:----------------------------------------------------------------------------|:--------------------------------------------------------------------------|
|  0 | I3D         | i3d_r50_8x8           | [model](https://drive.google.com/open?id=1fIKxdL1cxsFAR1rZ5TLFVfmBDFICY1yl) | [log](https://drive.google.com/open?id=1hLyreipOe7DMMctvUiJkE2kzx-duey2R) |
|  1 | I3D+NL      | i3d_r50_nl_8x8        | [model](https://drive.google.com/open?id=1y9GRN-qd2ZIt6FraBPfNbu_B47FQk4kM) | [log](https://drive.google.com/open?id=11MBIkVasnZo9qvZtW28BWFH7LzfOAIWp) |
|  2 | Slow+NL     | slow_nl_r50_8x8       | [model](https://drive.google.com/open?id=1nO5JcTSZXHQJzr3LKBg9oXZfs_GQ5UZM) | [log](https://drive.google.com/open?id=1yakkRQ6LglnY7R4h85S2-AaBHNC0YuLC) |
|  3 | SlowFast+NL | slow_fast_nl_r50_8x8  | [model](https://drive.google.com/open?id=1WcBmKhvK_gmcLlfgdGE9CNOHDsIZqiAd) | [log](https://drive.google.com/open?id=1_w9Lq4s3aRPGVm08plqeHxlxyN3woldl) |
|  4 | I3D         | kpret_i3d_r50_8x8     | [model](https://drive.google.com/open?id=1n8iO85a5kZc5qqaD3GgC2awLwcUi1nUp) | [log](https://drive.google.com/open?id=1TUlKo1HGolOuS2bf05CBUrotM_5IIrc4) |
|  5 | I3D+NL      | kpret_i3d_r50_nl_8x8  | [model](https://drive.google.com/open?id=1C1Q2JhU-1ZeJjqHYhIA2Bf2ecSdrvmr0) | [log](https://drive.google.com/open?id=1W1RVm_MtM0IGvPd8MfD-cJ8Hxsy7_Ion) |
|  6 | Slow+NL     | kpret_slow_nl_r50_8x8 | [model](https://drive.google.com/open?id=1DDWMLibf7KWWmcoFIPHOs7hJZil58qJq) | [log](https://drive.google.com/open?id=1NDyRkm-dSZiP654EiZvlWKzBQcOVqpjB) |
|  7 | SlowFast+NL | kpret_slow_nl_r50_8x8 | [model](https://drive.google.com/open?id=1956sJhUKv5lKTmmaUTR4S-PALawWFTTF) | [log](https://drive.google.com/open?id=18bKlwJi4oiNH6J2MQaGYHylKkok8QOKE) |

# Semantic Role Prediction

|    | Name        | Model Tag     | Model                                                                       | Log                                                                       | Pred                                                                       |
|---:|:------------|:--------------|:----------------------------------------------------------------------------|:--------------------------------------------------------------------------|:---------------------------------------------------------------------------|
|  0 | GPT2        | gpt2          | [model](https://drive.google.com/open?id=1LcglkMLR33B-1URZJBvwOlMHaCY7S03y) | [log](https://drive.google.com/open?id=1uDBR2VnVVK9jIO5NPPYwQQRfm0xMuB3i) | [pred](https://drive.google.com/open?id=1hBvWYCMJXCmuN2uXeVkmtpY2NiDBwPyX) |
|  1 | TxD         | txd           | [model](https://drive.google.com/open?id=1E6F7S-kjJQdpjKu2UJRIX62xj5ju2-wM) | [log](https://drive.google.com/open?id=1tcYf6jO46wRdwJ2VUoaaJAzdzRi1eUEQ) | [pred](https://drive.google.com/open?id=1vc_eQLdv344EeueCxn-jPW3437OPtvKi) |
|  2 | SF+TxD      | sfast_txd     | [model](https://drive.google.com/open?id=1ak0Zz76g35A8r0Byo_E8cExfz0jWmo6l) | [log](https://drive.google.com/open?id=1dNdSdLs3LcHrFw_1lOddXVXT0wkkgLGD) | [pred](https://drive.google.com/open?id=1rt_Hc4Qkijk5-upWAvFgd9rHFLIDiIfN) |
|  3 | SF+TxE+TxD  | sfast_txe_txd | [model](https://drive.google.com/open?id=1jAafAgdmYnU4TO6BfAQ0R25N0LQ1mqgn) | [log](https://drive.google.com/open?id=1QvLgjgrdcM650eZ3thMF5aJrD4icV6w0) | [pred](https://drive.google.com/open?id=1gddLSj2vcCszLjy7DFJo8f6pQ1VxvMIS) |
|  4 | I3D+TxD     | i3d_txd       | [model](https://drive.google.com/open?id=1K08Nh3yjyollavZUX-9ukJXM1BC57I4e) | [log](https://drive.google.com/open?id=1sCqBX7MU83fsBDxoEpem3yBmrG8vUxux) | [pred](https://drive.google.com/open?id=1x10s-fnirzf26d0o6YsLy31NjXnscwYI) |
|  5 | I3D+TxE+TxD | i3d_txe_txd   | [model](https://drive.google.com/open?id=1eb6z-09zECV5yXX5gGDszXmDH6-uQDtx) | [log](https://drive.google.com/open?id=1WJRN1FIRpR8tpDc4ZaPGsd3LHMr8ZUXf) | [pred](https://drive.google.com/open?id=1XB9eewZyb39cJLHx0Gxvcq1cSq1kYGga) |


# Event Relation

|    | Name                          | Model Tag            | Model                                                                       | Log                                                                       |
|---:|:------------------------------|:---------------------|:----------------------------------------------------------------------------|:--------------------------------------------------------------------------|
|  0 | Roberta Encoder               | rob_evrel            | [model](https://drive.google.com/open?id=138DUSruSp-2QKhEeVqDnR9LxF4UPUX-5) | [log](https://drive.google.com/open?id=1CV4wjf6lTWNeNvi33-Ol-j8_4Zxg69Og) |
|  1 | Roberta Encoder               | txe_evrel            | [model](https://drive.google.com/open?id=1HnHOuiPKjSXjaBS_s71HGgtHu0xSKzc3) | [log](https://drive.google.com/open?id=1IeTgm4OG-gEIRt2Orqo5GC1L8tPdDy4G) |
|  2 | Roberta with only Verbs + I3D | sfpret_vbonly_evrel  | [model](https://drive.google.com/open?id=1yXvkcGazwjXi_83nVbMDcBg44ruvWei-) | [log](https://drive.google.com/open?id=1O-tPjvHZkA6hXCOrhqrbOY3XRBEhbJh1) |
|  3 | Only Video I3D                | sfpret_onlyvid_evrel | [model](https://drive.google.com/open?id=1YD53Awyv8ZDatNsmTdEGbC6o_5nGQfJZ) | [log](https://drive.google.com/open?id=1Dnzr86Tp_fbq_SmMGTnmykYlJz6nWLaI) |
|  4 | Roberta + I3D                 | sfpret_evrel         | [model](https://drive.google.com/open?id=1kicpNWyjwtPBDL-cJ5MFmc6LEzaP4nJi) | [log](https://drive.google.com/open?id=1ZSxCqhoTjnm2G62RFJ27DehZhd7wLPxs) |
