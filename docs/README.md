# Example Outputs

Two annotations from each of the Pickle files are pasted here for convenience

# Verbs Prediction Output

## Format
    Verb Prediction:
        ```
        List[Dict]
        Dict:
            # Both lists of length 5. Outer list denotes Events 1-5, inner list denotes Top-5 VerbID predictions
            pred_vbs_ev: List[List[str]]
            # Both lists of length 5. Outer list denotes Events 1-5, inner list denotes the scores for the Top-5 VerbID predictions
            pred_scores_ev: List[List[float]]
            #the index of the video segment used. Corresponds to the number in {valid|test}_split_file.json
            ann_idx: int
        ```
## Example

```
[
    {
        "pred_vbs_ev": [
            [
                "speak.01",
                "walk.01",
                "gesture.01",
                "open.01",
                "stare.01"
            ],
            [
                "speak.01",
                "walk.01",
                "stare.01",
                "open.01",
                "gesture.01"
            ],
            [
                "speak.01",
                "walk.01",
                "open.01",
                "stare.01",
                "gesture.01"
            ],
            [
                "speak.01",
                "walk.01",
                "open.01",
                "stare.01",
                "gesture.01"
            ],
            [
                "speak.01",
                "walk.01",
                "open.01",
                "stare.01",
                "gesture.01"
            ]
        ],
        "pred_scores_ev": [
            [
                0.0030466204043477774,
                0.002083742758259177,
                0.0018629760015755892,
                0.001817089505493641,
                0.001718472340144217
            ],
            [
                0.0022052170243114233,
                0.0016227938467636704,
                0.0015743094263598323,
                0.0015046339249238372,
                0.0013543792301788926
            ],
            [
                0.002279863925650716,
                0.0016998160863295197,
                0.0016223834827542305,
                0.0015388673637062311,
                0.0013340790756046772
            ],
            [
                0.0023030810989439487,
                0.0017610617214813828,
                0.0016733736265450716,
                0.001517956960014999,
                0.001354981679469347
            ],
            [
                0.002461366355419159,
                0.0018218016484752297,
                0.0017280317842960358,
                0.00154727918561548,
                0.001423937501385808
            ]
        ],
        "ann_idx": 0
    },
    {
        "pred_vbs_ev": [
            [
                "speak.01",
                "walk.01",
                "open.01",
                "stare.01",
                "gesture.01"
            ],
            [
                "speak.01",
                "walk.01",
                "open.01",
                "stare.01",
                "gesture.01"
            ],
            [
                "speak.01",
                "walk.01",
                "open.01",
                "stare.01",
                "gesture.01"
            ],
            [
                "speak.01",
                "walk.01",
                "stare.01",
                "open.01",
                "gesture.01"
            ],
            [
                "speak.01",
                "walk.01",
                "stare.01",
                "open.01",
                "gesture.01"
            ]
        ],
        "pred_scores_ev": [
            [
                0.0021817495580762625,
                0.0016885860823094845,
                0.0015648703556507826,
                0.0015622647479176521,
                0.001361470902338624
            ],
            [
                0.0022408771328628063,
                0.001727297087199986,
                0.0015824941219761968,
                0.0015677119372412562,
                0.001424293965101242
            ],
            [
                0.002233398612588644,
                0.0017284195637330413,
                0.0015865974128246307,
                0.001565285143442452,
                0.00143557193223387
            ],
            [
                0.0021836988162249327,
                0.0016955292085185647,
                0.0015635901363566518,
                0.001559897093102336,
                0.0014098974643275142
            ],
            [
                0.0022019033785909414,
                0.0017219308065250516,
                0.00157328846398741,
                0.0015720903174951673,
                0.0014175721444189548
            ]
        ],
        "ann_idx": 1
    }
]
```

# Semantic Roles Output

## Format
Semantic Role Labeling Prediction
        ```
        List[Dict]
        Dict:
            # same as above
            ann_idx: int
            # The main output used for evaluation. Outer Dict is for Events 1-5.
            vb_output: Dict[Dict]
            # The inner dict has the following keys:
                # VerbID of the event
                vb_id: str
                ArgX: str
                ArgY: str
                ...
        ```
        Note that ArgX, ArgY depend on the specific VerbID

## Example Output

```
[
    {
        "ann_idx": 0,
        "vb_output": {
            "Ev1": {
                "vb_id": "drive.01",
                "Arg1": "man in a white",
                "AScn": "in a home"
            },
            "Ev2": {
                "vb_id": "drive.01",
                "Arg1": "man in a white",
                "AScn": "in a home"
            },
            "Ev3": {
                "vb_id": "drive.01",
                "Arg0": "woman in a white the woman in a white",
                "Arg1": "the woman in a white",
                "AScn": "in a white"
            },
            "Ev4": {
                "vb_id": "look.01",
                "Arg0": "man in a white the bed",
                "Arg1": "the bed",
                "AScn": "in a white"
            },
            "Ev5": {
                "vb_id": "hold.01",
                "Arg0": "man in a white",
                "Arg1": "the woman in a white",
                "AMnr": "the bed",
                "AScn": "in a white"
            }
        }
    },
    {
        "ann_idx": 1,
        "vb_output": {
            "Ev1": {
                "vb_id": "collapse.01",
                "Arg1": "man in a white"
            },
            "Ev2": {
                "vb_id": "agonize.01",
                "Arg1": "man in a white"
            },
            "Ev3": {
                "vb_id": "wave.01",
                "Arg1": "man in a white"
            },
            "Ev4": {
                "vb_id": "approach.01",
                "Arg1": "man in a white",
                "AMnr": "",
                "AScn": "in a white"
            },
            "Ev5": {
                "vb_id": "walk.01",
                "Arg0": "man in a white"
            }
        }
    }
]
```


# Event Relations Output

## Format
Event Relation Prediction
        ```
        List[Dict]
        Dict:
            # same as above
            ann_idx: int
            # Ouuter list of length 4 and denotes Event Relation {1-3, 2-3, 3-4, 4-5}. Inner list denotes three Event Relations for given Verb+Semantic Role Inputs
            pred_evrels_ev: List[List[str]]
            # Scores for the above
            pred_scores_ev: List[List[float]]
        ```
## Example Output

```
[
    {
        "pred_evrels_ev": [
            [
                "NoRel",
                "NoRel",
                "NoRel"
            ],
            [
                "Causes",
                "Causes",
                "Causes"
            ],
            [
                "Causes",
                "Causes",
                "Causes"
            ],
            [
                "Causes",
                "Causes",
                "Causes"
            ]
        ],
        "pred_scores_ev": [
            [
                0.5378286242485046,
                0.5378297567367554,
                0.5378293991088867
            ],
            [
                0.9600680470466614,
                0.9600679278373718,
                0.9600680470466614
            ],
            [
                0.9526531100273132,
                0.9526529908180237,
                0.9526529908180237
            ],
            [
                0.868851900100708,
                0.8688518404960632,
                0.868851900100708
            ]
        ],
        "ann_idx": 0
    },
    {
        "pred_evrels_ev": [
            [
                "Enables",
                "Enables",
                "Enables"
            ],
            [
                "Causes",
                "Causes",
                "Causes"
            ],
            [
                "NoRel",
                "NoRel",
                "NoRel"
            ],
            [
                "NoRel",
                "NoRel",
                "NoRel"
            ]
        ],
        "pred_scores_ev": [
            [
                0.6660529375076294,
                0.6660532355308533,
                0.6660540699958801
            ],
            [
                0.5226033926010132,
                0.5226028561592102,
                0.5226027965545654
            ],
            [
                0.4319555163383484,
                0.43195492029190063,
                0.4319511651992798
            ],
            [
                0.45402148365974426,
                0.45402079820632935,
                0.4540177583694458
            ]
        ],
        "ann_idx": 1
    }
]
```