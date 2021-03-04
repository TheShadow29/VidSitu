# Installation

We provide instructions to install the required dependencies.

Requirements:
+ python>=3.6
+ pytorch==1.5 (should work with pytorch >=1.5 as well but not tested)

1. Clone the repo:
```
git clone https://github.com/TheShadow29/VidSitu.git
cd VidSitu
export ROOT=$(pwd)
```

2. To use the same environment you can use conda and the environment file vsitu_pyt_env.yml file provided.
Please refer to [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for details on installing conda.

```
MINICONDA_ROOT=[to your Miniconda/Anaconda root directory]
conda env create -f vsitu_pyt_env.yml --prefix $MINICONDA_ROOT/envs/vsitu_pyt
conda activate vsitu_pyt
```
