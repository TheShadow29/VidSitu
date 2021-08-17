# Data Preparation

There are three main steps in setting up the dataset. See [./data/README.md]('./data/README.md') for annotation file structure.

1.  Download the Annotation and Split Files
    ```
    export ROOT=$(PWD)
    export VDS_LINK="https://ai2-prior-vidsitu.s3-us-west-2.amazonaws.com/vsitu_data/vidsitu_data.zip"
    mkdir $ROOT/data
    cd $ROOT/data
    wget -c $VDS_LINK
    unzip vidsitu_data.zip -d vidsitu_annotations/
    rm vidsitu_data.zip
    ```

    The directory should look as follows:

    ```
    data
    └── vidsitu_annotations
        ├── split_files
        │   ├── vseg_split_testevrel_lb.json
        │   ├── vseg_split_testsrl_lb.json
        │   ├── vseg_split_testvb_lb.json
        │   ├── vseg_split_train_lb.json
        │   └── vseg_split_valid_lb.json
        ├── vinfo_files
        │   ├── vinfo_train_lb.json
        │   └── vinfo_valid_lb.json
        └── vseg_ann_files
            ├── vsann_testevrel_noann_lb.json
            ├── vsann_testsrl_noann_lb.json
            ├── vsann_train_lb.json
            └── vsann_valid_lb.json
    ```

1.  Download the Videos from youtube. It should take around ~8-10 hours depending on network speed and is around 50GB in size.
    In case any video is not available, please contact Arka (asadhu@usc.edu).

    1. Install required dependencies:

        - tqdm
        - yacs
        - yt-dlp

        NOTE: yt-dlp needs to be updated with the following command before downloading the videos using

        ```
        pip install yt-dlp
        ```

        We also provide a basic conda environment:
        ```
        MINICONDA_ROOT=[to your Miniconda/Anaconda root directory]
        conda env create -f barebones_data_setup.yml --prefix $MINICONDA_ROOT/envs/vsitu_data_env
        conda activate vsitu_data_env
        ```
        
        NOTE: You may have to install `yt-dlp` separately since the original environment used `youtub-dl` 

        If you are using your own conda environment, you need to run:
        ```
        conda activate $ENV_NAME
        conda install tqdm
        pip install yacs
        pip install yt-dlp --upgrade
        conda install ffmpeg
        ```

    1. Download the videos: (Following works for ~28k / 29k videos)
        ```
        cd $ROOT
        export PYTHONPATH=$(pwd)
        python prep_data/dwn_yt.py --task_type='dwn_vids' --max_processes=30
        ```

        There are around 1k Age-restricted videos. To download these you need to add a cookies.txt file, which you can give using `--cookies_file=/path/to/cookies.txt`. So the command would be:

        ```
        cd $ROOT
        export PYTHONPATH=$(pwd)
        python prep_data/dwn_yt.py --task_type='dwn_vids' --max_processes=30 --cookies_file=/path/to/cookies.txt
        ```


        To generate cookies.txt, follow the steps below:
        + Download Get Cookies.txt extension [Chrome](https://chrome.google.com/webstore/detail/get-cookiestxt/bgaddhkoddajcdgocldbbfleckgcbcid)
        + Login to youtube.com, and use the extension to export the cookies in txt form.


1.  Extract the frames from the video.

    ```
    cd $ROOT
    export PYTHONPATH=$(pwd)
    python prep_data/dwn_yt.py --task_type='extract_frames'
    ```

1.  Alternatively, you can skip the video download process and directly use the pre-extracted features from [google drive link](https://drive.google.com/file/d/1rBrRmew7Soul51MjLN6F55oTEzUfzyXv/view)

    To download directly on the remote, you can use the following convenience function

    ```
    function gdrive_download () {
        CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
        wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
        rm -rf /tmp/cookies.txt
    }

    cd $ROOT/data
    export FEATURE_ZIP_DRIVE_ID="1rBrRmew7Soul51MjLN6F55oTEzUfzyXv" # to be filled after upload
    gdrive_download "1rBrRmew7Soul51MjLN6F55oTEzUfzyXv" vsitu_vidfeats_drive.zip
    unzip vsitu_vidfeats_drive.zip -d vsitu_vid_feats
    rm vsitu_vidfeats_drive.zip
    ```

1. Download the vocabulary files from here: https://drive.google.com/file/d/1TAreioObLGKqU7M9wmnuaXh4b5s_2YdK/view?usp=sharing and place them under `data/vsitu_vocab`
    ```
    function gdrive_download () {
        CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
        wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
        rm -rf /tmp/cookies.txt
    }

    cd $ROOT/data
    export VOCAB_ZIP_DRIVE_ID="1TAreioObLGKqU7M9wmnuaXh4b5s_2YdK" # to be filled after upload
    gdrive_download $VOCAB_ZIP_DRIVE_ID vsitu_vocab.zip
    unzip vsitu_vocab.zip -d vsitu_vocab
    rm vsitu_vocab.zip
    ```
    
1. Feature Extration: After training a verb model, one might be interested in re-extracting the features on their end. We provide [vidsitu_code/feat_extractor.py](vidsitu_code/feat_extractor.py). 

    To run a particular saved model, use the following command:
    ```
    export PYTHONPATH=$(pwd)
    python vidsitu_code/feat_extractor.py ----mdl_resume_path='/path/to/saved_model' --mdl_name_used='some_name_used_as_dir_for_feats' --ds.vsitu.vsu_frm_feats='/top/level/featuredir' --mdl.mdl_name='sf_base' --mdl.sf_mdl_name='i3d_r50_nl_8x8' --is_cu=False
    ```
    
    Note that `sf_mdl_name` needs to match the name in `extended_config.py`. If you want to use a checkpoint from Slowfast repository where some of the models are saved in caffe2, use `--is_cu=True` in the argument.
    
    Thus, for using I3D_NL model trained on vidsitu verbs, the command could be:
    ```
    export PYTHONPATH=$(pwd)
    CUDA_VISIBLE_DEVICES=5 python vidsitu_code/feat_extractor.py --mdl_resume_path='./weights/i3d_nln_r50_vsitu.pth' --mdl_name_used='i3d_recheck' --ds.vsitu.vsitu_frm_feats='./data/vsitu_features' --mdl.mdl_name='sf_base' --mdl.sf_mdl_name='i3d_r50_nl_8x8' --is_cu=False
    ```
    
    To use I3D_NL model from Slowfast, it would be:

    ```
    export PYTHONPATH=$(pwd)
    CUDA_VISIBLE_DEVICES=5 python vidsitu_code/feat_extractor.py --mdl_resume_path='./weights/I3D_NLN_8x8_R50.pkl' --mdl_name_used='i3d_recheck_kpret' --ds.vsitu.vsitu_frm_feats='./data/vsitu_features' --mdl.mdl_name='sf_base' --mdl.sf_mdl_name='i3d_r50_nl_8x8' --is_cu=True    
    ```

