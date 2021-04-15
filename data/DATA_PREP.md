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



1.  Download the Videos from youtube. It should take around 4-5 hours depending on network speed and is around 50GB in size.
    In case any video is not available, please contact Arka (asadhu@usc.edu).

    1. Install required dependencies:

        - tqdm
        - yacs
        - youtube-dl

        NOTE: youtube-dl needs to be updated with the following command before downloading the videos using

        ```
        pip install youtube-dl --upgrade
        ```

        We also provide a basic conda environment:
        ```
        MINICONDA_ROOT=[to your Miniconda/Anaconda root directory]
        conda env create -f barebones_data_setup.yml --prefix $MINICONDA_ROOT/envs/vsitu_data_env
        conda activate vsitu_data_env
        ```

        If you are using your own conda environment, you need to run:
        ```
        conda activate $ENV_NAME
        conda install tqdm
        pip install yacs
        pip install youtube-dl --upgrade
        conda install ffmpeg
        ```

    1. Download the videos
        ```
        cd $ROOT
        export PYTHONPATH=$(pwd)
        python prep_data/dwn_yt.py --task_type='dwn_vids'
        ```

1.  Extract the frames from the video.

    ```
    cd $ROOT
    export PYTHONPATH=$(pwd)
    python prep_data/dwn_yt.py --task_type='extract_frames'
    ```

1.  Alternatively, you can skip the video download process and directly use the pre-extracted features:

    ```
    cd $ROOT
    export FEATURE_ZIP_LINK=.... # to be filled after upload
    wget -c $FEATURE_ZIP_LINK
    ```
