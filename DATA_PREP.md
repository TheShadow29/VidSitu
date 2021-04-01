# Data Preparation

There are three main steps in setting up the dataset.

1. Download the Annotation and Split Files
```
export ROOT=$(PWD)
export VDS_LINK="https://ai2-prior-vidsitu.s3-us-west-2.amazonaws.com/vsitu_data/vidsitu_annotations.zip"
mkdir $ROOT/data
cd $ROOT/data
wget -c $VDS_LINK
unzip vidsitu_annotations.zip -d vidsitu_data/
```

1. Download the Videos from youtube. In case any video is not available, please contact Arka (asadhu@usc.edu)

```
cd $ROOT
export PYTHONPATH=$(pwd)
python prep_data/dwn_yt.py --task_type='dwn_vids'
```

1. Extract the frames from the video.

```
cd $ROOT
export PYTHONPATH=$(pwd)
python prep_data/dwn_yt.py --task_type='extract_frames'
```

1. Alternatively, you can skip the video download process and directly use the pre-extracted features:

```
cd $ROOT
export FEATURE_ZIP_LINK=.... # to be filled after upload
wget -c $FEATURE_ZIP_LINK
```