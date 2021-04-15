# VidSitu Annotation Files:

File Structure:
```
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
    ├── vsann_train_lb.json
    └── vsann_valid_lb.json
```


There are three directories:

1. `split_files/`:
    + Containing Splits for Train, Validation, and Test Sets (three test sets).
    + Each `*.json` file is a list of video segment ids of the form `v_{video_id}_seg_{start_second}_{end_second}`
    + By construction, duration is 10 second for all videos
+ `vinfo_files/`:
    + Containing Video Information for Train and Validation.
    + Each `*.json` file is List[Dict]
    + Keys for one instance:
        - `vid_uid`: unique identifier for a video
        - `genres`: Genres of the corresponding movie
        - `duration`: Entire duration of the video (not 10 seconds)
        - `name`: Video ID corresponding to YouTube
        - `text`: Text provided in Youtube Descriptions
        - `start`: Start time in the video
        - `end`: End time in the video
        - `vid_seg_int`: Identifier for the 10-second movie clip of the form `v_{video_id}_seg_{start_second}_{end_second}`
        - `movie_name`: Name of the corresponding movie
        - `upload_year`: Year it was uploaded to Youtube.
        - `clip_name`: Title of the Youtube Video
        - `imdb_id`: Imdb id for the movie
        - `year_rel`: Year when the movie was released.
+ `vseg_ann_files/`: Containing Video Segment Annotations (SRL + Event Relations) for Train and Validation

Please refer to https://github.com/TheShadow29/VidSitu/blob/main/data/DATA_PREP.md for instructions on setting up the dataset.
