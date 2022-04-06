# Week 3: Object Tracking

The goal for this week is providing trackings of cars in the AI Cities dataset
and refining the techniques to produce them. In particular, we have reimplemented
both **Kalman Filtering** and **Max Overlap Tracking** to substitute our previous
faulty implementations.

We also study different optical flow algorithms and systems to incorporate it to
tracking, although we have not been able to produce meaningful results out of it
(we were running short on time mostly).

## Running the code

We have adopted a similar modular strategy as last week in order to be able to
precompute significant chunks of data and then pipeline their processing with
either global python scripts or shell scripts.

### Directories

The following directory structures are assumed

- AI Cities dataset: Default structure

```text
+-- cam_framenum
+-- cam_loc
+-- cam_timestamp
\-- train
    +-- S01
    │    +-- c001
    |   ...
    │    \-- c005
    +-- S03
    │    +-- c010
    |   ...
    │    \-- c015
    \-- S04
        +-- c016
        ...
        \-- c040   <Example inner directory>
            +-- det
            +-- gt              // We add a COCO dict with the same info as gt.txt here
            +-- mtsc
            +-- segm
            +-- vdo_of          // Generated Optical Flow
            \-- vdo_frames      // Generated per-frame images
```

- Results will be generated in a tree like the following

```text
\-- train_models_test<SEQUENCE>
    +-- <MODEL NAME>
        +--- ai_cities<SEQUENCE><CAMERA>
            +-- coco_instances_results.json     // Prediction bboxes in COCO format
            +-- demo_<NAME>.mp4                 // Various demos in GIF or MP4 format
            +-- instances_predictions.pth       // Predictions for the last layer of the model
            +-- stats.json                      // Detection evaluation metrics
            +-- summary[_purge].csv             // Tracking evaluation summary
            \-- track[_purge].txt               // Track in MOT format
```

We provide our results tree [here](https://drive.google.com/file/d/1xa2zWrlLdvIVEBt0Nq52W1ao1nN94020/view?usp=sharing)

### Requisites

We have a requirements.txt file in the root directory detailing packages needed
to run our code. We use
[**Detectron2**](https://github.com/facebookresearch/detectron2) for object detectors, which can be
installed following their documentation.

Additionally, we require ffmpeg to split frames and make gifs
(and also output videos with the opencv wrapper) and the various optical flow
libraries we use need to be installed separately.

### Script usage

There is a fair share of paths that can be found within the scripts that may 
need to be adjusted to your machine (we have used argument parsers whenever
we could, but some automations like bash scripts are designed to exploit
path structures as much as possible).

Core scripts that implement each task are named accordingly:

- [```task_11.py```](./task_11.py) performs Task 1.1 - our implementation
of the Block Matching Algorithm. The script itself automates experiments, 
whereas all tooling and detailed implementations may be found in
[```of.py```](./of.py) and [```of_utils.py```](./of_utils.py) (this also applies
to Task 1.2 and 1.3).
- [```task_12.py```](./task_12.py) performs Task 1.2, which tests some current
implementations of algorithms
- [```task_13.py```](./task_13.py) performs Task 1.3. The implementation of
the optical flow tracker may be found in the [```track.py```](./track.py)
module.
- [```task_2.py```](./task_2.py) performs Task 2. It computes tracks using max
overlap and our simple track purging system (can be disabled or enabled).

Some other important scripts include
- [```eval_track.py```](./eval_track.py), which evaluates the performance of
a track comparing either against a .xml or a .txt mots file.
- [```build_opticalflow.py```](./build_opticalflow.py) generates an optical 
flow .npy file for each pair of frames in a video. **Use with care, results
are uncompressed and may use a lot of disk space**.

Many of these scripts are designed to be run through bash scripts, if they are 
to be run as singletons, use the ```help``` command for each of them.

We have automated many tasks in our quest to be able to generate global results
for comprehensive sets of sequences. The [```bash```](./bash) directory contains
all of these automatisations. Essentially:

- [```make_track_video.sh```](./bash/make_track_video.sh), 
[```make_video.sh```](./bash/make_video.sh) and [```make_gif.sh```](./bash/make_gif.sh)
use track information in each of the **results** directories and produce our
visualisations.
- [```run_train.sh```](./bash/run_train.sh) executes training for all models
in all combinations of output datasets.
- [```split_in_frames.sh```](./bash/split_in_frames.sh) converts ```vdo.avi``` files
everywhere in the dataset into singleton frames stored in ```vdo_frames```.
- [```track_detections.sh```](./bash/track_detections.sh) and 
[```track_detections_purge.sh```](./bash/track_detections_purge.sh) produce
track files for all combinations of model and input data. This wraps
the [```task_2.py```](./task_2.py) code.
