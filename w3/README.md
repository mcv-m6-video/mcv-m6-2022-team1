# Week 3: Object Tracking

## Running the code

For this week we took a slightly different approach. As we have many paths to
take care of and scripts have a tendency of fulfilling modular tasks, we have
written them as small programs and orchestrated them through bash scripts. We
also tested ideas and data using Jupyter Notebooks.

## Task overview

### Task 1: Working with Object Detectors

Task 1 Consists on running inference on off-the shelf detectors and also
fine-tuning some of them to our data (the AI Cities sequence we have been
working with so far).

Our choice for models:

- **Faster RCNN + FPN**
- **Faster RCNN + C4**
- **Faster RCNN + DC5**
- **RetinaNet + FPN**

We wanted to see differences between the different modifications of the Faster
RCNN workflow and compare them against a RetinaNet. We also had all of these
models readily available on the same framework, which meant we could re-use our
code and work with standard file formats (in this case, COCO object detection
outputs). All Python scripts have automatic argument help for input parameters,
so we refer to those for usage.

To perform off-the-shelf inference, we have the [```off_shelf.py```](./off_shelf.py)
script. It uses a custom version of our dataset files in COCO format in which
we changed the object IDs to match those from COCO and 91 placeholder
classes. We run them in batch for all architectures using
[```run_holdout_off_shelf.sh```](./run_holdout_off_shelf.sh) and
[```run_kfold_off_shelf.sh```](./run_kfold_off_shelf.sh), depending on whether
we used k-fold evaluation or holdout.

To train the models we used the [```train_arch.py```](./train_arch.py) script.
We wrote three training bash scripts for different scenarios, but they essentially
do the same (train holdout or kfold variations of the models): 
[```run_train_holdout.sh```](./run_train_holdout.sh),
[```run_kfold_cluster.sh```](./run_kfold_cluster.sh),
[```run_train_retinanet.sh```](./run_train_retinanet.sh).

To create GIFs we have written the [```make_video.py```](./make_video.py) and
[```make_video.sh```](./make_video.sh), which output mp4 files that can be
converted using ```ffmpeg``` with the [```make_gif.sh```](./make_gif.sh) script.

![](./plots/trained_fasterfpn.gif)

### Task 2: Tracking

This task consists on performing tracking of moving objects on the scene. To
this end, two methods are to be implemented: **maximum overlap tracking** and
**Kalman filter tracking**.

Tracking with maximum overlap is implemented in [```task2.py```](./task2.py).
The tracking logic is implemented within the [```track.py```](./track.py) file
and the homonimous class. 

This task has 3 hardcoded variables:

- ```detections_path```: The .json file with the annotations from the task1 object detection.
- ```frame_path```: Where to find the input video divided by frames. We
  extracted them using ```ffmpeg```.
- ```out_data```: to save track list, in order to save computation time.

The program generates a max_overlap.cvs in MOT format:
```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```
with ```<conf>, <x>, <y>, <z>``` as ```-1,-1,-1,-1```.

