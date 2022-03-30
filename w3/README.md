# Week 3: Object Tracking

# Task 2.1

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

