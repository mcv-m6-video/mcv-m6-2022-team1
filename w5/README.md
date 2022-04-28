# Week 5: Object Tracking

The goal for this week is to give results for SEQ3 for Multi-target single-camera tracking, assigning a unique
individual ID for every vehicle. For Multi-target multi-camera tracking, for a collection of cameras, keep track of
individual cars, assigning them the same unique ID in every camera (cars could appear in different time instances, 
perspectives, etc.)

## Running the code

For Task 1, implementations from w3 and w4 are used. Please refer to them for more details ([**week 3**](../w3) and [**week 4**](../w4)).

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
### Data creation

In order to begin with the training, one must first create the dataset. For that run the script 
[**create_car_dataset.py**](./misc/create_car_dataset.py), where a hardcode variable ```gt_path``` indicates the path to AI 
Cities dataset.

### Metric Learning Train
To start the metric learning process one might run the script [**train_metric.py**](./train_metric.py), indicating the
 json file as the first argument  ```config_path``` (refer to [**configs**](./configs) for an example) and the path to 
the AI Cities dataset as second argument 
```dataset_path``` file . One can follow the training process using [**w&b**](https://wandb.ai/) tool.

### Multi Camera Tracking
For the multi camera tracking several variables are hardcoded:
- ```results_path```: path to the folder containing the cropped car images clustered by id.
- ```root_dataset_path```: dataset to the AI Cities dataset.
- ```model_weights```: path where the model weights are stored.

Once they are all set, run the script to find the car id's pairs.





 

