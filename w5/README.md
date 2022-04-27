# Week 5: Object Tracking

The goal for this week is to give results for SEQ3 for Multi-target single-camera tracking, assigning a unique
individual ID for every vehicle. For Multi-target multi-camera tracking, for a collection of cameras, keep track of
individual cars, assigning them the same unique ID in every camera (cars could appear in different time instances, 
perspectives, etc.)

## Running the code

For Task 1, implementations from w3 and w4 are used. Please refer to them for more details [**Week 3**](./w3) [[**Week 4**](./w4)]

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
`

