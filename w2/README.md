# Week 2: Background estimation

## Task overview

---
### Task 1: Gaussian modelling for background estimation

The task is implemented in [```t1.py```](./t1.py).

This task consists on the implementation of a simple single-gaussian background
estimation model. In our code, this is written in the script file
[```background_estimation.py```](./background_estimation.py), specifically the
class ```StillBackgroundEstimatorGrayscale```. To train an instance of this
model, one should run the following

```python
from background_estimation import StillBackgroundEstimatorGrayscale
from data import FrameLoader
from pathlib import Path

path = Path(...)
train_loader = FrameLoader(path, perc, partition) # See implementation for details

estimator = StillBackgroundEstimatorGrayscale(train_loader, tol)
estimator.fit()
```

For inference one should run

```python
output_mask = estimator.predict(rgb_image)
```

This output mask may be refined using the ```cleanup_mask()``` function, which
performs median filtering and some morphology. To obtain bounding boxes, run
```get_bboxes()``` on a mask output.

It should be noted this implementation will load the entire video on memory and
compute statistics from it, hence using a limited amount of frames is
recommended (we considered using running statistics for this, but since the
number of frames so happens to fit in our machines we kept it as is).

The following image shows an example prediction on frame 1617 of the provided
video sequence (S03 - c010) from the AI Cities dataset.

![](./plots/t11_mask_predict.png)

---
