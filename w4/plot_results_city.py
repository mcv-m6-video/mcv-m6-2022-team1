import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from pathlib import Path
from os.path import join
import os
import json

origin = Path('C:/Users/Dani/Downloads/w4')
tests = ['testS01','testS03','testS04']
detectors = ['faster_c4','faster_dc5','faster_fpn','retina_fpn']

for test in tests:
    for detector in detectors:
        data_path = join(join(origin,test),detector)
        cameras_path = [x[0] for x in os.walk(data_path)][1:]
        for camera_path in cameras_path:
            not_purging = pd.read_csv(join(camera_path, "summary.csv"))
            purging = pd.read_csv(join(camera_path, "summary_purge.csv"))    
            
            instances = pd.read_json(join(camera_path,'coco_instances_results.json'))
            
            # with open( join(camera_path,'coco_instances_results.json'), 'r') as myfile:
            #     instances = myfile.read()