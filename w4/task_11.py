import cv2
import time
from pdb import set_trace as bp

import numpy as np
import pickle as pkl
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.feature_extraction.image import extract_patches_2d
from utils import imshow, mse, block_matching, OpticalFlowBlockMatching, compute_of_metrics, read_of, load_flow, flow_msen, flow_pepn
from of import optical_flow_block_matching, optical_flow_block_matching_mat

prev = cv2.imread('plots/colored_0/000045_10.png', cv2.IMREAD_GRAYSCALE).astype(float)
post = cv2.imread('plots/colored_0/000045_10.png', cv2.IMREAD_GRAYSCALE).astype(float)
# prev = cv2.imread('colored_0/000045_10.png', cv2.IMREAD_COLOR)
# post = cv2.imread('colored_0/000045_10.png', cv2.IMREAD_COLOR)

# tock = optical_flow_block_matching(prev,post, "forward", 32, 32,'MSE')
# print('Time: ', tock)

gt_of = read_of("plots/flow_noc/000045_10.png")
# gt_of2 = load_flow("plots/flow_noc/000045_10.png")

const_types = ["forward", "backward"]
distances = ['MSE','MAD']
block_sizes = [64,81,128]
search_radii = [8,16,32,48,21,41]

results = []

for const_type in const_types:
    for distance in distances:
        for block_size in block_sizes:
            for search_radius in search_radii:
                
                # estimated_of,elapsed_time = optical_flow_block_matching(prev, post, const_type, block_size, search_radius, distance)
                # estimated_of,elapsed_time = optical_flow_block_matching_mat(prev, post, const_type, block_size, search_radius, distance)
                
                start_time = time.time()
                estimated_of = block_matching(prev, post, const_type, block_size, search_radius, distance)
                end_time = time.time()
                elapsed_time = end_time-start_time
                # msen, pepn, of_error1 = compute_of_metrics(estimated_of, gt_of)
                
                msen = flow_msen(estimated_of,gt_of)
                pepn = flow_pepn(estimated_of,gt_of)
                results.append([const_type, distance, block_size, search_radius, elapsed_time, msen, pepn])
                print([const_type, distance, block_size, search_radius, elapsed_time, msen, pepn])
                
with open('results3.pkl', 'wb') as handle:
    pkl.dump(results, handle, protocol=pkl.HIGHEST_PROTOCOL)

# tock = optical_flow_block_matching(prev,post, "forward", 32, 32,'MSE')
# print('Time: ', tock)

# prev = cv2.imread('colored_0/000045_10.png', cv2.COLOR_BGR2RGB)
# post = cv2.imread('colored_0/000045_10.png', cv2.COLOR_BGR2RGB)

# start_time = time.time()
# estimated_of = block_matching(prev, post, const_type, block_size, search_radius, 'MSE')
# end_time = time.time()
# print('Team 5 2021: ',end_time-start_time)

# start_time = time.time()
# flow_func = OpticalFlowBlockMatching(type="FW", block_size=block_size, area_search=search_radius, error_function="SSD",window_stride=1)
# flow = flow_func.compute_optical_flow(prev, post)
# end_time = time.time()
# print('Team 4 2020: ',end_time-start_time)