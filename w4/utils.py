import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2 
from skimage.feature import match_template

def imshow(img):
    plt.figure(dpi=150)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    plt.close()
    
def mse(ref, tar):
    return ((ref - tar)**2).mean()
def mad(ref, tar):
    return (np.abs(ref - tar)).mean()

def block_matching(img_past, img_future, estimation_dir, block_size, search_border, method):
    height, width = img_past.shape[:2]
    of = np.zeros((height, width, 3), dtype=float)

    # Define reference/target image
    if estimation_dir == "backward":
        im_ref = img_future
        im_target = img_past
    elif estimation_dir == "forward":
        im_ref = img_past
        im_target = img_future

    # Iter rows
    for i in tqdm(range(0, height - block_size, block_size)):
        # Iter cols
        for j in range(0, width - block_size, block_size):

            # Crop reference block and target area to search
            i_start_area = max(i - search_border, 0)
            j_start_area = max(j - search_border, 0)

            i_end =  min(i + block_size + search_border, height)
            j_end = min(j + block_size + search_border, width)

            ref_block = im_ref[i: i + block_size, j: j + block_size]
            target_area = im_target[i_start_area: i_end, j_start_area:j_end]

            # Obtain the position of the block with lower distance
            pos = block_matching_block(ref_block, target_area, method)

            # Scale position to image axis
            u = pos[1] - (j - j_start_area)
            v = pos[0] - (i - i_start_area)

            # Save the optical flow (all pixels are considered as valid: last axis = 1)
            if estimation_dir == "backward":
                of[i:i + block_size, j:j + block_size, :] = [-u, -v, 1]
            if estimation_dir == "forward":
                of[i:i + block_size, j:j + block_size, :] = [u, v, 1]
    return of

def block_matching_block(ref_block, target_area, method):
    height_ref = ref_block.shape[0]
    width_ref = ref_block.shape[1]
    min_dist = np.zeros(shape=(target_area.shape[0] - height_ref, target_area.shape[1] - width_ref))

    # Exhaustive search
    if method == "SSD" or method == "SAD" or method == "MSE" or method == "MAD" or method == "euclidean":
        for i in range(0, target_area.shape[0] - height_ref):
            for j in range(0, target_area.shape[1] - width_ref):
                target_block = target_area[i: i + height_ref, j: j + width_ref]
                
                dist = compute_dist(ref_block, target_block, method)

                min_dist[i, j] = dist
        
        arg_min_dist = np.unravel_index(np.argmin(min_dist, axis=None), min_dist.shape)
        return arg_min_dist

    # Match template (Sklearn)
    if method == "template":
        corr = match_template(target_area, ref_block)
        arg_min_dist = np.unravel_index(np.argmin(corr, axis=None), corr.shape)
        return arg_min_dist

    # Match template (OpenCV)
    if method == "template1": metric = cv2.TM_CCOEFF
    if method == "template2": metric = cv2.TM_CCOEFF_NORMED
    if method == "template3": metric = cv2.TM_CCORR
    if method == "template4": metric = cv2.TM_CCORR_NORMED
    if method == "template5": metric = cv2.TM_SQDIFF
    if method == "template6": metric = cv2.TM_SQDIFF_NORMED

    res = cv2.matchTemplate(target_area,ref_block,metric)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the metric is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if metric in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    arg_min_dist = (top_left[1], top_left[0])

    return arg_min_dist

def compute_dist(ref_b, target_b, method = "SSD"):

    if method == "SSD":
        distance = np.sum((target_b - ref_b) ** 2)
    if method == "SAD":
        distance = np.sum(np.abs(target_b - ref_b))
    if method == 'MSE':
        distance = np.mean((target_b - ref_b) ** 2)
    if method == 'MAD':
        distance = np.mean(np.abs(target_b - ref_b))
    if method == 'euclidean':
        distance = np.sqrt(np.sum((target_b - ref_b) ** 2))
    return distance

def compute_of_metrics(flow, gt):
    # Binary mask to discard non-occluded areas
    #non_occluded_areas = gt[:,:,2] != 0

    # Only for the first 2 channels
    square_error_matrix = (flow[:,:,0:2] - gt[:,:,0:2]) ** 2
    square_error_matrix_valid = square_error_matrix*np.stack((gt[:,:,2],gt[:,:,2]),axis=2)
    #square_error_matrix_valid = square_error_matrix[non_occluded_areas]

    #non_occluded_pixels = np.shape(square_error_matrix_valid)[0]
    non_occluded_pixels = np.sum(gt[:,:,2] != 0)

    # Compute MSEN
    pixel_error_matrix = np.sqrt(np.sum(square_error_matrix_valid, axis= 2)) # Pixel error for both u and v
    msen = (1/non_occluded_pixels) * np.sum(pixel_error_matrix) # Average error for all non-occluded pixels
    
    # Compute PEPN
    erroneous_pixels = np.sum(pixel_error_matrix > 3)
    pepn = erroneous_pixels/non_occluded_pixels
    
    return msen, pepn, pixel_error_matrix

def read_of(flow_path):

    flow_raw = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED).astype(np.double)

    # Transform data (DevKit Stereo Flow - KITTI)
    flow_u = (flow_raw[:,:,2] - 2**15) / 64.0
    flow_v = (flow_raw[:,:,1] - 2**15) / 64.0
    flow_valid = flow_raw[:,:,0] == 1

    # Set to 0 the points where the flow is not valid
    flow_u[~flow_valid] = 0
    flow_v[~flow_valid] = 0

    # Reorder channels
    return np.stack((flow_u, flow_v, flow_valid), axis=2)

def load_flow(path):
  flow = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.double)

  u_flow = (flow[:,:,2] - 2**15)/ 64
  v_flow = (flow[:,:,1] - 2**15)/ 64
  b_valid = flow[:,:,0]

  # # remove invalid points
  # u_flow[b_valid == 0] = 0
  # v_flow[b_valid == 0] = 0

  # flow = [u_flow, v_flow, b_valid]
  flow = np.stack((u_flow, v_flow, b_valid),axis=-1)
  return flow

def flow_error_distance(gt,kitti):
  return np.sqrt(np.square(gt[:,:,0] - kitti[:,:,0]) + np.square(gt[:,:,1] - kitti[:,:,1]))

def flow_msen(gt, kitti):
    return np.mean(flow_error_distance(gt,kitti)[gt[:,:,2]==1])

# percentage of Erroneous Pixels in Non-occluded areas
def flow_pepn(gt, kitti, th=3):
    return 100 * (flow_error_distance(gt,kitti)[gt[:,:,2]==1] > th).sum() / (gt[:,:,2] != 0).sum()

class OpticalFlowBlockMatching:
    def __init__(self, type="FW", block_size=5, area_search=40, error_function="SSD", window_stride=5):
        """
        Class that performs optical flow with the block matching algorithm
        type :
            - "FW": for forward block matching
            - "BW": for bckward block matching
        block_size: size of the blocks
        area_search: number of pixels of expected movement in every direction
        error_function:
            - "SSD": sum of squared differences
            - "SAD": sum of absolute differences
        window_stride: step to look for matching block
        """
        if type == "FW" or type == "BW":
            self.type = type
        else:
            raise NameError("Unexpected type in OpticalFlowBlockMatching")
        self.block_size = block_size
        self.area_search = area_search
        self.window_stride = window_stride
        if error_function == "SSD":
            self.error_function = self.SSD
        elif error_function == "SAD":
            self.error_function = self.SAD
        else:
            raise NameError("Unexpected error function name in OpticalFlowBlockMatching")

    def SSD(self, block1, block2):
        return float(np.sum(np.power(block1 - block2, 2)))

    def SAD(self, block1, block2):
        return float(np.sum(np.abs(block1 - block2)))

    def compute_optical_flow(self, first_frame, second_frame):
        optical_flow = np.zeros((first_frame.shape[0], first_frame.shape[1], 3))
        if self.type == "FW":
            reference_image = first_frame.astype(float) / 255
            estimated_frame = second_frame.astype(float) / 255
        else:
            reference_image = second_frame.astype(float) / 255
            estimated_frame = first_frame.astype(float) / 255

        for i in range(self.block_size//2, reference_image.shape[0] - self.block_size // 2, self.block_size):
            for j in range(self.block_size//2, reference_image.shape[1] - self.block_size // 2, self.block_size):
                block_ref = reference_image[i - self.block_size // 2:i + self.block_size // 2 + 1, j - self.block_size // 2:j + self.block_size // 2 + 1, :]
                optical_flow[i - self.block_size // 2:i + self.block_size // 2, j - self.block_size // 2:j + self.block_size // 2, :] = self.find_deviation_matching_block(block_ref, estimated_frame, (i,j))
                #optical_flow[i, j, :] = self.find_deviation_matching_block(block_ref, estimated_frame, (i,j))
        if self.type == "FW":
            return optical_flow
        else:
            return optical_flow * -1

    def find_deviation_matching_block(self, block_ref, estimated_frame, position):
        min_likelihood = float('inf')
        min_direction = (0, 0)
        for i in range(max(self.block_size//2, position[0]-self.area_search), min(estimated_frame.shape[0] - self.block_size // 2, position[0]+self.area_search), self.window_stride):
            for j in range(max(self.block_size//2, position[1]-self.area_search), min(estimated_frame.shape[1] - self.block_size // 2, position[1]+self.area_search), self.window_stride):
                block_est = estimated_frame[i - self.block_size // 2:i + self.block_size // 2 + 1, j - self.block_size // 2:j + self.block_size // 2 + 1, :]
                likelihood = self.error_function(block_ref, block_est)
                if likelihood < min_likelihood:
                    min_likelihood = likelihood
                    min_direction = (i - position[0],j - position[1]) # TODO: SURE?
                elif likelihood == min_likelihood and np.sum(np.power(min_direction, 2)) > j ** 2 + i ** 2:
                    min_direction = (i - position[0],j - position[1]) # TODO: SURE?
        ret_block = np.ones((self.block_size, self.block_size, 3))
        ret_block[:, :, 0] = min_direction[1]
        ret_block[:, :, 1] = min_direction[0]
        return ret_block
        #return [min_direction[0], min_direction[1], 1]