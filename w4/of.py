import time
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from pdb import set_trace as bp
from utils import mse, mad

def optical_flow_block_matching(prev,post, const_type, block_size, search_radius, distance):
    
    tick = time.time()

    if const_type == "forward":
        reference = prev
        target = post
    elif const_type == "backward":
        reference = post
        target = prev
        
    if distance == 'MSE':
        dist = mse
    elif distance == 'MAD':
        dist = mad
        
    assert reference.shape == target.shape, "Image sizes do not match"

    reference = reference.astype(float)
    target = target.astype(float)

    height, width = reference.shape[0:2]
    nheight, nwidth = int(np.ceil(height / block_size)) * block_size, \
                      int(np.ceil(width / block_size)) * block_size

    if reference.ndim == 3:
        channels = reference.shape[2]
    else:
        channels = 1
        
    vblocks = nheight // block_size
    hblocks = nwidth // block_size

    nreference = np.full((nheight, nwidth), np.mean(reference.flatten()))
    nreference[:height, :width] = reference

    nreference = nreference.reshape(
        vblocks,
        block_size,
        hblocks,
        block_size,
        channels #if rgb
    )
    nreference = nreference.swapaxes(1, 2).reshape((-1, block_size, block_size))

    of = np.zeros((height, width, 3), dtype=float)
    
    
    for ind_ref, block in enumerate(nreference):
        top = max(0, ((ind_ref % vblocks) * block_size) - search_radius)
        left = max(0, ((ind_ref % hblocks) * block_size) - search_radius)
        bot = min(height, block_size + top + search_radius)
        right = min(width, block_size + left + search_radius)

        patches = extract_patches_2d(
            target[top:bot, left:right],
            patch_size=(block_size, block_size),
        )

        # mse_matrix = ((block[None, :] - patches[:, None])**2).mean()
        # mad_matrix = (np.abs(block[None, :] - patches[:, None])).mean()
        
        dist_matrix = np.zeros(len(patches))
        for ind_tar, patch in enumerate(patches):
            dist_matrix[ind_tar] = dist(block, patch)
    
        index = np.argmin(dist_matrix[:,0,0,0])
        u,v = np.unravel_index(index, (block.shape[0]+1,block.shape[1]+1), 'F')
        
        of[top:bot, left:right, :] = [u, v, 1]
        
    if const_type == "backward":
        of[:,:,0:2] = -of[:,:,0:2]
        
    tock = time.time() - tick
    
    return of, tock

def optical_flow_block_matching_mat(prev,post, const_type, block_size, search_radius, distance):
    
    tick = time.time()

    if const_type == "forward":
        reference = prev
        target = post
    elif const_type == "backward":
        reference = post
        target = prev
        
    assert reference.shape == target.shape, "Image sizes do not match"

    reference = reference.astype(float)
    target = target.astype(float)

    height, width = reference.shape[0:2]
    nheight, nwidth = int(np.ceil(height / block_size)) * block_size, \
                      int(np.ceil(width / block_size)) * block_size

    if reference.ndim == 3:
        channels = reference.shape[2]
    else:
        channels = 1
        
    vblocks = nheight // block_size
    hblocks = nwidth // block_size

    nreference = np.full((nheight, nwidth), np.mean(reference.flatten()))
    nreference[:height, :width] = reference

    nreference = nreference.reshape(
        vblocks,
        block_size,
        hblocks,
        block_size,
        channels #if rgb
    )
    nreference = nreference.swapaxes(1, 2).reshape((-1, block_size, block_size))

    of = np.zeros((height, width, 3), dtype=float)
    
    
    if distance == 'MSE':    
        for ind_ref, block in enumerate(nreference):
            top = max(0, ((ind_ref % vblocks) * block_size) - search_radius)
            left = max(0, ((ind_ref % hblocks) * block_size) - search_radius)
            bot = min(height, block_size + top + search_radius)
            right = min(width, block_size + left + search_radius)
    
            patches = extract_patches_2d(
                target[top:bot, left:right],
                patch_size=(block_size, block_size),
            )
    
            dist_matrix = ((block[None, :] - patches[:, None])**2)
        
            index = np.argmin(dist_matrix.sum(axis = (1,2,3)))
            
            dim_u = bot-top-block_size+1
            dim_v = right-left-block_size+1
            u,v = np.unravel_index(index, (dim_u,dim_v), 'C')
            
            of[top:bot, left:right, :] = [u, v, 1]
            # of[top:bot, left:right, :] = [u - np.round(dim_u/2), v - np.round(dim_v/2), 1]
            # of[top:bot, left:right, :] = [u - np.round((bot-top)/2), v - np.round((right-left)/2), 1]
            # of[top:bot, left:right, :] = [u - np.round((block.shape[0]+1)/2), v - np.round((block.shape[1]+1)/2), 1]
    elif distance == 'MAD':
        for ind_ref, block in enumerate(nreference):
            top = max(0, ((ind_ref % vblocks) * block_size) - search_radius)
            left = max(0, ((ind_ref % hblocks) * block_size) - search_radius)
            bot = min(height, block_size + top + search_radius)
            right = min(width, block_size + left + search_radius)
    
            patches = extract_patches_2d(
                target[top:bot, left:right],
                patch_size=(block_size, block_size),
            )
    
            dist_matrix = (np.abs(block[None, :] - patches[:, None]))
        
            index = np.argmin(dist_matrix.sum(axis = (1,2,3)))
            
            dim_u = bot-top-block_size+1
            dim_v = right-left-block_size+1
            
            u,v = np.unravel_index(index, (dim_u,dim_v), 'C')
            
            of[top:bot, left:right, :] = [u, v, 1]
            # of[top:bot, left:right, :] = [u - np.round((bot-top)/2), v - np.round((right-left)/2), 1]
            # of[top:bot, left:right, :] = [u - np.round((block.shape[0]+1)/2), v - np.round((block.shape[1]+1)/2), 1]
        
    if const_type == "backward":
        of[:,:,0:2] = -of[:,:,0:2]
        
    tock = time.time() - tick
    
    return of, tock