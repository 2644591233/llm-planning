import numpy as np
objects = [...]

block_names = ['yellow block', 'brown block', 'cyan block']
bowl_positions = [get_obj_pos(name) for name in objects if 'bowl' in name]
bowl_position = np.mean(bowl_positions, axis=0)  
# Assuming \"the bowl\" refers to the average position of all bowls
block_positions = get_obj_positions_np(block_names)
closest_block_idx = get_closest_idx(points=block_positions, point=bowl_position)
closest_block_name = block_names[closest_block_idx]
ret_val = closest_block_name

def get_obj_positions_np(block_names):   
  positions = []   
  for block_name in block_names:
    pos_2d = get_obj_pos(block_name)        
    positions.append(pos_2d)    
  return np.array(positions)

def get_closest_idx(points, point):    
# Compute the Euclidean distance from each point in points to the reference point
  distances = np.linalg.norm(points - point, axis=1)    
# Return the index of the point with the smallest distance\n    closest_block_idx = np.argmin(distances)
  return closest_block_idx

def get_obj_pos(): #predefined
  ...