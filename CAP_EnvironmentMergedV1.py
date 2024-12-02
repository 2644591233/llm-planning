


import numpy as np
import cv2
from moviepy.editor import ImageSequenceClip

# imports for LMPs
from time import sleep
from shapely.geometry import *
from shapely.affinity import *
from environment import *
from LMP import *
from prompt import *
from global_param import *
'''
if not os.path.exists('ur5e/ur5e.urdf'):
    # Using gdown module to download files by their Google Drive ID
    gdown.download('https://drive.google.com/uc?id=1Cc_fDSBL6QiDvNT4dpfAEbhbALSVoWcc', 'ur5e.zip', quiet=False)
    gdown.download('https://drive.google.com/uc?id=1yOMEm-Zp_DL3nItG9RozPeJAmeOldekX', 'robotiq_2f_85.zip', quiet=False)
    gdown.download('https://drive.google.com/uc?id=1GsqNLhEl9dd4Mc3BM0dX3MibOI1FVWNM', 'bowl.zip', quiet=False)

    # Unzip the downloaded files
    subprocess.run(['unzip', 'ur5e.zip'])
    subprocess.run(['unzip', 'robotiq_2f_85.zip'])
    subprocess.run(['unzip', 'bowl.zip'])

# Show useful GPU info
subprocess.run(['nvidia-smi'])
'''


def cv2_imshow(image):
  cv2.imshow('Image', image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
  
if __name__ == "__main__":
  #@title Initialize Env { vertical-output: true }
  num_blocks = 1 #@param {type:"slider", min:0, max:4, step:1}
  num_bowls = 1 #@param {type:"slider", min:0, max:4, step:1}
  num_penpots = 1 #@param {type:"slider", min:0, max:4, step:1}
  num_pencils = 3 #@param {type:"slider", min:0, max:4, step:1}
  high_resolution = True #@param {type:"boolean"}
  high_frame_rate = False #@param {type:"boolean"}

  # setup env and LMP
  env = PickPlaceEnv(render=True, high_res=high_resolution, high_frame_rate=high_frame_rate)
  block_list = np.random.choice(ALL_BLOCKS, size=num_blocks, replace=False).tolist()
  bowl_list = np.random.choice(ALL_BOWLS, size=num_bowls, replace=False).tolist()
  penpot_list = np.random.choice(ALL_PENPOTS, size=num_penpots, replace=False).tolist()
  pencil_list = np.random.choice(ALL_PENCILS, size=num_pencils, replace=False).tolist()
  obj_list = block_list + bowl_list + penpot_list + pencil_list
  _ = env.reset(obj_list)
  lmp_tabletop_ui = setup_LMP(env, cfg_tabletop)

  # display env
  cv2_imshow(cv2.cvtColor(env.get_camera_image(), cv2.COLOR_BGR2RGB))

  print('available objects:')
  print(obj_list)
  description = ''
  #@title Interactive Demo { vertical-output: true }
  while True:
    print("do you need video learning(Y/N)")
    if(input()=='Y'):
        print("enter your video address")
        video_address = input()
        print("enter your query")
        user_query = input()
        user_query = f'{prompt_video}\n{user_query}'
        description = get_video_description(video_address,user_query)
        print(description)
    print("Please enter your command 请用英文输入机器人操作指令")
    user_input = f'{input()}{description}' #@param {allow-input: true, type:"string"}

    env.cache_video = []

    print('Running policy and recording video...')
    ##
    lmp_tabletop_ui(user_input, f'objects = {env.object_list}')

    # render video
    if env.cache_video:
      rendered_clip = ImageSequenceClip(env.cache_video, fps=35 if high_frame_rate else 25)
      looped_clip = rendered_clip.loop(n=20)
      looped_clip.write_videofile("looped_video.mp4", codec="libx264")
