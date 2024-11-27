import pybullet
import time
import pybullet_data
import numpy as np
import math
from code_as_policies_interactive_demo_copy import Robotiq2F85
from code_as_policies_interactive_demo_copy import PickPlaceEnv as env

half_plane_length = 1
half_plane_width = 1
half_plane_thick = 0.001
# # Global constants: pick and place objects, colors, workspace bounds
COLORS = {
    'blue':   (78/255,  121/255, 167/255, 255/255),
    'red':    (255/255,  87/255,  89/255, 255/255),
    'green':  (89/255,  169/255,  79/255, 255/255),
    'orange': (242/255, 142/255,  43/255, 255/255),
    'yellow': (237/255, 201/255,  72/255, 255/255),
    'purple': (176/255, 122/255, 161/255, 255/255),
    'pink':   (255/255, 157/255, 167/255, 255/255),
    'cyan':   (118/255, 183/255, 178/255, 255/255),
    'brown':  (156/255, 117/255,  95/255, 255/255),
    'gray':   (186/255, 176/255, 172/255, 255/255),
}

CORNER_POS = {
  'top left corner':     (-half_plane_width + 0.05, -0.5+half_plane_length - 0.05, 0),
  'top side':            (0,                        -0.5+half_plane_length - 0.05, 0),
  'top right corner':    (half_plane_width - 0.05,  -0.5+half_plane_length - 0.05, 0),
  'left side':           (-half_plane_width + 0.05, -0.5,                          0),
  'middle':              (0,                        -0.5,                          0),
  'right side':          (half_plane_width - 0.05,  -0.5,                          0),
  'bottom left corner':  (-half_plane_width + 0.05, -0.5-half_plane_length + 0.05, 0),
  'bottom side':         (0,                        -0.5-half_plane_length + 0.05, 0),
  'bottom right corner': (half_plane_width - 0.05,  -0.5-half_plane_length + 0.05, 0),
}

ALL_BLOCKS = ['blue block', 'red block', 'green block', 'orange block', 'yellow block', 'purple block', 'pink block', 'cyan block', 'brown block', 'gray block']
ALL_BOWLS = ['blue bowl', 'red bowl', 'green bowl', 'orange bowl', 'yellow bowl', 'purple bowl', 'pink bowl', 'cyan bowl', 'brown bowl', 'gray bowl']

PIXEL_SIZE = 0.00267857
BOUNDS = np.float32([[-0.3, 0.3], [-0.8, -0.2], [0, 0.15]])  # X Y Z


def pushstep(self, action=None):
    """push sth to target place"""
    # reserve a delta distance
    pick_pos, place_pos = action['pick'].copy(), action['push'].copy()
    delta_vector = (place_pos - pick_pos)/np.linalg.norm(place_pos - pick_pos)
    # Set fixed primitive z-heights.
    hover_xyz = np.float32([pick_pos[0] + delta_vector[0], pick_pos[1] + delta_vector[1], pick_pos[2]])

    orientation = []
    if pick_pos.shape[-1] == 2:
      pick_xyz = np.append(pick_pos, 0.025)
    else:
      pick_xyz = pick_pos
      pick_xyz[2] = 0.025
    if place_pos.shape[-1] == 2:
      place_xyz = np.append(place_pos, 0.15)
    else:
      place_xyz = place_pos
      place_xyz[2] = 0.15

    # Move to object.
    ee_xyz = env.get_ee_pos()
    while np.linalg.norm(hover_xyz - ee_xyz) > 0.01:
      env.movep(hover_xyz, orientation)
      env.step_sim_and_render()
      ee_xyz = env.get_ee_pos()

    while np.linalg.norm(pick_xyz - ee_xyz) > 0.01:
      env.movep(pick_xyz)
      env.step_sim_and_render()
      ee_xyz = env.get_ee_pos()

    # Pick up object.
    self.gripper.activate()
    for _ in range(240):
      self.step_sim_and_render()
    while np.linalg.norm(hover_xyz - ee_xyz) > 0.01:
      self.movep(hover_xyz)
      self.step_sim_and_render()
      ee_xyz = self.get_ee_pos()

    for _ in range(50):
      self.step_sim_and_render()

    # Move to place location.
    while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
      self.movep(place_xyz)
      self.step_sim_and_render()
      ee_xyz = self.get_ee_pos()

    # Place down object.
    while (not self.gripper.detect_contact()) and (place_xyz[2] > 0.03):
      place_xyz[2] -= 0.001
      self.movep(place_xyz)
      for _ in range(3):
        self.step_sim_and_render()
    self.gripper.release()
    for _ in range(240):
      self.step_sim_and_render()
    place_xyz[2] = 0.2
    ee_xyz = self.get_ee_pos()
    while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
      self.movep(place_xyz)
      self.step_sim_and_render()
      ee_xyz = self.get_ee_pos()
    place_xyz = np.float32([0, -0.5, 0.2])
    while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
      self.movep(place_xyz)
      self.step_sim_and_render()
      ee_xyz = self.get_ee_pos()

    observation = self.get_observation()
    reward = self.get_reward()
    done = False
    info = {}
    return observation, reward, done, info

home_joints = (np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 3 * np.pi / 2, 0)  # ur5e Joint angles: (J0, J1, J2, J3, J4, J5). 
home_ee_euler = (np.pi, 0, np.pi)  # gripper (RX, RY, RZ) rotation in Euler angles.
ee_link_id = 9  # Link ID of UR5 end effector.
tip_link_id = 10  # Link ID of gripper finger tips.


physicsClient = pybullet.connect(pybullet.GUI)#or p.DIRECT for non-graphical version
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
pybullet.setGravity(0,0,-9.81)
pybullet.resetDebugVisualizerCamera(cameraDistance=2,cameraYaw=0,cameraPitch=-40,cameraTargetPosition=[0.5,-0.9,0.5])#转变视角
planeId = pybullet.loadURDF("plane.urdf")
robotId = pybullet.loadURDF("./ur5e/ur5e.urdf")
joint_ids = [pybullet.getJointInfo(robotId, i) for i in range(pybullet.getNumJoints(robotId))]
joint_ids = [j[0] for j in joint_ids if j[2] == pybullet.JOINT_REVOLUTE]

robotStartPos = [0,0,0]
#0
robotStartOrientation = pybullet.getQuaternionFromEuler([0,0,0])#[roll,pitch,yaw]
# #1
# robotStartOrientation = pybullet.getQuaternionFromEuler([math.pi/2,0,0])#[roll,pitch,yaw]
# #2
# robotStartOrientation = pybullet.getQuaternionFromEuler([0,math.pi/2,0])#[roll,pitch,yaw]
# #3
# robotStartOrientation = pybullet.getQuaternionFromEuler([0,0,math.pi/2])#[roll,pitch,yaw]

print("robotStartOrientation:",robotStartOrientation)

pybullet.resetBasePositionAndOrientation(robotId,robotStartPos,robotStartOrientation)
for i in range(len(joint_ids)):
      pybullet.resetJointState(robotId, joint_ids[i], home_joints[i])
gripper = Robotiq2F85(robotId, ee_link_id)
gripper.release()

plane_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[half_plane_length, half_plane_width, half_plane_thick])
plane_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[half_plane_length, half_plane_width, half_plane_thick])
plane_id = pybullet.createMultiBody(0, plane_shape, plane_visual, basePosition=[0, -0.5, 0])
pybullet.changeVisualShape(plane_id, -1, rgbaColor=[0.2, 0.2, 0.2, 1.0])
start_pos = [ 0.04624302, -0.48661476,  0.3]
target_pos = [ 0.13526691, -0.64376354,   0.3]

obj_xyz = [ 0.13526691, -0.64376354,   0.1]
object_color = COLORS['red']
object_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
object_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
object_id = pybullet.createMultiBody(0.01, object_shape, object_visual, basePosition=obj_xyz)
pybullet.changeVisualShape(object_id, -1, rgbaColor=object_color)

# object_list = object_list
#     self.obj_name_to_id = {}
#     obj_xyz = start_pos
#     for obj_name in object_list:
#       if ('block' in obj_name) or ('bowl' in obj_name):

#         # Get random position 15cm+ from other objects.
#         while True:
#           rand_x = np.random.uniform(BOUNDS[0, 0] + 0.1, BOUNDS[0, 1] - 0.1)
#           rand_y = np.random.uniform(BOUNDS[1, 0] + 0.1, BOUNDS[1, 1] - 0.1)
#           rand_xyz = np.float32([rand_x, rand_y, 0.03]).reshape(1, 3)
#           if len(obj_xyz) == 0:
#             obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
#             break
#           else:
#             nn_dist = np.min(np.linalg.norm(obj_xyz - rand_xyz, axis=1)).squeeze()
#             if nn_dist > 0.15:
#               obj_xyz = np.concatenate((obj_xyz, rand_xyz), axis=0)
#               break

#         object_color = COLORS[obj_name.split(' ')[0]]
#         object_type = obj_name.split(' ')[1]
#         object_position = rand_xyz.squeeze()
#         if object_type == 'block':
#           object_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
#           object_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.02])
#           object_id = pybullet.createMultiBody(0.01, object_shape, object_visual, basePosition=object_position)
#         elif object_type == 'bowl':
#           object_position[2] = 0
#           object_id = pybullet.loadURDF("bowl/bowl.urdf", object_position, useFixedBase=1)
#         pybullet.changeVisualShape(object_id, -1, rgbaColor=object_color)
#         self.obj_name_to_id[obj_name] = object_id
def movep(position,orientation = None):
  if orientation == None:
    orientation = [np.pi, 0, np.pi]
  targetPositionsJoints = pybullet.calculateInverseKinematics( bodyUniqueId=robotId,
      endEffectorLinkIndex=tip_link_id,
      targetPosition=position,
      targetOrientation=pybullet.getQuaternionFromEuler(orientation), ##if we need action push, just change ending orientation
      maxNumIterations=100)
  servoj(targetPositionsJoints)
def servoj(joints):  
  pybullet.setJointMotorControlArray(bodyIndex=robotId,
    jointIndices=joint_ids,
    controlMode=pybullet.POSITION_CONTROL,
    targetPositions=joints,
    positionGains=[0.01]*6)
for i in range (500):
    pybullet.stepSimulation()
    if i % 50 == 0:
      print(np.float32(pybullet.getLinkState(robotId, tip_link_id)[0]))
    time.sleep(1./100.)
print(pybullet.getEulerFromQuaternion(np.float32(pybullet.getLinkState(robotId, tip_link_id)[1])))
while True:
    
    pybullet.stepSimulation()