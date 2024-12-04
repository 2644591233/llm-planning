import pybullet
import time
import pybullet_data
import numpy as np
import math
from environment import Robotiq2F85
from environment import PickPlaceEnv as env

half_plane_length = 0.3
half_plane_width = 0.3
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



def get_ee_pos(robot_id, tip_link_id):
    ee_xyz = np.float32(pybullet.getLinkState(robot_id, tip_link_id)[0])
    return ee_xyz

def check_proximity(robot, tool, body):
    ee_pos = np.array(pybullet.getLinkState(robot, tool)[0])
    tool_pos = np.array(pybullet.getLinkState(body, 0)[0])  
    vec = (tool_pos - ee_pos) / np.linalg.norm((tool_pos - ee_pos))
    ee_targ = ee_pos + vec
    ray_data = pybullet.rayTest(ee_pos, ee_targ)[0]
    obj, link, ray_frac = ray_data[0], ray_data[1], ray_data[2]
    return obj, link, ray_frac

def pushstep(gripper, robot_id, tip_link_id, start_orientation=None, end_orientation=None, action=None ):
    """push sth to target place"""
    # reserve a delta distance
    
    pick_pos, place_pos = np.array(action['pick'].copy()), np.array(action['push'].copy())
    theta = np.arctan2(place_pos[1] - pick_pos[1],place_pos[0] - pick_pos[0])
    # Set fixed primitive z-heights.
    
    pick_pos[2] = max(0.18, pick_pos[2])
    place_pos[2] = max(0.18, place_pos[2])
    print("delta_vector:",delta_vector)
    hover_xyz = np.float32([pick_pos[0] - np.sign(place_pos[0] - pick_pos[0])*np.cos(theta)*0.15, 
                          pick_pos[1] - np.sign(place_pos[1] - pick_pos[1])*np.sin(theta)*0.15, pick_pos[2]])
    pick_pos = np.float32([pick_pos[0] + np.sign(place_pos[0] - pick_pos[0])*np.cos(theta)*0.02, 
                          pick_pos[1] + np.sign(place_pos[1] - pick_pos[1])*np.sin(theta)*0.02, pick_pos[2]])
    if pick_pos.shape[-1] == 2:
      pick_xyz = np.append(pick_pos, 0.025)
    else:
      pick_xyz = pick_pos
    if place_pos.shape[-1] == 2:
      place_xyz = np.append(place_pos, 0.15)
    else:
      place_xyz = place_pos
    print("place_xyz:",place_xyz,"pick_xyz:",pick_xyz,"hover_xyz:",hover_xyz)
    # Move to object.
    ee_xyz = get_ee_pos(robot_id, tip_link_id)
    while np.linalg.norm(hover_xyz - ee_xyz) > 0.01:
      movep(hover_xyz, start_orientation)
      pybullet.stepSimulation()
      time.sleep(1./100.)
      ee_xyz = get_ee_pos(robot_id, tip_link_id)
      print(ee_xyz)
    print('moving to init done')
    while np.linalg.norm(pick_xyz - ee_xyz) > 0.001:
      movep(pick_xyz,start_orientation)
      pybullet.stepSimulation()
      time.sleep(1./100.)
      ee_xyz = get_ee_pos(robot_id, tip_link_id)
      print('ee_location:',ee_xyz,'we need:',pick_xyz)
    print('moving to obj done')
    # Push object.
    gripper.activate()
    for _ in range(240):
      pybullet.stepSimulation()
    # Move to place location.
    
    while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
      movep(place_xyz, end_orientation)
      pybullet.stepSimulation()
      time.sleep(1./100.)
      ee_xyz = get_ee_pos(robot_id, tip_link_id)
    print('Push object done')
    # reset the pose.
    gripper.release()
    for _ in range(400):
      pybullet.stepSimulation()
    ee_xyz = get_ee_pos(robot_id, tip_link_id)
    while np.linalg.norm(pick_xyz - ee_xyz) > 0.01:
      movep(pick_xyz, start_orientation)
      pybullet.stepSimulation()
      time.sleep(1./100.)
      ee_xyz = get_ee_pos(robot_id, tip_link_id)
    place_xyz  = np.float32([0, -0.5, 0.2])
    while np.linalg.norm(place_xyz - ee_xyz) > 0.01:
      movep(place_xyz)
      pybullet.stepSimulation()
      time.sleep(1./1000.)
      ee_xyz = get_ee_pos(robot_id, tip_link_id)
    print('reset done')
home_joints = (np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 3 * np.pi / 2, 0)  # ur5e Joint angles: (J0, J1, J2, J3, J4, J5). 
home_ee_euler = (np.pi, 0, np.pi)  # gripper (RX, RY, RZ) rotation in Euler angles.
ee_link_id = 9  # Link ID of UR5 end effector.
tip_link_id = 10  # Link ID of gripper finger tips.


physicsClient = pybullet.connect(pybullet.GUI)#or p.DIRECT for non-graphical version
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
pybullet.setGravity(0,0,-9.81)
pybullet.resetDebugVisualizerCamera(cameraDistance=1,cameraYaw=0,cameraPitch=-40,cameraTargetPosition=[0.1,-0.9,0.5])#转变视角
planeId = pybullet.loadURDF("plane.urdf")
robotId = pybullet.loadURDF("/source/ur5e/ur5e.urdf")
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
target_pos = [0.23526691, -0.34376354,   0.15]
obj_xyz = [ 0.13526691, -0.44376354,   0.15]

object_color = COLORS['red']
# object_shape = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.15])
# object_visual = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.02, 0.02, 0.15])
# object_id = pybullet.createMultiBody(0.01, object_shape, object_visual, basePosition=obj_xyz)
# pybullet.changeVisualShape(object_id, -1, rgbaColor=object_color)
radius = 0.03  # 半径
height = 0.3  # 高度

# 创建圆柱的碰撞形状和视觉形状
cylinder_collision_shape = pybullet.createCollisionShape(pybullet.GEOM_CYLINDER, radius=radius, height=height)
cylinder_visual_shape = pybullet.createVisualShape(pybullet.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=object_color)

# 创建圆柱体
cylinder_id = pybullet.createMultiBody(
    baseMass=0.01,  # 质量
    baseCollisionShapeIndex=cylinder_collision_shape,
    baseVisualShapeIndex=cylinder_visual_shape,
    basePosition=obj_xyz  # 圆柱底部放置在地面
)
pybullet.changeDynamics(cylinder_id, -1, lateralFriction=1, rollingFriction=0.5, spinningFriction=0.03)   # 横向摩擦力


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

theta = np.arctan2(target_pos[1] - obj_xyz[1], target_pos[0] - obj_xyz[0])
push_point = obj_xyz
push_point_end = target_pos
print("push_point:",push_point,"push_point_end:",push_point_end)
delta_vector = np.array(push_point_end) - np.array(push_point)
yaw = np.arctan2(delta_vector[1], delta_vector[0])
start_orientation = [-np.pi/2, 0, yaw - np.pi/2] 
end_orientation = [-np.pi/2, 0, yaw - np.pi/2]
for i in range (10000):
    pybullet.stepSimulation()
pushstep(gripper, robotId, tip_link_id, start_orientation=start_orientation, 
    end_orientation=end_orientation, 
    action={'pick': push_point, 'push':push_point_end} )
for i in range (500):
    pybullet.stepSimulation()
    if i % 50 == 0:
      print(np.float32(pybullet.getLinkState(robotId, tip_link_id)[0]))
    time.sleep(1./100.)
while True:
    
    pybullet.stepSimulation()