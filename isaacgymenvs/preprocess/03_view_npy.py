import os
import sys
# sys.path.append('home/mingxian/project/IsaacGymEnvs/isaacgymenvs')
sys.path.append('.')
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonTree, SkeletonState
from poselib.poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion



if __name__ == '__main__':
    asset_path = '/home/mingxian/project/IsaacGymEnvs/assets/amp/motions/amp_humanoid_run.npy'
    curr_motion = SkeletonMotion.from_file(asset_path)
    curr_motion.skeleton_tree
    # curr_motion.fps


    mjcf_path = '/home/mingxian/project/IsaacGymEnvs/assets/hand/mjcf/open_ai_assets/hand/shadow_hand.xml'
    mjcf_path = '/home/mingxian/project/IsaacGymEnvs/assets/hand/mjcf/shadow_hand.xml'
    mjcf_path = '/home/mingxian/project/IsaacGymEnvs/assets/hand/mjcf/shadow_hand_24.xml'

    t = SkeletonTree.from_mjcf(mjcf_path)
    # t.node_names
    t._local_translation
    zero_pose = SkeletonState.zero_pose(t)
    zero_pose.global_translation
    plot_skeleton_state(zero_pose)
    pass 
    pass

