# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from .ant import Ant
from .humanoid_amp import HumanoidAMP
from .shadow_hand import ShadowHand
from .shadow_hand_grasp_v2 import ShadowHandGrasp
from .shadow_hand_manip import ShadowHandGrasp as ShadowHandManip
from .humanoid_amp_base import HumanoidAMPBase
from .fullbody import FullBody
from .fullbody_static import FullBodyStatic
from .moving_arm import MovingArm
from .fullbody_amp import FullBodyAMP
from .allegro_hand_grasp import AllegroHandGrasp
from .allegro_hand_tracking import AllegroHandTracking
from .allegro_hand_tracking_w_diffusion import AllegroHandTrackingDiff
from .allegro_hand_tracking_vision import AllegroHandTrackingVision
from .allegro_hand_tracking_generalist import AllegroHandTrackingGeneralist
from .allegro_hand_tracking_play import AllegroHandTrackingPlay
# AllegroHandTrackingPlayDemo
from .allegro_hand_tracking_play_demo import AllegroHandTrackingPlayDemo
from .allegro_hand_tracking_generalist_deploy import AllegroHandTrackingGeneralistDeploy
# AllegroHandTrackingGeneralistWForecasting
from .allegro_hand_tracking_generalist_wforecast import AllegroHandTrackingGeneralistWForecasting
from .allegro_hand_tracking_generalist_chunking import AllegroHandTrackingGeneralist as AllegroHandTrackingGeneralistChunking
# from .allegro_hand_tracking_generalist_v2 import AllegroHandTrackingGeneralistV2 # as AllegroHandTrackingGeneralistV2



# Mappings from strings to environments
isaacgym_task_map = {
    "Ant": Ant,
    # "Humanoid": Humanoid,
    "HumanoidAMP": HumanoidAMP,
    "HumanoidAMPBase": HumanoidAMPBase,
    "ShadowHand": ShadowHand,
    "ShadowHandGrasp": ShadowHandGrasp,
    "ShadowHandManip": ShadowHandManip,
    "FullBody": FullBody,
    "FullBodyStatic":FullBodyStatic,
    "MovingArm":MovingArm,
    "FullBodyAMP": FullBodyAMP,
    "AllegroHandGrasp": AllegroHandGrasp,
    "AllegroHandTracking": AllegroHandTracking,
    "AllegroHandTrackingDiff": AllegroHandTrackingDiff,
    "AllegroHandTrackingVision": AllegroHandTrackingVision,
    "AllegroHandTrackingGeneralist": AllegroHandTrackingGeneralist,
    'AllegroHandTrackingPlay': AllegroHandTrackingPlay,
    'AllegroHandTrackingPlayDemo': AllegroHandTrackingPlayDemo,
    'AllegroHandTrackingGeneralistDeploy': AllegroHandTrackingGeneralistDeploy,
    'AllegroHandTrackingGeneralistWForecasting': AllegroHandTrackingGeneralistWForecasting,
    'AllegroHandTrackingGeneralistChunking': AllegroHandTrackingGeneralistChunking,
    # 'AllegroHandTrackingGeneralistV2': AllegroHandTrackingGeneralistV2,
}
