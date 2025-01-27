from ...tools.general_utils import Enum
from . import elastic
from . import water
from . import snow
from . import sand
from . import simple_muscle
from . import fake_rigid
from . import diffaqua_muscle
from . import mud
from . import plasticine


class Material(Enum):
    Water = 0
    Elastic = 1
    Snow = 2
    Sand = 3
    Stationary = 4
    SimpleMuscle = 5
    FakeRigid = 6
    DiffAquaMuscle = 7
    Mud = 8
    Plasticine = 9


Material.Elastic.get_material_model = elastic.get_material_model
Material.Water.get_material_model = water.get_material_model
Material.Snow.get_material_model = snow.get_material_model
Material.Sand.get_material_model = sand.get_material_model
Material.SimpleMuscle.get_material_model = simple_muscle.get_material_model
Material.FakeRigid.get_material_model = fake_rigid.get_material_model
Material.DiffAquaMuscle.get_material_model = diffaqua_muscle.get_material_model
Material.Mud.get_material_model = mud.get_material_model
Material.Plasticine.get_material_model = plasticine.get_material_model
