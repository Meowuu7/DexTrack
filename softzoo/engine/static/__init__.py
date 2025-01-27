from ...tools.general_utils import Enum
from . import bounding_box
from . import flat_surface
from . import terrain


class Static(Enum):
    BoundingBox = 0
    FlatSurface = 1
    Terrain = 2


Static.BoundingBox.add = bounding_box.add
Static.FlatSurface.add = flat_surface.add
Static.Terrain.add = terrain.add
