from enum import Enum

class Modality(str, Enum):
    RGB = "rgb"
    DEPTH = "depth"
    POINT_CLOUD = "point_cloud"
    MESH = "mesh"
    SPLAT = "splat"
    LATENT = "latent"

class InputForm(str, Enum):
    SINGLE = "single"        # one tensor / object
    LIST = "list"            # unordered list
    TEMPORAL = "temporal"    # ordered sequence (time)