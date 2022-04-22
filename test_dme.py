import os
import struct
from dme_loader import DME, D3DXParamClass, D3DXParamType, jenkins

with open("dme/assets_x64_23/1598760830853998527.dme", "rb") as f:
    dme = DME.load(f)
