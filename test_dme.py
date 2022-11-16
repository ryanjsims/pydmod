import os
import struct
from argparse import ArgumentParser
from dme_loader import DME, D3DXParamClass, D3DXParamType, jenkins
from numpy import matrix
# from english_words import english_words_lower_alpha_set as words

import logging

logger = logging.getLogger("DME testing")

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

logging.basicConfig(level=logging.INFO, handlers=[handler])

parser = ArgumentParser(description="DME Tester for cmd line inspection of DME files")
parser.add_argument("file", help="The file to load")
args = parser.parse_args()

with open(args.file, "rb") as f:
    dme = DME.load(f)

for bone in dme.bones:
    print(bone)
    print(*list(map(lambda x: -x, (bone.inverse_bind_pose * [[0], [0], [0], [1]]).flatten().tolist()[0])))
    print()

def bone_name(dme: DME, name: str):
    namehash = jenkins.oaat(name.encode("utf-8"))
    for bone in dme.bones:
        if bone.name == '' and bone.namehash == namehash:
            print(f'{namehash}: "{name}",')
            bone.name = name
