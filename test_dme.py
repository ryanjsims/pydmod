import os
import struct
from argparse import ArgumentParser
from dme_loader import DME, D3DXParamClass, D3DXParamType, jenkins

parser = ArgumentParser(description="DME Tester for cmd line inspection of DME files")
parser.add_argument("file", help="The file to load")
args = parser.parse_args()

with open(args.file, "rb") as f:
    dme = DME.load(f)

