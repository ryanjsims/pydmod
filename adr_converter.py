import multiprocessing
import xml.etree.ElementTree as ET
import logging

from argparse import ArgumentParser
from DbgPack import AssetManager
from io import BytesIO, FileIO
from pygltflib import GLTF2
from typing import Optional

from dme_loader import DME
from dme_converter import get_manager, to_glb, to_gltf
from gltf_helpers import append_dme_to_gltf

def load_adr(file: FileIO) -> Optional[ET.Element]:
    tree = ET.parse(file)
    root = tree.getroot()
    if root.tag != "ActorRuntime":
        logging.error("File's root XML tag was not ActorRuntime!")
        return None
    return root

def get_base_model(root: ET.Element) -> Optional[str]:
    base = root.find("Base")
    if not base:
        logging.error("No base model present in Actor Runtime file")
        return None
    return base.get("fileName")

def dme_from_adr(manager: AssetManager, adr_file: FileIO) -> Optional[DME]:
    root = load_adr(adr_file)
    if root is None:
        return None
    dme_name = get_base_model(root)
    if dme_name is None:
        return None
    if not manager.loaded.is_set():
        logging.info("Waiting for assets to load...")
    manager.loaded.wait()
    dme_asset = manager.get_raw(dme_name)
    if dme_asset is None:
        logging.error(f"Could not find {dme_name} in loaded assets")
        return None
    return DME.load(BytesIO(dme_asset.get_data()))


def main():
    parser = ArgumentParser(description="Actor Runtime (.adr) to gltf/glb converter")
    parser.add_argument("input_file", type=str, help="Path of the input ADR file")
    parser.add_argument("output_file", type=str, help="Path of the output file")
    parser.add_argument("--format", "-f", choices=["gltf", "glb"], help="The output format to use, required for conversion")
    args = parser.parse_args()
    with multiprocessing.Pool(8) as pool:
        manager = get_manager(pool)
        f = open(args.input_file)
        dme = dme_from_adr(manager, f)
        if args.format == "gltf":
            to_gltf(dme, args.output_file, manager)
        elif args.format == "glb":
            to_glb(dme, args.output_file, manager)


if __name__ == "__main__":
    main()