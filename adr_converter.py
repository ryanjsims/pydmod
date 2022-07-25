import multiprocessing
import xml.etree.ElementTree as ET
import logging

from argparse import ArgumentParser
from DbgPack import AssetManager
from io import BytesIO, FileIO
from typing import Optional

from dme_loader import DME
from dme_converter import get_manager, to_glb, to_gltf

logger = logging.getLogger("ADR Converter")

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

def load_adr(filename: str) -> Optional[ET.Element]:
    logger.info(f"Loading ADR file {filename}...")
    with open(filename) as file:
        tree = ET.parse(file)
    root = tree.getroot()
    if root.tag != "ActorRuntime":
        logger.error("File's root XML tag was not ActorRuntime!")
        return None
    return root

def get_base_model(root: ET.Element) -> Optional[str]:
    base = root.find("Base")
    if base is None:
        logger.error("No base model present in Actor Runtime file")
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
        logger.info("Waiting for assets to load...")
    manager.loaded.wait()
    dme_asset = manager.get_raw(dme_name)
    if dme_asset is None:
        logger.error(f"Could not find {dme_name} in loaded assets")
        return None
    return DME.load(BytesIO(dme_asset.get_data()))


def main():
    parser = ArgumentParser(description="Actor Runtime (.adr) to gltf/glb converter")
    parser.add_argument("input_file", type=str, help="Path of the input ADR file")
    parser.add_argument("output_file", type=str, help="Path of the output file")
    parser.add_argument("--format", "-f", choices=["gltf", "glb"], help="The output format to use, required for conversion")
    parser.add_argument("--verbose", "-v", help="Increase log level, can be specified multiple times", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG), handlers=[handler])

    with multiprocessing.Pool(8) as pool:
        manager = get_manager(pool)
        dme = dme_from_adr(manager, args.input_file)
        if args.format == "gltf":
            to_gltf(dme, args.output_file, manager)
        elif args.format == "glb":
            to_glb(dme, args.output_file, manager)


if __name__ == "__main__":
    main()