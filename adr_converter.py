import multiprocessing
from pathlib import Path
import xml.etree.ElementTree as ET
import logging

from argparse import ArgumentParser
from DbgPack import AssetManager
from io import SEEK_END, BytesIO, FileIO, StringIO
from typing import Optional, Tuple

from dme_loader import DME, DMAT
from dme_converter import get_manager, to_glb, to_gltf

logger = logging.getLogger("ADR Converter")

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

def load_adr(file: FileIO) -> Optional[ET.Element]:
    tree = ET.parse(file)
    root = tree.getroot()
    if root.tag != "ActorRuntime":
        logger.error("File's root XML tag was not ActorRuntime!")
        return None
    return root

def get_base_model(root: ET.Element) -> Optional[Tuple[str, str]]:
    base = root.find("Base")
    if base is None:
        logger.error("No base model present in Actor Runtime file")
        return None
    return base.get("fileName"), base.get("paletteName")

def dme_from_adr(manager: AssetManager, adr_file: str) -> Optional[DME]:
    logger.info(f"Loading ADR file {adr_file}...")
    try:
        file = open(adr_file)
    except FileNotFoundError:
        if not manager.loaded.is_set():
            logger.info("Waiting for assets to load...")
        manager.loaded.wait()
        adr_asset = manager.get_raw(adr_file)
        if adr_asset is None:
            logger.error(f"{adr_file} not found in either game assets or filesystem")
            return None
        file = StringIO(adr_asset.get_data().decode("utf-8"))

    root = load_adr(file)
    file.close()
    if root is None:
        return None
    dme_props = get_base_model(root)
    if dme_props is None:
        return None
    dme_name, dma_palette = dme_props
    if not manager.loaded.is_set():
        logger.info("Waiting for assets to load...")
    manager.loaded.wait()
    dme_asset = manager.get_raw(dme_name)
    if dme_asset is None:
        logger.error(f"Could not find {dme_name} in loaded assets")
        return None
    dmat: DMAT = None
    if Path(dma_palette).stem.lower() != Path(dme_name).stem.lower():
        logger.info(f"Loading palette with different name: {Path(dma_palette).stem} vs {Path(dme_name).stem}")
        dma_asset = manager.get_raw(dma_palette)
        if dma_asset:
            dma_file = BytesIO(dma_asset.get_data())
            logger.info(f"Loaded palette asset with length {len(dma_asset.get_data())}, creating DMAT...")
            dmat = DMAT.load(dma_file, len(dma_asset.get_data()))

    dme_file = BytesIO(dme_asset.get_data())
    dme = DME.load(dme_file)
    dme_file.close()
    dme.name = dme_name
    if dmat is not None:
        dme.dmat = dmat
    return dme


def main():
    parser = ArgumentParser(description="Actor Runtime (.adr) to gltf/glb converter")
    parser.add_argument("input_file", type=str, help="Path of the input ADR file")
    parser.add_argument("output_file", type=str, help="Path of the output file")
    parser.add_argument("--format", "-f", choices=["gltf", "glb"], help="The output format to use, required for conversion")
    parser.add_argument("--live", "-l", action="store_true", help="Load assets from live server rather than test")
    parser.add_argument("--verbose", "-v", help="Increase log level, can be specified multiple times", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG), handlers=[handler])

    with multiprocessing.Pool(8) as pool:
        manager = get_manager(pool, args.live)

        dme = dme_from_adr(manager, args.input_file)
        if args.format == "gltf":
            to_gltf(dme, args.output_file, manager, dme.name)
        elif args.format == "glb":
            to_glb(dme, args.output_file, manager, dme.name)


if __name__ == "__main__":
    main()