from argparse import ArgumentParser
from dataclasses import astuple
import math
from DbgPack import AssetManager
from io import BytesIO
from glob import glob
from pathlib import Path
from PIL import Image as PILImage
from pygltflib import *
from scipy.spatial.transform import Rotation

import logging
import multiprocessing
import multiprocessing.pool

from adr_converter import dme_from_adr
from cnk_loader import ForgelightChunk
from dme_converter import append_dme_to_gltf, save_textures
from gltf_helpers import add_chunk_to_gltf
from zone_loader import Zone

logger = logging.getLogger("Zone Converter")

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

manager = None

def get_manager(pool: multiprocessing.pool.Pool) -> AssetManager:
    global manager
    if manager is not None:
        return manager
    test_server = Path(r"/mnt/e/Users/Public/Daybreak Game Company/Installed Games/PlanetSide 2 Test/Resources/Assets")
    if not test_server.exists():
        logger.error(f"Test server installation not found at expected location! Please update path in {__file__} to extract textures automatically!")
        raise FileNotFoundError(str(test_server))
    else:
        logger.info("Loading game assets asynchronously...")
        manager = AssetManager(
            [Path(p) for p in glob(str(test_server) + "/assets_x64_*.pack2")]
            # + [Path(p) for p in glob(str(test_server) + "/Amerish_x64_*.pack2")]
            + [Path(p) for p in glob(str(test_server) + "/Cleanroom_x64_*.pack2")]
            # + [Path(p) for p in glob(str(test_server) + "/Esamir_x64_*.pack2")]
            # + [Path(p) for p in glob(str(test_server) + "/Hossin_x64_*.pack2")]
            # + [Path(p) for p in glob(str(test_server) + "/Indar_x64_*.pack2")]
            + [Path(p) for p in glob(str(test_server) + "/Nexus_x64_*.pack2")]
            + [Path(p) for p in glob(str(test_server) + "/Oshur_x64_*.pack2")]
            + [Path(p) for p in glob(str(test_server) + "/quickload_x64_*.pack2")]
            # + [Path(p) for p in glob(str(test_server) + "/Sanctuary_x64_*.pack2")]
            + [Path(p) for p in glob(str(test_server) + "/VR_x64_*.pack2")],
            p = pool
        )
        logger.info(f"Manager created, assets loaded: {manager.loaded.is_set()}")
    return manager


def main():
    parser = ArgumentParser(description="A utility to convert Zone files to GLTF2 files")
    parser.add_argument("input_file", type=str, help="Path of the input Zone file")
    parser.add_argument("output_file", type=str, help="Path of the output file")
    parser.add_argument("--format", "-f", choices=["gltf", "glb"], help="The output format to use, required for conversion")
    parser.add_argument("--verbose", "-v", help="Increase log level, can be specified multiple times", action="count", default=0)
    parser.add_argument("--skip-textures", "-s", help="Skips saving textures", action="store_true")
    parser.add_argument("--terrain-enabled", "-t", help="Load terrain chunks as models into the result", action="store_true")
    parser.add_argument("--actors-enabled", "-a", help="Loads static actor files as models (buildings, trees, etc)", action="store_true")
    args = parser.parse_args()

    if not (args.terrain_enabled or args.actors_enabled):
        parser.error("No model loading enabled! Use either/both -a and -t to load models")

    logging.basicConfig(level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG), handlers=[handler])

    with multiprocessing.Pool(8) as pool:
        manager = get_manager(pool)
        try:
            with open(args.input_file, "rb") as file:
                zone = Zone.load(file)
            if not manager.loaded.is_set():
                logger.info("Waiting for assets to load...")
            manager.loaded.wait()
        except FileNotFoundError:
            logger.warning(f"File not found: {args.input_file}. Loading from game assets...")
            if not manager.loaded.is_set():
                logger.info("Waiting for assets to load...")
            manager.loaded.wait()
            zone_asset = manager.get_raw(args.input_file)
            if zone_asset is None:
                logger.error(f"{args.input_file} not found in game assets!")
                return -1
            zone = Zone.load(BytesIO(zone_asset.get_data()))
    
    gltf = GLTF2()
    mats: Dict[int, Material] = {}
    textures: Dict[str, PILImage.Image] = {}
    blob: bytes = b''
    offset = 0
    instance_nodes = []

    if args.terrain_enabled:
        for x in range(zone.header.start_x, zone.header.start_x + zone.header.chunks_x, 4):
            for y in range(zone.header.start_y, zone.header.start_y + zone.header.chunks_y, 4):
                chunk_name = f"{Path(args.input_file).stem}_{x}_{y}.cnk0"
                logger.info(f" - {chunk_name} - ")
                chunk = ForgelightChunk.load(BytesIO(manager.get_raw(chunk_name).get_data()))
                node_start = len(gltf.nodes)
                offset, blob = add_chunk_to_gltf(gltf, chunk, offset, blob)
                gltf.nodes.append(Node(
                    name=Path(chunk_name).stem, 
                    children=list(range(node_start, len(gltf.nodes))),
                    translation=[64.0 * (y + 4), 0, 64.0 * (x + 4)]
                ))

    if args.actors_enabled:
        for object in zone.objects:
            dme = dme_from_adr(manager, object.actor_file)
            if dme is None:
                logger.warning(f"Skipping {object.actor_file}...")
                continue

            node_start = len(gltf.nodes)
            offset, blob = append_dme_to_gltf(gltf, dme, manager, mats, textures, offset, blob, object.actor_file)
            node_end = len(gltf.nodes)

            logger.info(f"Adding {len(object.instances)} instances of {object.actor_file}")
            instances = []
            for i, instance in enumerate(object.instances):
                #print(repr(i))
                if i > 0:
                    children = []
                    for i in range(node_start, node_end):
                        children.append(len(gltf.nodes))
                        gltf.nodes.append(Node(mesh=gltf.nodes[i].mesh))
                else:
                    children = list(range(node_start, node_end))
                #constrain_rad = lambda x: x - math.ceil(x / math.pi - 0.5) * math.pi
                #rotx, roty, rotz = (instance.rotation.y, instance.rotation.x, instance.rotation.z)
                r = Rotation.from_euler("yzx", astuple(instance.rotation)[:3], False)
                rot = list(r.as_quat())
                #print(rot)
                    
                rot[1] *= -1
                rot[3] *= -1
                temp = rot[0] * -1
                rot[0] = rot[2] * -1
                rot[2] = temp
                #print(rot)
                instances.append(len(gltf.nodes))
                gltf.nodes.append(Node(
                    name=Path(object.actor_file).stem,
                    children=children,
                    rotation=rot,
                    translation=astuple(instance.translation)[:3],
                    scale=astuple(instance.scale)[:3]
                ))

            instance_nodes.extend(instances)
    
    gltf.buffers.append(Buffer(
        byteLength=offset
    ))

    gltf.scene = 0
    gltf.scenes.append(Scene(nodes=instance_nodes))

    logger.info("Saving GLTF file...")
    if args.format == "glb":
        gltf.set_binary_blob(blob)
        gltf.save_binary(args.output_file)
    elif args.format == "gltf":
        blobpath = Path(args.output_file).with_suffix(".bin")
        with open(blobpath, "wb") as f:
            f.write(blob)
        gltf.buffers[0].uri = blobpath.name
        gltf.save_json(args.output_file)
    
    if not args.skip_textures:
        logger.info("Saving Textures...")
        save_textures(args.output_file, textures)
        logger.info(f"Saved {len(textures)} textures")

if __name__ == "__main__":
    main()