from argparse import ArgumentParser
from dataclasses import astuple
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
from cnk_loader import ForgelightChunk, CNK1
from dme_converter import append_dme_to_gltf, save_textures
from gltf_helpers import add_chunk_to_gltf
from zone_loader import Zone
from zone_loader.data_classes import LightType

logger = logging.getLogger("Zone Converter")

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

manager = None

def get_manager(pool: multiprocessing.pool.Pool, live: bool = False) -> AssetManager:
    global manager
    if manager is not None:
        return manager
    test_server = Path(r"/mnt/e/Users/Public/Daybreak Game Company/Installed Games/PlanetSide 2 Test/Resources/Assets")
    live_server = Path(r"/mnt/c/Users/Public/Daybreak Game Company/Installed Games/PlanetSide 2/Resources/Assets")
    server = test_server if not live else live_server
    if not server.exists():
        logger.error(f"Test server installation not found at expected location! Please update path in {__file__} to extract textures automatically!")
        raise FileNotFoundError(str(server))
    else:
        logger.info("Loading game assets asynchronously...")
        manager = AssetManager(
            [Path(p) for p in glob(str(server) + "/assets_x64_*.pack2")]
            # + [Path(p) for p in glob(str(server) + "/Amerish_x64_*.pack2")]
            + [Path(p) for p in glob(str(server) + "/Cleanroom_x64_*.pack2")]
            # + [Path(p) for p in glob(str(server) + "/Esamir_x64_*.pack2")]
            # + [Path(p) for p in glob(str(server) + "/Hossin_x64_*.pack2")]
            # + [Path(p) for p in glob(str(server) + "/Indar_x64_*.pack2")]
            + [Path(p) for p in glob(str(server) + "/nexus_x64_*.pack2")]
            + [Path(p) for p in glob(str(server) + "/Oshur_x64_*.pack2")]
            + [Path(p) for p in glob(str(server) + "/quickload_x64_*.pack2")]
            # + [Path(p) for p in glob(str(server) + "/Sanctuary_x64_*.pack2")]
            + [Path(p) for p in glob(str(server) + "/VR_x64_*.pack2")],
            p = pool
        )
        logger.info(f"Manager created, assets loaded: {manager.loaded.is_set()}")
    return manager


def save_chunk_textures(gltf: GLTF2, chunk_lod: CNK1, input_file: str, output_file: str, x: int, y: int, sampler: int, skip_textures: bool = False) -> int:
    color = chunk_lod.color()
    specular = chunk_lod.specular()
    normal = chunk_lod.normal()

    base_dir = Path(output_file).parent / "textures" / "chunks"
    base_dir.mkdir(exist_ok=True, parents=True)
    rel_dir = Path("textures") / "chunks"
    color_name = str(rel_dir / f"{Path(input_file).stem}_{x}_{y}_C.png")
    color_info = TextureInfo(index=len(gltf.textures))
    gltf.textures.append(Texture(
        name=color_name,
        source=len(gltf.images),
        sampler=sampler
    ))
    gltf.images.append(Image(uri=color_name))

    spec_name = str(rel_dir / f"{Path(input_file).stem}_{x}_{y}_S.png")
    spec_info = TextureInfo(index=len(gltf.textures))
    gltf.textures.append(Texture(
        name=spec_name,
        source=len(gltf.images),
        sampler=sampler
    ))
    gltf.images.append(Image(uri=spec_name))

    norm_name = str(rel_dir / f"{Path(input_file).stem}_{x}_{y}_N.png")
    norm_info = TextureInfo(index=len(gltf.textures))
    gltf.textures.append(Texture(
        name=norm_name,
        source=len(gltf.images),
        sampler=sampler
    ))
    gltf.images.append(Image(uri=norm_name))

    index = len(gltf.materials)

    gltf.materials.append(Material(
        extensions={
            "KHR_materials_specular": {
                "specularTexture": spec_info
            }
        },
        name=f"{Path(input_file).stem}_{x}_{y}",
        pbrMetallicRoughness=PbrMetallicRoughness(
            baseColorTexture=color_info,
            metallicRoughnessTexture=spec_info
        ),
        normalTexture=norm_info,
        alphaCutoff=None
    ))

    if not skip_textures:
        color.save(Path(output_file).parent / color_name)
        specular.save(Path(output_file).parent / spec_name)
        normal.save(Path(output_file).parent / norm_name)

    return index


def get_gltf_rotation(rotation: Tuple[float, float, float]):
    r = Rotation.from_euler("yzx", rotation, False)
    rot = list(r.as_quat())        
    rot[1] *= -1
    rot[3] *= -1
    temp = rot[0] * -1
    rot[0] = rot[2] * -1
    rot[2] = temp
    return rot

def main():
    parser = ArgumentParser(description="A utility to convert Zone files to GLTF2 files")
    parser.add_argument("input_file", type=str, help="Path of the input Zone file, either already extracted or from game assets")
    parser.add_argument("output_file", type=str, help="Path of the output file")
    parser.add_argument("--format", "-f", choices=["gltf", "glb"], help="The output format to use, required for conversion")
    parser.add_argument("--verbose", "-v", help="Increase log level, can be specified multiple times", action="count", default=0)
    parser.add_argument("--skip-textures", "-s", help="Skips saving textures", action="store_true")
    parser.add_argument("--terrain-enabled", "-t", help="Load terrain chunks as models into the result", action="store_true")
    parser.add_argument("--actors-enabled", "-a", help="Loads static actor files as models (buildings, trees, etc)", action="store_true")
    parser.add_argument("--lights-enabled", "-i", help="Adds lights to the output file", action="store_true")
    parser.add_argument("--live", "-l", help="Loads live assets rather than test", action="store_true")
    args = parser.parse_args()

    if not (args.terrain_enabled or args.actors_enabled or args.lights_enabled):
        parser.error("No model/light loading enabled! Use either/all of -a, -t, or -i to load models and/or lights")

    logging.basicConfig(level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG), handlers=[handler])

    with multiprocessing.Pool(8) as pool:
        manager = get_manager(pool, args.live)
        try:
            file = open(args.input_file, "rb")
        except FileNotFoundError:
            logger.warning(f"File not found: {args.input_file}. Loading from game assets...")
            if not manager.loaded.is_set():
                logger.info("Waiting for assets to load...")
            manager.loaded.wait()
            zone_asset = manager.get_raw(args.input_file)
            if zone_asset is None:
                logger.error(f"{args.input_file} not found in game assets!")
                return -1
            file = BytesIO(zone_asset.get_data())
    if not manager.loaded.is_set():
        logger.info("Waiting for assets to load...")
    manager.loaded.wait()
    
    zone = Zone.load(file)

    gltf = GLTF2()
    mats: Dict[int, Material] = {}
    textures: Dict[str, PILImage.Image] = {}
    blob: bytes = b''
    offset = 0
    instance_nodes = []
    chunk_nodes = []
    terrain_parent = None
    actor_parent = None
    light_parent = None

    if args.terrain_enabled:
        # Create a sampler for the terrain to use that clamps the textures to the edge, avoiding obvious lines between chunks
        gltf.extensionsUsed.append("KHR_materials_specular")
        sampler = len(gltf.samplers)
        gltf.samplers.append(Sampler(
            minFilter=LINEAR,
            magFilter=LINEAR,
            wrapS=CLAMP_TO_EDGE,
            wrapT=CLAMP_TO_EDGE
        ))
        for x in range(zone.header.start_x, zone.header.start_x + zone.header.chunks_x, 4):
            for y in range(zone.header.start_y, zone.header.start_y + zone.header.chunks_y, 4):
                chunk_name = f"{Path(args.input_file).stem}_{x}_{y}.cnk0"
                chunk_texture_name = f"{Path(args.input_file).stem}_{x}_{y}.cnk1"
                logger.info(f" - {chunk_name} - ")
                chunk = ForgelightChunk.load(BytesIO(manager.get_raw(chunk_name).get_data()))
                chunk_lod = ForgelightChunk.load(BytesIO(manager.get_raw(chunk_texture_name).get_data()))
                
                assert type(chunk_lod) == CNK1
                material = save_chunk_textures(gltf, chunk_lod, args.input_file, args.output_file, x, y, sampler, args.skip_textures)

                node_start = len(gltf.nodes)
                offset, blob = add_chunk_to_gltf(gltf, chunk, material, offset, blob)
                chunk_nodes.append(len(gltf.nodes))
                gltf.nodes.append(Node(
                    name=Path(chunk_name).stem, 
                    children=list(range(node_start, len(gltf.nodes))),
                    translation=[64.0 * (y + 4), 0, 64.0 * (x + 4)]
                ))
        terrain_parent = len(gltf.nodes)
        gltf.nodes.append(Node(name="Terrain", children=chunk_nodes))

    if args.actors_enabled:
        image_indices: Dict[str, int] = {}
        for object in zone.objects:
            dme = dme_from_adr(manager, object.actor_file)
            if dme is None:
                logger.warning(f"Skipping {object.actor_file}...")
                continue

            node_start = len(gltf.nodes)
            offset, blob = append_dme_to_gltf(gltf, dme, manager, mats, textures, image_indices, offset, blob, object.actor_file)
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
                rot = get_gltf_rotation(astuple(instance.rotation)[:3])
                instances.append(len(gltf.nodes))
                gltf.nodes.append(Node(
                    name=Path(object.actor_file).stem,
                    children=children,
                    rotation=rot,
                    translation=astuple(instance.translation)[:3],
                    scale=astuple(instance.scale)[:3]
                ))

            instance_nodes.extend(instances)
        actor_parent = len(gltf.nodes)
        gltf.nodes.append(Node(name="Object Instances", children=instance_nodes))
    
    if args.lights_enabled:
        gltf.extensionsUsed.append("KHR_lights_punctual")
        gltf.extensions["KHR_lights_punctual"] = {
            "lights": []
        }
        light_nodes = []
        light_def_to_index = {}
        logger.info(f"Adding {len(zone.lights)} lights to the scene...")
        for light in zone.lights:
            light_nodes.append(len(gltf.nodes))
            light_def = {
                "color": [light.color_val.r / 255.0, light.color_val.g / 255.0, light.color_val.b / 255.0],
                "type": "point" if light.type == LightType.Point else "spot",
                "intensity": light.unk_floats[0] * 100
            }
            if light.type == LightType.Spot:
                light_def["spot"] = {}
            key = str(light_def["color"]) + light_def["type"] + str(light_def["intensity"])
            if key not in light_def_to_index:
                light_def_to_index[key] = len(gltf.extensions["KHR_lights_punctual"]["lights"])
                gltf.extensions["KHR_lights_punctual"]["lights"].append(light_def)

            gltf.nodes.append(Node(
                name=light.name, 
                translation=astuple(light.translation)[:3],
                rotation=get_gltf_rotation(astuple(light.rotation)[:3]),
                scale=[1, 1, -1],
                extensions={
                    "KHR_lights_punctual": {
                        "light": light_def_to_index[key]
                    }
                }
            ))
        logger.info(f"Added {len(light_nodes)} instances of {len(light_def_to_index)} unique lights")
        light_parent = len(gltf.nodes)
        gltf.nodes.append(Node(name="Lights", children=light_nodes))
        
    
    gltf.buffers.append(Buffer(
        byteLength=offset
    ))

    scene_nodes = []
    if terrain_parent:
        scene_nodes.append(terrain_parent)
    if actor_parent:
        scene_nodes.append(actor_parent)
    if light_parent:
        scene_nodes.append(light_parent)
    gltf.scene = 0
    gltf.scenes.append(Scene(nodes=scene_nodes))

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