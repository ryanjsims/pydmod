import logging
import multiprocessing
import multiprocessing.pool
import os
import numpy
from PIL import Image as PILImage
from argparse import ArgumentParser
from io import FileIO
from stl import mesh
from pygltflib import *
from pathlib import *
from glob import glob
from DbgPack import AssetManager

from dme_loader import DME
from gltf_helpers import *

logger = logging.getLogger("DME Converter")

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

def to_stl(dme: DME, output: str):
    faces = []
    for i in range(0, len(dme.meshes[0].indices), 3):
        faces.append((dme.meshes[0].indices[i+2], dme.meshes[0].indices[i+1], dme.meshes[0].indices[i]))
    faces = numpy.array(faces)
    vertex_array = numpy.array(dme.meshes[0].vertices)
    stl_mesh = mesh.Mesh(numpy.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = vertex_array[f[j], :]

    stl_mesh.save(output)

def to_obj(dme: DME, output: FileIO):
    for mesh in dme.meshes:
        for vertex in mesh.vertices:
            output.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        for normal in mesh.normals:
            output.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
        
        for uv in mesh.uvs:
            output.write(f"vt {uv[0]} {uv[1]}\n")
        
        for i in range(0, len(mesh.indices), 3):
            i0, i1, i2 = mesh.indices[i] + 1, mesh.indices[i+1] + 1, mesh.indices[i+2] + 1
            output.write(f"f {i1}/{i1} {i0}/{i0} {i2}/{i2}\n")

def dme_to_gltf(dme: DME, manager: AssetManager, dme_name: str, output_name: str) -> Tuple[GLTF2, bytes, Dict[str, PILImage.Image]]:
    gltf = GLTF2()
    blob = b''
    offset = 0
    mats = {}
    textures = {}
    image_indices: Dict[str, int] = {}
    if not manager.loaded.is_set():
        manager.loaded.wait()
    offset, blob = append_dme_to_gltf(gltf, dme, manager, mats, textures, image_indices, offset, blob, dme_name)

    gltf.buffers.append(Buffer(
        byteLength=len(blob)
    ))
    gltf.scene = 0
    if len(gltf.nodes) > 1:
        gltf.scenes.append(Scene(nodes=[len(gltf.nodes)]))
        gltf.nodes.append(Node(
            name=output_name,
            children=[i for i in range(len(gltf.nodes))]
        ))
    elif len(gltf.nodes) == 1:
        gltf.scenes.append(Scene(nodes=[0]))
        gltf.nodes[0].name = output_name

    return gltf, blob, textures

def save_textures(output: str, textures: Dict[str, PILImage.Image]):
    save_directory = Path(output).parent / "textures"
    save_directory.mkdir(exist_ok=True, parents=True)
    for texture_name in textures:
        if textures[texture_name] is None:
            continue
        textures[texture_name].save(save_directory / texture_name, format="png")
        textures[texture_name].close()
        textures[texture_name] = None

def to_glb(dme: DME, output: str, manager: AssetManager, dme_name: str):
    gltk, blob, textures = dme_to_gltf(dme, manager, dme_name, str(Path(output).stem))
    gltk.set_binary_blob(blob)
    gltk.save_binary(output)
    save_textures(output, textures)


def to_gltf(dme: DME, output: str,  manager: AssetManager, dme_name: str):
    gltk, blob, textures = dme_to_gltf(dme, manager, dme_name, str(Path(output).stem))
    blobpath = Path(output).with_suffix(".bin")
    with open(blobpath, "wb") as blob_out:
        blob_out.write(blob)
    gltk.buffers[0].uri = blobpath.name
    gltk.save_json(output)
    save_textures(output, textures)

manager: AssetManager = None
pool: multiprocessing.pool.Pool = None


def get_manager(pool: multiprocessing.pool.Pool, live: bool = False) -> AssetManager:
    global manager
    if manager is not None:
        return manager
    test_server = Path(r"/mnt/e/Users/Public/Daybreak Game Company/Installed Games/PlanetSide 2 Test/Resources/Assets")
    live_server = Path(r"/mnt/c/Users/Public/Daybreak Game Company/Installed Games/PlanetSide 2/Resources/Assets")
    server = test_server if not live else live_server
    if not server.exists():
        logger.error(f"Installation not found at expected location! Please update path in {__file__} to extract textures automatically!")
        raise FileNotFoundError(str(server))
    else:
        logger.info("Loading game assets asynchronously...")
        manager = AssetManager([Path(p) for p in glob(str(server) + "/assets_x64_*.pack2")], p = pool)
        logger.info(f"Manager created, assets loaded: {manager.loaded.is_set()}")
    return manager

def main():
    global pool
    parser = ArgumentParser(description="DME v4 to GLTF/OBJ/STL converter")
    parser.add_argument("input_file", type=str, help="Name of the input DME model")
    parser.add_argument("--output-file", "-o", type=str, help="Where to store the converted file. If not provided, will use the input filename and change the extension")
    parser.add_argument("--format", "-f", choices=["stl", "gltf", "obj", "glb"], help="The output format to use, required for conversion")
    parser.add_argument("--material-hashes", "-m", type=int, nargs="+", help="The name hash(es) of the materials to use for each mesh when loading the DME data")
    parser.add_argument("--dump-textures", "-t", action="store_true", help="Dump the filenames of the textures used by the model to stdout and exit")
    parser.add_argument("--verbose", "-v", help="Increase log level, can be specified multiple times", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG), handlers=[handler])
    pool = multiprocessing.Pool(8)
    try:
        if not args.dump_textures:
            manager = get_manager(pool)
        with open(args.input_file, "rb") as in_file:
            dme = DME.load(
                in_file,
                material_hashes = args.material_hashes,
                textures_only   = args.dump_textures
            )
        
        if args.dump_textures:
            for texture in dme.dmat.textures:
                print(texture)
            return 0
        
        if not args.format:
            logger.error("File format is required for conversion!")
            return 1
        if not args.output_file:
            output_path = Path(args.input_file).with_suffix("." + args.format)
        else:
            output_path = Path(args.output_file)
        tmp_output_path = output_path.with_suffix(".tmp")
        
        if args.format == "obj":
            with open(tmp_output_path, "w") as output:
                to_obj(dme, output)
        elif args.format == "stl":
            to_stl(dme, str(tmp_output_path))
        elif args.format == "glb":
            to_glb(dme, str(tmp_output_path), manager, Path(args.input_file).name)
        elif args.format == "gltf":
            to_gltf(dme, str(tmp_output_path), manager, Path(args.input_file).name)

        os.replace(tmp_output_path, output_path)
    except FileNotFoundError:
        logger.error(f"Could not find {args.input_file}")
    except AssertionError as e:
        logger.error(f"{e}")
    
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()