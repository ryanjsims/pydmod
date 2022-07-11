import logging
import multiprocessing
import multiprocessing.pool
import os
import numpy
from PIL import Image as PILImage
from argparse import ArgumentParser
from io import BytesIO, FileIO
from stl import mesh
from pygltflib import *
from pathlib import *
from glob import glob
from DbgPack import AssetManager

from dme_loader import DME

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d - %(levelname)s] %(message)s",
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

def dme_to_gltf(dme: DME, embed_textures: bool = False) -> Tuple[GLTF2, bytes, Dict[str, PILImage.Image]]:
    gltf = GLTF2()
    blob = b''
    offset = 0
    for i, mesh in enumerate(dme.meshes):
        logging.info(f"Writing mesh {i + 1} of {len(dme.meshes)}")
        vertices_bin = numpy.array(mesh.vertices, dtype=numpy.single).flatten().tobytes()
        indices_bin = numpy.array(mesh.indices, dtype=numpy.ushort if mesh.index_size == 2 else numpy.uintc).tobytes()
        normals_bin = numpy.array([[-n for n in normal] for normal in mesh.normals], dtype=numpy.single).flatten().tobytes()
        tangents_bin = numpy.array([[*tangent[:3], (-tangent[3]) if len(tangent) > 3 else (-1)] for tangent in mesh.tangents], dtype=numpy.single).flatten().tobytes()
        
        attributes = []
        attributes.append([POSITION, len(gltf.accessors)])
        gltf.accessors.append(Accessor(
            bufferView=len(gltf.bufferViews),
            componentType=FLOAT,
            count=len(mesh.vertices),
            type=VEC3,
            max=dme.aabb.corners[7],
            min=dme.aabb.corners[0]
        ))
        gltf.bufferViews.append(BufferView(
            buffer=0,
            byteStride=12,
            byteOffset=offset,
            byteLength=len(vertices_bin),
            target=ARRAY_BUFFER
        ))
        offset += len(vertices_bin)
        blob += vertices_bin
        gltf_mesh_indices = len(gltf.accessors)
        gltf.accessors.append(Accessor(
            bufferView=len(gltf.bufferViews),
            componentType=UNSIGNED_SHORT if mesh.index_size == 2 else UNSIGNED_INT,
            count=len(mesh.indices),
            type=SCALAR,
            min=[0],
            max=[len(mesh.vertices) - 1]
        ))
        gltf.bufferViews.append(BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(indices_bin),
            target=ELEMENT_ARRAY_BUFFER
        ))
        offset += len(indices_bin)
        blob += indices_bin
        if mesh.normals:
            attributes.append([NORMAL, len(gltf.accessors)])
            gltf.accessors.append(Accessor(
                bufferView=len(gltf.bufferViews),
                componentType=FLOAT,
                count=len(mesh.normals),
                type=VEC3
            ))
            gltf.bufferViews.append(BufferView(
                buffer=0,
                byteOffset=offset,
                byteStride=len(mesh.normals[0]) * 4,
                byteLength=len(normals_bin),
                target=ARRAY_BUFFER
            ))
            offset += len(normals_bin)
            blob += normals_bin
        
        
        if mesh.tangents:
            attributes.append([TANGENT, len(gltf.accessors)])
            gltf.accessors.append(Accessor(
                bufferView=len(gltf.bufferViews),
                componentType=FLOAT,
                count=len(mesh.tangents),
                type=VEC4
            ))
            gltf.bufferViews.append(BufferView(
                buffer=0,
                byteOffset=offset,
                byteStride=len(mesh.tangents[0]) * 4,
                byteLength=len(tangents_bin),
                target=ARRAY_BUFFER
            ))
            offset += len(tangents_bin)
            blob += tangents_bin
        
        
        for i, uvs in mesh.uvs.items():
            bin = numpy.array(uvs, dtype=numpy.single).flatten().tobytes()
            attributes.append([f"TEXCOORD_{i}", len(gltf.accessors)])
            gltf.accessors.append(Accessor(
                bufferView=len(gltf.bufferViews),
                componentType=FLOAT,
                count=len(uvs),
                type=VEC2
            ))
            gltf.bufferViews.append(BufferView(
                buffer=0,
                byteOffset=offset,
                byteStride=8,
                byteLength=len(bin),
                target=ARRAY_BUFFER
            ))
            offset += len(bin)
            blob += bin
        
        for i, colors in mesh.colors.items():
            bin = numpy.array(colors, dtype=numpy.single).flatten().tobytes()
            attributes.append([f"COLOR_{i}", len(gltf.accessors)])
            gltf.accessors.append(Accessor(
                bufferView=len(gltf.bufferViews),
                componentType=FLOAT,
                count=len(colors),
                type=VEC4
            ))
            gltf.bufferViews.append(BufferView(
                buffer=0,
                byteOffset=offset,
                byteStride=8,
                byteLength=len(bin),
                target=ARRAY_BUFFER
            ))
            offset += len(bin)
            blob += bin
        
        gltf.nodes.append(Node(
            mesh=len(gltf.meshes)
        ))

        gltf.meshes.append(Mesh(
            primitives=[Primitive(
                attributes=Attributes(**{name: value for name, value in attributes}),
                indices=gltf_mesh_indices
            )]
        ))
        
    gltf.buffers.append(Buffer(
        byteLength=len(blob)
    ))

    textures = {}
    if embed_textures:
        global pool
        manager = get_manager(pool)
        if not manager.loaded.is_set():
            logging.info("Waiting for assets to load...")
        manager.loaded.wait()
        logging.info("Game assets loaded! Dumping textures")
        for name in set(dme.dmat.textures):
            texture = manager.get_raw(name)
            if texture is not None:
                logging.info(f"Loaded {name}")
                im = PILImage.open(BytesIO(texture.get_data()))
                texture.name = str(Path(name).with_suffix(".png"))
                textures[texture.name] = im
                gltf.images.append(Image(uri="textures" + os.sep + texture.name))
            else:
                logging.warning(f"Could not find {name} in loaded game assets, skipping...")

    return gltf, blob, textures

def save_textures(output: str, textures: Dict[str, PILImage.Image]):
    save_directory = Path(output).parent / "textures"
    save_directory.mkdir(exist_ok=True, parents=True)
    for texture_name in textures:
        textures[texture_name].save(save_directory / texture_name, format="png")

def to_glb(dme: DME, output: str, embed_textures: bool = False):
    gltk, blob, textures = dme_to_gltf(dme, embed_textures)
    gltk.set_binary_blob(blob)
    gltk.save_binary(output)
    if embed_textures:
        save_textures(output, textures)


def to_gltf(dme: DME, output: str, embed_textures: bool = False):
    gltk, blob, textures = dme_to_gltf(dme, embed_textures)
    blobpath = Path(output).with_suffix(".bin")
    with open(blobpath, "wb") as blob_out:
        blob_out.write(blob)
    gltk.buffers[0].uri = blobpath.name
    gltk.save_json(output)
    if embed_textures:
        save_textures(output, textures)

manager: AssetManager = None
pool: multiprocessing.pool.Pool = None


def get_manager(pool) -> AssetManager:
    global manager
    if manager is not None:
        return manager
    test_server = Path(r"/mnt/e/Users/Public/Daybreak Game Company/Installed Games/PlanetSide 2 Test/Resources/Assets")
    if not test_server.exists():
        logging.error(f"Test server installation not found at expected location! Please update path in {__file__} to extract textures automatically!")
        raise FileNotFoundError(str(test_server))
    else:
        logging.info("Loading game assets asynchronously...")
        manager = AssetManager([Path(p) for p in glob(str(test_server) + "/assets_x64_*.pack2")], p = pool)
        logging.info(f"Manager created, assets loaded: {manager.loaded.is_set()}")
    return manager

def main():
    global pool
    parser = ArgumentParser(description="DME v4 to GLTF/OBJ/STL converter")
    parser.add_argument("input_file", type=str, help="Name of the input DME model")
    parser.add_argument("--output-file", "-o", type=str, help="Where to store the converted file. If not provided, will use the input filename and change the extension")
    parser.add_argument("--format", "-f", choices=["stl", "gltf", "obj", "glb"], help="The output format to use, required for conversion")
    parser.add_argument("--material-hashes", "-m", type=int, nargs="+", help="The name hash(es) of the materials to use for each mesh when loading the DME data")
    parser.add_argument("--new-materials", "-n", action="store_true", help="Use a more recently generated materials.json file. Not super helpful yet since the hash function seems to be changed")
    parser.add_argument("--dump-textures", "-t", action="store_true", help="Dump the filenames of the textures used by the model to stdout and exit")
    parser.add_argument("--embed-textures", "-e", action="store_true", help="Embed the texture filenames used by the model in the output file, saving the textures alongside the output (GLTF/GLB only)")
    parser.add_argument("--verbose", "-v", help="Increase log level, can be specified multiple times", action="count", default=0)
    args = parser.parse_args()

    logging.basicConfig(level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG), handlers=[handler])
    pool = multiprocessing.Pool(8)
    try:
        if args.embed_textures:
            get_manager(pool)
        with open(args.input_file, "rb") as in_file:
            dme = DME.load(
                in_file,
                material_hashes = args.material_hashes,
                new_mats        = args.new_materials,
                textures_only   = args.dump_textures
            )
        
        if args.dump_textures:
            for texture in dme.dmat.textures:
                print(texture)
            return 0
        
        if not args.format:
            logging.error("File format is required for conversion!")
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
            to_glb(dme, str(tmp_output_path), args.embed_textures)
        elif args.format == "gltf":
            to_gltf(dme, str(tmp_output_path), args.embed_textures)

        os.replace(tmp_output_path, output_path)
    except FileNotFoundError:
        logging.error(f"Could not find {args.input_file}")
    except AssertionError as e:
        logging.error(f"{e}")
    
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()