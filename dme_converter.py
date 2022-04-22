import logging
import os
import sys
import numpy
from argparse import ArgumentParser
from io import FileIO
from stl import mesh
from pygltflib import *

from dme_loader import DME

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logging.basicConfig(level=logging.DEBUG, handlers=[handler])

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

def dme_to_gltk(dme: DME):
    gltf = GLTF2()
    blob = b''
    offset = 0
    for mesh in dme.meshes:
        vertices_bin = numpy.array(mesh.vertices, dtype=numpy.single).flatten().tobytes()
        indices_bin = numpy.array(mesh.indices, dtype=numpy.ushort if mesh.index_size == 2 else numpy.uintc).tobytes()
        normals_bin = numpy.array([[-n for n in normal] for normal in mesh.normals], dtype=numpy.single).flatten().tobytes()
        tangents_bin = numpy.array([[*tangent[:3], -tangent[3]] for tangent in mesh.tangents], dtype=numpy.single).flatten().tobytes()
        
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
        
        """
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
        """
        
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

    return gltf, blob

def to_glb(dme: DME, output: str):
    gltk, blob = dme_to_gltk(dme)
    gltk.set_binary_blob(blob)
    gltk.save_binary(output)

def to_gltf(dme: DME, output: str):
    gltk, blob = dme_to_gltk(dme)
    blobpath = Path(output).with_suffix(".bin")
    with open(blobpath, "wb") as blob_out:
        blob_out.write(blob)
    gltk.buffers[0].uri = blobpath.name
    gltk.save_json(output)

def main():
    parser = ArgumentParser(description="DME v4 to GLTF/OBJ/STL converter")
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("format", choices=["stl", "gltf", "obj", "glb"])
    args = parser.parse_args()
    try:
        output_path = Path(args.output_file)
        tmp_output_path = output_path.with_suffix(".tmp")
        with open(args.input_file, "rb") as in_file:
            dme = DME.load(in_file)
        
        if args.format == "obj":
            with open(tmp_output_path, "w") as output:
                to_obj(dme, output)
        elif args.format == "stl":
            to_stl(dme, str(tmp_output_path))
        elif args.format == "glb":
            to_glb(dme, str(tmp_output_path))
        elif args.format == "gltf":
            to_gltf(dme, str(tmp_output_path))

        os.replace(tmp_output_path, output_path)
    except FileNotFoundError:
        print(f"Error: Could not find {args.input_file}", file=sys.stderr)
    except AssertionError as e:
        print(f"Error: {e}", file=sys.stderr)
    


if __name__ == "__main__":
    main()