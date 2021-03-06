import logging
import numpy
import os
import re

from cnk_loader import CNK0
from dme_loader import DME, Mesh as DMEMesh, jenkins
from DbgPack import AssetManager
from io import BytesIO
from PIL import Image as PILImage, ImageChops
from pygltflib import *

logger = logging.getLogger("GLTF Helpers")

TEXTURE_PER_MATERIAL = 4
EMISSIVE_STRENGTH = 1

def append_dme_to_gltf(gltf: GLTF2, dme: DME, manager: AssetManager, mats: Dict[int, Material], textures: Dict[str, PILImage.Image], image_indices: Dict[str, int], offset: int, blob: bytes, dme_name: str) -> Tuple[int, bytes]:
    texture_groups = []
    for texture in dme.dmat.textures:
        match = re.match("(.*)_(C|c|N|n|S|s).dds", texture)
        if match and match.group(1) not in texture_groups:
            texture_groups.append(match.group(1))
    logger.info(f"Texture groups: {texture_groups}")
    
    mat_info = []
    for name in texture_groups:
        for suffix in ["_C.dds", "_N.dds", "_S.dds"]:
            if str(Path(name + suffix).with_suffix(".png")) not in textures:
                load_texture(manager, gltf, textures, name + suffix, image_indices)
        mat_info_entry = {"base": None, "met": None, "norm": None, "emis": None}
        for suffix in ["_C.png", "_MR.png", "_N.png", "_E.png"]:
            if name + suffix not in image_indices:
                continue
            if suffix == "_C.png":
                mat_info_entry["base"] = TextureInfo(index=len(gltf.textures))
            elif suffix == "_MR.png":
                mat_info_entry["met"] = TextureInfo(index=len(gltf.textures))
            elif suffix == "_N.png":
                mat_info_entry["norm"] = NormalMaterialTexture(index=len(gltf.textures))
            elif suffix == "_E.png":
                mat_info_entry["emis"] = TextureInfo(index=len(gltf.textures))
            gltf.textures.append(Texture(
                name=name + suffix,
                source=image_indices[name + suffix]
            ))
        mat_info.append(mat_info_entry)
    
    for i, material in enumerate(dme.dmat.materials):
        materialhash = ((material.namehash << 32) + jenkins.oaat(dme_name.encode("utf-8")))
        logger.warning(f"Material hash: {materialhash}")
        mats[materialhash] = len(gltf.materials)
        if i < len(mat_info):
            mat_textures = mat_info[i]
        else:
            mat_textures = {"base": None, "met": None, "norm": None, "emis": None}

        new_mat = Material(
            name=material.name(),
            pbrMetallicRoughness = PbrMetallicRoughness(
                baseColorTexture=mat_textures["base"],
                metallicRoughnessTexture=mat_textures["met"]
            ),
            normalTexture=mat_textures["norm"],
            emissiveTexture=mat_textures["emis"],
            emissiveFactor=[EMISSIVE_STRENGTH if mat_textures["emis"] is not None else 0] * 3,
            alphaCutoff=None
        )
        gltf.materials.append(new_mat)

    
    for i, mesh in enumerate(dme.meshes):
        logger.info(f"Writing mesh {i + 1} of {len(dme.meshes)}")
        hash = (dme.dmat.materials[i].namehash << 32) + jenkins.oaat(dme_name.encode("utf-8")) if i < len(dme.dmat.materials) else 0
        material_index = mats[hash] if hash in mats else 0
        offset, blob = add_mesh_to_gltf(gltf, dme, mesh, material_index, offset, blob)
        
    
    return offset, blob

def unpack_specular(manager: AssetManager, gltf: GLTF2, textures: Dict[str, PILImage.Image], im: PILImage.Image, name: str, texture_indices: Dict[str, int]):
    metallic = im.getchannel("R")
    roughness = im.getchannel("A")
    metallicRoughness = PILImage.merge("RGB", [metallic, roughness, metallic])
    albedoName = name[:-5] + "C.dds" if name[-5] == "S" else "c.dds"
    albedoAsset = manager.get_raw(albedoName)
    if albedoAsset is not None:
        albedo = PILImage.open(BytesIO(albedoAsset.get_data())).convert(mode="RGB")
        emissive = ImageChops.multiply(im.getchannel("B").convert(mode="RGB"), albedo)
        albedo.close()
    else:
        emissive = im.getchannel("B").convert(mode="RGB")
    ename = name[:-5] + "E.png"
    textures[ename] = emissive
    texture_indices[ename] = len(gltf.images)
    #gltf.textures[min(CNS_seen) * 4 + 3].name = ename
    #gltf.textures[min(CNS_seen) * 4 + 3].source = len(gltf.images)
    gltf.images.append(Image(uri="textures" + os.sep + ename))
    mrname = name[:-5] + "MR.png"
    textures[mrname] = metallicRoughness
    texture_indices[mrname] = len(gltf.images)
    #gltf.textures[min(CNS_seen) * 4 + 1].name = mrname
    #gltf.textures[min(CNS_seen) * 4 + 1].source = len(gltf.images)
    gltf.images.append(Image(uri="textures" + os.sep + mrname))

def unpack_normal(gltf: GLTF2, textures: Dict[str, PILImage.Image], im: PILImage.Image, name: str, texture_indices: Dict[str, int]):
    is_packed = PILImage.eval(im.getchannel("B"), (lambda x: 0 if x > 127 else 255)).getbbox() is not None
    if is_packed:
        #Blue channel is not all >= 0.5, so its not a regular normal map
        x = im.getchannel("A")
        y = im.getchannel("G")
        z = ImageChops.constant(im.getchannel("A"), 255)
    else:
        x = im.getchannel("A")
        y = im.getchannel("G")
        z = im.getchannel("B")
    normal = PILImage.merge("RGB", [x, y, z])
    normal_name = str(Path(name).with_suffix(".png"))
    #gltf.textures[min(CNS_seen) * 4 + 2].name = normal_name
    #gltf.textures[min(CNS_seen) * 4 + 2].source = len(gltf.images)
    textures[normal_name] = normal
    texture_indices[normal_name] = len(gltf.images)
    gltf.images.append(Image(uri="textures" + os.sep + normal_name))

    if is_packed:
        secondary_tint = PILImage.eval(im.getchannel("R"), lambda x: 255 if x < 50 else 0)
        primary_tint = PILImage.eval(im.getchannel("B"), lambda x: 255 if x < 50 else 0)
        camo_tint = PILImage.eval(im.getchannel("B"), lambda x: 255 if x > 150 else 0)
        tints = PILImage.merge("RGB", [primary_tint, secondary_tint, camo_tint])
        tints_name = normal_name[:-5] + "T.png"
        textures[tints_name] = tints
        texture_indices[tints_name] = len(gltf.images)
        gltf.images.append(Image(uri="textures" + os.sep + tints_name))

def add_mesh_to_gltf(gltf: GLTF2, dme: DME, mesh: DMEMesh, material_index: int, offset: int, blob: bytes) -> Tuple[int, bytes]:
    vertices_bin = numpy.array(mesh.vertices[0], dtype=numpy.single).flatten().tobytes()
    indices_bin = numpy.array(mesh.indices, dtype=numpy.ushort if mesh.index_size == 2 else numpy.uintc).tobytes()
    if 0 in mesh.normals:
        normals_bin = numpy.array([[-n for n in normal] for normal in mesh.normals[0]], dtype=numpy.single).flatten().tobytes()
    else:
        normals_bin = b''
    if 0 in mesh.tangents:
        tangents_bin = numpy.array([[*tangent[:3], (-tangent[3]) if len(tangent) > 3 else (-1)] for tangent in mesh.tangents[0]], dtype=numpy.single).flatten().tobytes()
    else:
        tangents_bin = b''
    attributes = []
    attributes.append([POSITION, len(gltf.accessors)])
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews),
        componentType=FLOAT,
        count=len(mesh.vertices[0]),
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
        max=[len(mesh.vertices[0]) - 1]
    ))
    gltf.bufferViews.append(BufferView(
        buffer=0,
        byteOffset=offset,
        byteLength=len(indices_bin),
        target=ELEMENT_ARRAY_BUFFER
    ))
    offset += len(indices_bin)
    blob += indices_bin
    if 0 in mesh.normals:
        attributes.append([NORMAL, len(gltf.accessors)])
        gltf.accessors.append(Accessor(
            bufferView=len(gltf.bufferViews),
            componentType=FLOAT,
            count=len(mesh.normals[0]),
            type=VEC3
        ))
        gltf.bufferViews.append(BufferView(
            buffer=0,
            byteOffset=offset,
            byteStride=len(mesh.normals[0][0]) * 4,
            byteLength=len(normals_bin),
            target=ARRAY_BUFFER
        ))
        offset += len(normals_bin)
        blob += normals_bin
    
    
    if 0 in mesh.tangents:
        attributes.append([TANGENT, len(gltf.accessors)])
        gltf.accessors.append(Accessor(
            bufferView=len(gltf.bufferViews),
            componentType=FLOAT,
            count=len(mesh.tangents[0]),
            type=VEC4
        ))
        gltf.bufferViews.append(BufferView(
            buffer=0,
            byteOffset=offset,
            byteStride=len(mesh.tangents[0][0]) * 4,
            byteLength=len(tangents_bin),
            target=ARRAY_BUFFER
        ))
        offset += len(tangents_bin)
        blob += tangents_bin
    
    
    for j, uvs in mesh.uvs.items():
        bin = numpy.array(uvs, dtype=numpy.single).flatten().tobytes()
        attributes.append([f"TEXCOORD_{j}", len(gltf.accessors)])
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
            indices=gltf_mesh_indices,
            material=material_index
        )]
    ))

    return offset, blob

def add_chunk_to_gltf(gltf: GLTF2, chunk: CNK0, offset: int, blob: bytes) -> Tuple[int, bytes]:
    chunk.calculate_verts()
    triangles = []
    for batch in chunk.triangles:
        for i in batch:
            triangles.append(i)


    vertices_bin = numpy.array(chunk.verts, dtype=numpy.single).flatten().tobytes()
    indices = numpy.array(triangles, dtype=numpy.uintc).flatten()
    indices_bin = indices.tobytes()
    
    attributes = []
    attributes.append([POSITION, len(gltf.accessors)])
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews),
        componentType=FLOAT,
        count=len(chunk.verts),
        type=VEC3,
        max=chunk.aabb[1],
        min=chunk.aabb[0]
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
        componentType=UNSIGNED_INT,
        count=len(indices),
        type=SCALAR,
        min=[0],
        max=[len(chunk.verts) - 1]
    ))
    gltf.bufferViews.append(BufferView(
        buffer=0,
        byteOffset=offset,
        byteLength=len(indices_bin),
        target=ELEMENT_ARRAY_BUFFER
    ))
    offset += len(indices_bin)
    blob += indices_bin
    
    uv_bin = numpy.array(chunk.uvs, dtype=numpy.single).flatten().tobytes()
    attributes.append([f"TEXCOORD_0", len(gltf.accessors)])
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews),
        componentType=FLOAT,
        count=len(chunk.uvs),
        type=VEC2
    ))
    gltf.bufferViews.append(BufferView(
        buffer=0,
        byteOffset=offset,
        byteStride=8,
        byteLength=len(uv_bin),
        target=ARRAY_BUFFER
    ))
    offset += len(uv_bin)
    blob += uv_bin
    
    gltf.nodes.append(Node(
        mesh=len(gltf.meshes)
    ))

    gltf.meshes.append(Mesh(
        primitives=[Primitive(
            attributes=Attributes(**{name: value for name, value in attributes}),
            indices=gltf_mesh_indices
        )]
    ))

    return offset, blob

def load_texture(manager: AssetManager, gltf: GLTF2, textures: Dict[str, PILImage.Image], name: str, texture_indices: Dict[str, int]):
    texture = manager.get_raw(name)
    if texture is None:
        logger.warning(f"Could not find {name} in loaded game assets, skipping...")
        return
    logger.info(f"Loaded {name}")

    im = PILImage.open(BytesIO(texture.get_data()))
    if re.match(".*_(s|S).dds", name):
        unpack_specular(manager, gltf, textures, im, name, texture_indices)
    elif re.match(".*_(n|N).dds", name):
        unpack_normal(gltf, textures, im, name, texture_indices)
    elif re.match(".*_(c|C).dds", name):
        #gltf.textures[min(CNS_seen) * 4].name = name
        texture_indices[str(Path(name).with_suffix(".png"))] = len(gltf.images)
    name = str(Path(name).with_suffix(".png"))
    textures[name] = im
    gltf.images.append(Image(uri="textures" + os.sep + name))