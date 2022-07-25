import logging
import numpy
import os
import re

from dme_loader import DME, Mesh as DMEMesh
from DbgPack import AssetManager
from io import BytesIO
from PIL import Image as PILImage, ImageChops
from pygltflib import *

TEXTURE_PER_MATERIAL = 4
EMISSIVE_STRENGTH = 50

def append_dme_to_gltf(gltf: GLTF2, dme: DME, manager: AssetManager, mats: Dict[int, Material], textures: Dict[str, PILImage.Image], offset: int, blob: bytes) -> Tuple[int, bytes]:
    for material in dme.dmat.materials:
        if material.namehash in mats:
            continue
        mats[material.namehash] = len(gltf.materials)

        new_mat = Material(
            name=material.name(),
            pbrMetallicRoughness = PbrMetallicRoughness(
                baseColorTexture=TextureInfo(
                    index=len(gltf.materials) * TEXTURE_PER_MATERIAL,
                    texCoord=0
                ),
                metallicRoughnessTexture=TextureInfo(
                    index=len(gltf.materials) * TEXTURE_PER_MATERIAL + 1,
                    texCoord=0
                )
            ),
            normalTexture=NormalMaterialTexture(
                index=len(gltf.materials) * TEXTURE_PER_MATERIAL + 2,
                texCoord=0
            ),
            emissiveTexture=TextureInfo(
                index=len(gltf.materials) * TEXTURE_PER_MATERIAL + 3,
                texCoord=0
            ),
            emissiveFactor=[EMISSIVE_STRENGTH] * 3
        )
        gltf.materials.append(new_mat)
        for i in range(TEXTURE_PER_MATERIAL):
            gltf.textures.append(Texture())

    
    for i, mesh in enumerate(dme.meshes):
        logging.info(f"Writing mesh {i + 1} of {len(dme.meshes)}")
        offset, blob = add_mesh_to_gltf(gltf, dme, mesh, mats, offset, blob)
    

    if len(dme.dmat.textures) > 0:
        if not manager.loaded.is_set():
            logging.info("Waiting for assets to load...")
        manager.loaded.wait()
        logging.info("Game assets loaded! Dumping textures")
        CNS_seen = [len(gltf.textures) // 4 - 1, len(gltf.textures) // 4 - 1, len(gltf.textures) // 4 - 1]
        for name in dme.dmat.textures:
            if str(Path(name).with_suffix(".png")) in textures:
                continue
            load_texture(manager, gltf, textures, name, CNS_seen)
    
    return offset, blob

def unpack_specular(manager: AssetManager, gltf: GLTF2, textures: Dict[str, PILImage.Image], im: PILImage.Image, name: str, CNS_seen: List[int]):
    metallic = im.getchannel("R")
    roughness = im.getchannel("A")
    metallicRoughness = PILImage.merge("RGB", [metallic, roughness, metallic])
    albedoName = name[:-5] + "C.dds" if name[-5] == "S" else "c.dds"
    albedo = PILImage.open(BytesIO(manager.get_raw(albedoName).get_data())).convert(mode="RGB")
    emissive = ImageChops.multiply(im.getchannel("B").convert(mode="RGB"), albedo)
    mrname = name[:-5] + "MR.png"
    ename = name[:-5] + "E.png"
    textures[mrname] = metallicRoughness
    textures[ename] = emissive
    gltf.textures[min(CNS_seen) * 4 + 1].name = mrname
    gltf.textures[min(CNS_seen) * 4 + 1].source = len(gltf.images)
    gltf.images.append(Image(uri="textures" + os.sep + mrname))
    gltf.textures[min(CNS_seen) * 4 + 3].name = ename
    gltf.textures[min(CNS_seen) * 4 + 3].source = len(gltf.images)
    gltf.images.append(Image(uri="textures" + os.sep + ename))
    albedo.close()

def unpack_normal(gltf: GLTF2, textures: Dict[str, PILImage.Image], im: PILImage.Image, name: str, CNS_seen: List[int]):
    x = im.getchannel("A")
    y = im.getchannel("G")
    z = ImageChops.constant(im.getchannel("A"), 255)
    normal_name = str(Path(name).with_suffix(".png"))
    gltf.textures[min(CNS_seen) * 4 + 2].name = normal_name
    gltf.textures[min(CNS_seen) * 4 + 2].source = len(gltf.images)
    gltf.images.append(Image(uri="textures" + os.sep + normal_name))
    normal = PILImage.merge("RGB", [x, y, z])
    textures[normal_name] = normal

    secondary_tint = PILImage.eval(im.getchannel("R"), lambda x: 255 if x < 50 else 0)
    primary_tint = PILImage.eval(im.getchannel("B"), lambda x: 255 if x < 50 else 0)
    camo_tint = PILImage.eval(im.getchannel("B"), lambda x: 255 if x > 150 else 0)
    tints = PILImage.merge("RGB", [primary_tint, secondary_tint, camo_tint])
    tints_name = normal_name[:-5] + "T.png"
    gltf.images.append(Image(uri="textures" + os.sep + tints_name))
    textures[tints_name] = tints

def add_mesh_to_gltf(gltf: GLTF2, dme: DME, mesh: DMEMesh, mats: Dict[int, Material], offset: int, blob: bytes) -> Tuple[int, bytes]:
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
            material=mats[dme.dmat.materials[gltf.nodes[-1].mesh].namehash] if len(mats) > 0 else None
        )]
    ))

    return offset, blob

def load_texture(manager: AssetManager, gltf: GLTF2, textures: Dict[str, PILImage.Image], name: str, CNS_seen: List[int]):
    texture = manager.get_raw(name)
    if texture is None:
        logging.warning(f"Could not find {name} in loaded game assets, skipping...")
        return
    logging.info(f"Loaded {name}")

    im = PILImage.open(BytesIO(texture.get_data()))
    if re.match(".*_(s|S).dds", name):
        unpack_specular(manager, gltf, textures, im, name, CNS_seen)
        CNS_seen[2] += 1
    elif re.match(".*_(n|N).dds", name):
        unpack_normal(gltf, textures, im, name, CNS_seen)
        CNS_seen[1] += 1
    else:
        texture.name = str(Path(name).with_suffix(".png"))
        textures[texture.name] = im
        if re.match(".*_(c|C).dds", name):
            gltf.textures[min(CNS_seen) * 4].name = texture.name
            gltf.textures[min(CNS_seen) * 4].source = len(gltf.images)
            CNS_seen[0] += 1
        gltf.images.append(Image(uri="textures" + os.sep + texture.name))