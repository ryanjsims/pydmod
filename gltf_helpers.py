import logging
import math
import numpy
import os
import re

from re import RegexFlag

from cnk_loader import CNK0
from dme_loader import DME, Mesh as DMEMesh, HIERARCHY, RIGIFY_MAPPINGS
from DbgPack import AssetManager
from io import BytesIO
from PIL import Image as PILImage, ImageChops
from scipy.spatial.transform import Rotation
from pygltflib import *

from dme_loader.dme_loader import Bone

logger = logging.getLogger("GLTF Helpers")

EMISSIVE_STRENGTH = 5
SUFFIX_TO_TYPE = {
    "_C.png": "base",
    "_MR.png": "met",
    "_N.png": "norm",
    "_E.png": "emis"
}

texture_name_to_indices = {}

def append_dme_to_gltf(gltf: GLTF2, dme: DME, manager: AssetManager, mats: Dict[int, List[int]], textures: Dict[str, PILImage.Image], image_indices: Dict[str, int], offset: int, blob: bytes, dme_name: str, include_skeleton: bool = True) -> Tuple[int, bytes]:
    global texture_name_to_indices
    if len(gltf.samplers) == 0:
        gltf.samplers.append(Sampler(magFilter=LINEAR, minFilter=LINEAR))
    texture_groups_dict = {}
    atlas_set = set()
    for texture in dme.dmat.textures:
        group_match = re.match("(.*)_(C|c|N|n|S|s)\.dds", texture)
        atlas_match = re.match(".*(atlas|CommandControlTerminal).*\.dds", texture, flags=RegexFlag.IGNORECASE)
        if not group_match and atlas_match and texture not in atlas_set:
            atlas_set.add(texture)
        if not group_match:
            continue
        if group_match.group(1) not in texture_groups_dict:
            texture_groups_dict[group_match.group(1)] = 0
        texture_groups_dict[group_match.group(1)] += 1
    
    texture_groups = [name for name, _ in sorted(list(texture_groups_dict.items()), key=lambda pair: pair[1], reverse=True)]

    logger.info(f"Texture groups: {texture_groups}")
    logger.info(f"Atlas textures: {list(atlas_set)}")
    
    mat_info = []
    for name in texture_groups:
        for suffix in ["_C.dds", "_N.dds", "_S.dds"]:
            if str(Path(name + suffix).with_suffix(".png")) not in textures:
                load_texture(manager, gltf, textures, name + suffix, image_indices)
        mat_info_entry = {"base": None, "met": None, "norm": None, "emis": None}
        for suffix in ["_C.png", "_MR.png", "_N.png", "_E.png"]:
            if (name + suffix) not in image_indices:
                continue

            if (name + suffix) in texture_name_to_indices:
                if suffix == "_N.png":
                    mat_info_entry["norm"] = NormalMaterialTexture(index=texture_name_to_indices[name + suffix])
                else:
                    mat_info_entry[SUFFIX_TO_TYPE[suffix]] = TextureInfo(index=texture_name_to_indices[name + suffix])
                continue

            if suffix == "_N.png":
                mat_info_entry["norm"] = NormalMaterialTexture(index=image_indices[name + suffix])
            else:
                mat_info_entry[SUFFIX_TO_TYPE[suffix]] = TextureInfo(index=image_indices[name + suffix])

            texture_name_to_indices[name + suffix] = image_indices[name + suffix]
            
        mat_info.append(mat_info_entry)
    
    atlas_texture = None
    for atlas in atlas_set:
        if str(Path(atlas).with_suffix(".png")) in texture_name_to_indices:
            atlas_texture = texture_name_to_indices[str(Path(atlas).with_suffix(".png"))]
            continue
        atlas_texture = len(gltf.textures)
        load_texture(manager, gltf, textures, atlas, image_indices)
    
    mesh_materials = []
    assert len(dme.meshes) == len(dme.dmat.materials), "Mesh count != material count"

    for i, material in enumerate(dme.dmat.materials):
        if material.namehash not in mats:
            mats[material.namehash] = []

        if i < len(mat_info):
            mat_textures: Dict[str, Optional[TextureInfo]] = mat_info[i]
        elif material.name() == 'BumpRigidHologram2SidedBlend':
            mat_textures: Dict[str, Optional[TextureInfo]] = {"base": None, "met": None, "norm": None, "emis": TextureInfo(index=atlas_texture)}
        else:
            mat_textures: Dict[str, Optional[TextureInfo]] = {"base": None, "met": None, "norm": None, "emis": None}
        
        
        # look for existing material that uses same textures
        for mat_index in mats[material.namehash]:
            logger.debug(f"gltf.materials[{mat_index}] == {gltf.materials[mat_index]}")
            logger.debug(f"mat_textures == {mat_textures}")
            baseColorTexture = gltf.materials[mat_index].pbrMetallicRoughness.baseColorTexture
            emissiveTexture = gltf.materials[mat_index].emissiveTexture
            if baseColorTexture is not None and mat_textures["base"] is not None and baseColorTexture.index == mat_textures["base"].index:
                logger.info("Found existing material with same base texture")
                mesh_materials.append(mat_index)
                break
            elif emissiveTexture is not None and mat_textures["emis"] is not None and emissiveTexture.index == mat_textures["emis"].index:
                logger.info("Found existing material with same emissive texture")
                mesh_materials.append(mat_index)
                break
            elif baseColorTexture is None and mat_textures["base"] is None:
                logger.info("Found existing material with same (null) base texture")
                mesh_materials.append(mat_index)
                break
        
        # material was found and assigned to this mesh, continue
        if len(mesh_materials) > i:
            continue
        
        # material was not found - create new
        logger.info(f"Creating new material instance #{len(mats[material.namehash]) + 1}")
        mats[material.namehash].append(len(gltf.materials))
        mesh_materials.append(len(gltf.materials))

        new_mat = Material(
            name=material.name(),
            pbrMetallicRoughness = PbrMetallicRoughness(
                baseColorTexture=mat_textures["base"],
                metallicRoughnessTexture=mat_textures["met"],
                baseColorFactor=[1, 1, 1, 1] if mat_textures["base"] is not None else [0, 0, 0, 1]
            ),
            normalTexture=mat_textures["norm"],
            emissiveTexture=mat_textures["emis"],
            emissiveFactor=[EMISSIVE_STRENGTH if mat_textures["emis"] is not None else 0] * 3,
            alphaCutoff=None,
            alphaMode=OPAQUE if material.name() != "Foliage" else BLEND
        )
        gltf.materials.append(new_mat)

    
    for i, mesh in enumerate(dme.meshes):
        logger.info(f"Writing mesh {i + 1} of {len(dme.meshes)}")
        material_index = mesh_materials[i]
        swapped = False
        if len(dme.bone_map2) > 0 and i == 1:
            logger.warning("Swapping around bone maps since there were bones with the same index in the dme bone map entries.")
            logger.warning("Theoretically this should only happen for high bone count models (Colossus is one)")
            swapped = True
            temp = dme.bone_map
            dme.bone_map = dme.bone_map2
            dme.bone_map2 = temp
        offset, blob = add_mesh_to_gltf(gltf, dme, mesh, material_index, offset, blob)
        if swapped:
            dme.bone_map2 = dme.bone_map
            dme.bone_map = temp

    
    if len(dme.bones) > 0 and include_skeleton:
        offset, blob = add_skeleton_to_gltf(gltf, dme, offset, blob)
    
    return offset, blob

def unpack_specular(manager: AssetManager, gltf: GLTF2, textures: Dict[str, PILImage.Image], im: PILImage.Image, name: str, texture_indices: Dict[str, int]):
    metallic = im.getchannel("R")
    roughness = im.getchannel("A")
    metallicRoughness = PILImage.merge("RGB", [metallic, roughness, metallic])
    albedoName = name[:-5] + "C.dds" if name[-5] == "S" else "c.dds"
    albedoAsset = manager.get_raw(albedoName)
    if albedoAsset is not None:
        albedo = PILImage.open(BytesIO(albedoAsset.get_data()))
        albedoRGB = albedo.convert(mode="RGB")
        constant = PILImage.new(mode="RGB", size=albedo.size)
        mask = ImageChops.multiply(PILImage.eval(im.getchannel("B").resize(constant.size), lambda x: 255 if x > 50 else 0), albedo.getchannel("A"))
        emissive = PILImage.composite(albedoRGB, constant, mask)
        albedo.close()
        constant.close()
    else:
        emissive = im.getchannel("B").convert(mode="RGB")
    ename = name[:-5] + "E.png"
    textures[ename] = emissive
    texture_indices[ename] = len(gltf.textures)
    gltf.textures.append(Texture(source=len(gltf.images), sampler=0, name=ename))
    gltf.images.append(Image(uri="textures" + os.sep + ename))
    mrname = name[:-5] + "MR.png"
    textures[mrname] = metallicRoughness
    texture_indices[mrname] = len(gltf.textures)
    gltf.textures.append(Texture(source=len(gltf.images), sampler=0, name=mrname))
    gltf.images.append(Image(uri="textures" + os.sep + mrname))

def unpack_normal(gltf: GLTF2, textures: Dict[str, PILImage.Image], im: PILImage.Image, name: str, texture_indices: Dict[str, int]):
    is_packed = True
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
    textures[normal_name] = normal
    texture_indices[normal_name] = len(gltf.textures)
    gltf.textures.append(Texture(source=len(gltf.images), sampler=0, name=normal_name))
    gltf.images.append(Image(uri="textures" + os.sep + normal_name))


    if is_packed:
        secondary_tint = PILImage.eval(im.getchannel("R"), lambda x: 255 if x < 50 else 0)
        primary_tint = PILImage.eval(im.getchannel("B"), lambda x: 255 if x < 50 else 0)
        camo_tint = PILImage.eval(im.getchannel("B"), lambda x: 255 if x > 150 else 0)
        tints = PILImage.merge("RGB", [primary_tint, secondary_tint, camo_tint])
        tints_name = normal_name[:-5] + "T.png"
        textures[tints_name] = tints
        texture_indices[tints_name] = len(gltf.textures)
        gltf.textures.append(Texture(source=len(gltf.images), sampler=0, name=tints_name))
        gltf.images.append(Image(uri="textures" + os.sep + tints_name))

def add_mesh_to_gltf(gltf: GLTF2, dme: DME, mesh: DMEMesh, material_index: int, offset: int, blob: bytes) -> Tuple[int, bytes]:
    if mesh is None:
        return (offset, blob)
    if len(mesh.vertices[0]) == 0 or len(mesh.indices) == 0:
        logger.info("Skipping empty mesh")
        return (offset, blob)
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
    
    if len(mesh.skin_indices) > 0:
        attributes.append([JOINTS_0, len(gltf.accessors)])
        if 63 not in dme.bone_map:
            dme.bone_map[63] = 0
        skin_indices = list(map(lambda x: [dme.bone_map[val] for val in x], mesh.skin_indices))
        skin_indices_bin = numpy.array(skin_indices, dtype=numpy.ubyte).flatten().tobytes()
        gltf.accessors.append(Accessor(
            bufferView=len(gltf.bufferViews),
            componentType=UNSIGNED_BYTE,
            count=len(mesh.skin_indices),
            type=VEC4,
        ))
        gltf.bufferViews.append(BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(skin_indices_bin),
            target=ARRAY_BUFFER
        ))
        offset += len(skin_indices_bin)
        blob += skin_indices_bin

    if len(mesh.skin_weights) > 0:
        attributes.append([WEIGHTS_0, len(gltf.accessors)])
        skin_weights_bin = numpy.array(mesh.skin_weights, dtype=numpy.float32).tobytes()
        gltf.accessors.append(Accessor(
            bufferView=len(gltf.bufferViews),
            componentType=FLOAT,
            count=len(mesh.skin_weights),
            type=VEC4,
        ))
        gltf.bufferViews.append(BufferView(
            buffer=0,
            byteOffset=offset,
            byteLength=len(skin_weights_bin),
            target=ARRAY_BUFFER
        ))
        offset += len(skin_weights_bin)
        blob += skin_weights_bin
    
    # if len(mesh.colors) > 0:
    #     for index in mesh.colors:
    #         attributes.append([f"COLOR_{index}", len(gltf.accessors)])
    #         colors_bin = numpy.array(mesh.colors[index], dtype=numpy.float32).tobytes()
    #         gltf.accessors.append(Accessor(
    #             bufferView=len(gltf.bufferViews),
    #             componentType=FLOAT,
    #             count=len(mesh.colors[index]),
    #             type=VEC4,
    #         ))
    #         gltf.bufferViews.append(BufferView(
    #             buffer=0,
    #             byteOffset=offset,
    #             byteLength=len(colors_bin),
    #             target=ARRAY_BUFFER
    #         ))
    #         offset += len(colors_bin)
    #         blob += colors_bin

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


def add_skeleton_to_gltf(gltf: GLTF2, dme: DME, offset: int, blob: bytes) -> Tuple[int, bytes]:
    logger.info(f"Adding skeleton with {len(dme.bones)} bones...")
    matrices_bin = b''
    joints = []
    bone_nodes: Dict[str, int] = {}
    for bone in dme.bones:
        matrices_bin += bone.inverse_bind_pose.tobytes()
        translation = bone.inverse_bind_pose.I[3,0:3].tolist()[0]
        translation = [translation[0], translation[1], translation[2]]
        scale = []
        rotation = bone.inverse_bind_pose.I.copy()
        rotation[3,0:3] = [0., 0., 0.]
        for i in range(3):
            value = float(numpy.linalg.norm(bone.inverse_bind_pose.I[i,0:3]))
            if math.fabs(1.0 - value) < 0.001:
                value = 1.0
            scale.append(value)
            rotation[i,0:3] /= value
        r = Rotation.from_matrix(rotation[0:3,0:3])
        bone_nodes[bone.name] = len(gltf.nodes)
        joints.append(len(gltf.nodes))
        gltf.nodes.append(Node(
            translation=translation,
            rotation=r.as_quat().tolist(),
            scale=scale,
            name="{}".format(RIGIFY_MAPPINGS[bone.name] if bone.name in RIGIFY_MAPPINGS else bone.name if bone.name != '' else bone.namehash)
        ))
    for name in bone_nodes:
        if name in HIERARCHY and HIERARCHY[name] in bone_nodes:
            gltf.nodes[bone_nodes[HIERARCHY[name]]].children.append(bone_nodes[name])
        elif name in HIERARCHY:
            next_bone = HIERARCHY[name]
            while next_bone and next_bone in HIERARCHY:
                if next_bone in bone_nodes:
                    gltf.nodes[bone_nodes[next_bone]].children.append(bone_nodes[name])
                    break
                next_bone = HIERARCHY[next_bone]
    def update_transform(gltf: GLTF2, root: Node, parent: Optional[Node]):
        if len(root.children) == 0:
            return
        for child_index in root.children:
            update_transform(gltf, gltf.nodes[child_index], root)
        translation = [0, 0, 0]
        if parent is not None:
            translation = parent.translation
        logger.debug(f"Updating {root.name} position: {root.translation} - {translation}")
        root.translation = [root.translation[i] - translation[i] for i in range(3)]
    if "WORLDROOT" in bone_nodes:
        node_index = bone_nodes["WORLDROOT"]
    else:
        node_index = bone_nodes[list(bone_nodes.keys())[0]]
        logger.warning(bone_nodes)
    update_transform(gltf, gltf.nodes[node_index], None)
    logger.info("Updated bone transforms")
    for i in range(len(gltf.nodes)):
        if gltf.nodes[i].mesh is not None:
            gltf.nodes[i].skin = 0
    gltf.skins.append(Skin(inverseBindMatrices=len(gltf.accessors), joints=joints))
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews),
        componentType=FLOAT,
        count=len(dme.bones),
        type=MAT4,
    ))
    gltf.bufferViews.append(BufferView(
        buffer=0,
        byteOffset=offset,
        byteLength=len(matrices_bin)
    ))
    offset += len(matrices_bin)
    blob += matrices_bin

    return offset, blob

def add_chunk_to_gltf(gltf: GLTF2, chunk: CNK0, material_index: int, offset: int, blob: bytes) -> Tuple[int, bytes]:
    chunk.calculate_verts()

    if len(chunk.verts) == 0 or len(chunk.triangles) == 0:
        logger.info("Skipping empty chunk")
        return (offset, blob)

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
            indices=gltf_mesh_indices,
            material=material_index
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
        return
    elif re.match(".*_(n|N).dds", name):
        unpack_normal(gltf, textures, im, name, texture_indices)
        return
    elif re.match(".*_(c|C).dds", name):
        #gltf.textures[min(CNS_seen) * 4].name = name
        texture_indices[str(Path(name).with_suffix(".png"))] = len(gltf.images)
        name = str(Path(name).with_suffix(".png"))
        textures[name] = im
    else:
        texture_indices[str(Path(name).with_suffix(".png"))] = len(gltf.images)
        name = str(Path(name).with_suffix(".png"))
        textures[name] = im
    gltf.textures.append(Texture(source=len(gltf.images), sampler=0, name=name))
    gltf.images.append(Image(uri="textures" + os.sep + name))