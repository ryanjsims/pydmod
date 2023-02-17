import math
import struct
import json
import os
import logging
from io import BytesIO
from typing import List, Optional, Tuple, Dict
from enum import IntEnum
from aabbtree import AABB
from numpy import indices
import numpy

from . import jenkins
from .data_classes import VertexStream, InputLayout, MaterialDefinition, ParameterGroup, LayoutUsage, input_layout_formats
from .ps2_bone_map import BONE_HASHMAP

logger = logging.getLogger("dme_loader")

with open(os.path.join(os.path.dirname(__file__), "materials_new.json")) as f:
    materialsJson: Dict[str, Dict[str, Dict]] = json.load(f)

InputLayouts = {key: InputLayout.from_json(value) for key, value in materialsJson["inputLayouts"].items()}
MaterialDefinitions = {int(key): MaterialDefinition.from_json(value) for key, value in materialsJson["materialDefinitions"].items()}
ParameterGroups = {key: ParameterGroup.from_json(value) for key, value in materialsJson["parameterGroups"].items()}

def normalize(vertex: Tuple[float, float, float]):
    length = math.sqrt(vertex[0] ** 2 + vertex[1] ** 2 + vertex[2] ** 2)
    if length > 0:
        return vertex[0] / length, vertex[1] / length, vertex[2] / length
    return vertex

class Bone:
    name: str
    def __init__(self):
        self.inverse_bind_pose: numpy.matrix = []
        self.bbox: AABB = None
        self._namehash: int = -1
        self.name = ""
    
    @property
    def namehash(self):
        return self._namehash

    @namehash.setter
    def namehash(self, value: int):
        self._namehash = value
        if value in BONE_HASHMAP:
            self.name = BONE_HASHMAP[value]
    
    def __repr__(self):
        return f"Bone(namehash={self.namehash}, bbox={self.bbox}, inverse_bind_pose={self.inverse_bind_pose})"
    
    def __str__(self):
        return f"Bone(name: {self.name if self.name != '' else self._namehash})"

class BoneMapEntry:
    def __init__(self, bone_index: int, global_index: int):
        self.bone_index = bone_index
        self.global_index = global_index
    
    def serialize(self):
        return struct.pack("<HH", self.bone_index, self.global_index)

    @classmethod
    def load(cls, data: BytesIO) -> 'BoneMapEntry':
        logger.debug("Loading bone map entry data")
        return cls(*struct.unpack("<HH", data.read(4)))

class DrawCall:
    def __init__(self, unknown0: int, bone_start: int, bone_count: int,
                        delta: int, unknown1: int, vertex_offset: int, 
                        vertex_count: int, index_offset: int, index_count: int):
        self.unknown0 = unknown0
        self.bone_start = bone_start
        self.bone_count = bone_count
        self.delta = delta
        self.unknown1 = unknown1
        self.vertex_offset = vertex_offset
        self.vertex_count = vertex_count
        self.index_offset = index_offset
        self.index_count = index_count
    
    def serialize(self):
        return struct.pack("<IIIIIIIII", 
            self.unknown0, self.bone_start, self.bone_count, 
            self.delta, self.unknown1, self.vertex_offset, 
            self.vertex_count, self.index_offset, self.index_count)
    
    def __len__(self):
        return 36
    
    def __str__(self):
        return f"""DrawCall(
    unknown0={self.unknown0},
    bone_start={self.bone_start},
    bone_count={self.bone_count},
    delta={self.delta},
    unknown1={self.unknown1},
    vertex_offset={self.vertex_offset},
    vertex_count={self.vertex_count},
    index_offset={self.index_offset},
    index_count={self.index_count}
)"""

    @classmethod
    def load(cls, data: BytesIO):
        logger.debug("Loading draw call data")
        return cls(*struct.unpack("<IIIIIIIII", data.read(36)))

class Mesh:
    def __init__(self, bytes_per_vertex: List[int], vertex_streams: List[VertexStream], vertices: Dict[int, List[Tuple[float]]], colors: Dict[int, List[Tuple]],
                    normals: Dict[int, List[Tuple[float]]], binormals: Dict[int, List[Tuple[float]]], tangents: Dict[int, List[Tuple[float]]], uvs: Dict[int, List[Tuple[float]]], 
                    skin_weights: List[Tuple[float]], skin_indices: List[Tuple[int]], index_size: int, 
                    indices: List[int], draw_offset: int, draw_count: int, bone_count: int):
        self.vertex_size = bytes_per_vertex
        self.vertex_streams = vertex_streams
        self.vertices = vertices
        self.colors = colors
        self.normals = normals
        self.binormals = binormals
        self.tangents = tangents
        self.uvs = uvs
        self.skin_weights = skin_weights
        self.skin_indices = skin_indices
        self.index_size = index_size
        self.indices = indices
        self.draw_offset = draw_offset
        self.draw_count = draw_count
        self.bone_count = bone_count
        self.__serialized = None
    
    def close(self):
        for stream in self.vertex_streams:
            stream.close()
        del self.vertices
        del self.colors
        del self.normals
        del self.binormals
        del self.tangents
        del self.uvs
        del self.skin_indices
        del self.skin_weights
        del self.indices

    def __str__(self) -> str:
        return f"Mesh (vertex count {len(self.vertices[0])} draw offset {self.draw_offset} indices {len(self.indices)})"
    
    def serialize(self) -> bytes:
        if not self.__serialized:
            self.__serialized = (
                struct.pack("<IIII", self.draw_offset, self.draw_count, self.bone_count, 0xffffffff)
                + struct.pack("<IIII", len(self.vertex_streams), self.index_size, len(self.indices), len(self.vertices))
                + b''.join([struct.pack("<I", stream.stride) + stream.data for stream in self.vertex_streams])
                + b''.join([struct.pack("<H" if self.index_size == 2 else "<I", index) for index in self.indices])
            )
        return self.__serialized
    
    def __len__(self):
        return (
            32 + sum([4 + len(stream.data) for stream in self.vertex_streams]) 
            + self.index_size * len(self.indices) + 4
        )
    
    @classmethod
    def load(cls, data: BytesIO, input_layout: Optional[InputLayout]) -> Optional['Mesh']:
        logger.info("Loading mesh data")
        draw_offset, draw_count, bone_count, unknown = struct.unpack("<IIII", data.read(16))
        logger.info(f"{draw_offset=} {draw_count=} {bone_count=}")
        assert unknown == 0xFFFFFFFF, "{:x}".format(unknown)
        vert_stream_count, index_size, index_count, vertex_count = struct.unpack("<IIII", data.read(16))
        
        bpv_list = []
        vertices: Dict[int, List[Tuple]] = {}
        colors: Dict[int, List[Tuple]] = {}
        uvs: Dict[int, List[Tuple]] = {}
        normals: Dict[int, List[Tuple]] = {}
        binormals: Dict[int, List[Tuple]] = {}
        tangents: Dict[int, List[Tuple]] = {}
        vertex_streams: List[VertexStream] = []
        skin_indices = []
        skin_weights = []
        for _ in range(vert_stream_count):
            bytes_per_vertex = struct.unpack("<I", data.read(4))[0]
            bpv_list.append(bytes_per_vertex)
            vertex_streams.append(VertexStream(bytes_per_vertex, data.read(bytes_per_vertex * vertex_count)))
        
        logger.info(f"Loaded {vert_stream_count} vertex streams - lengths {', '.join(map(str, map(len, vertex_streams)))}")
        logger.info(f"Byte strides: {', '.join(map(str, bpv_list))}")
        
        if not input_layout or (input_layout.sizes is not None and not all([input_layout.sizes[i] == bpv_list[i] for i in range(len(bpv_list))])):
            logger.warning("Input layout not provided or has incorrect strides, guessing based on byte strides...")
            options: List[Tuple[str, InputLayout]] = []
            for name, layout in InputLayouts.items():
                if len(bpv_list) != len(layout.sizes):
                    continue
                if all([size == layout.sizes[i] for i, size in enumerate(bpv_list)]):
                    options.append((name, layout))
            if len(options) == 1:
                input_layout = options[0][1]
                resp = 1
            else:
                logger.warning(f"Strides: {', '.join(map(str, bpv_list))}")
                logger.warning("Available matching layouts:")
                for i, (name, layout) in enumerate(options):
                    logger.warning(f"  {i+1}. {name} [{hash(layout)}] - {', '.join(map(str, layout.sizes))}")
                if len(options) == 0:
                    logger.warning("  None! Skipping model")
                    return None
                resp = ""
                while not resp.isnumeric() or (int(resp) - 1 < 0 or int(resp) > len(options)):
                    resp = input("Enter the number of the material to use: ")
                input_layout = options[int(resp) - 1][1]
            logger.warning(f"Using material '{options[int(resp) - 1][0]}'...")

        logger.info(f"Loading {vertex_count} vertices...")
        logger.debug(f"Entries - {input_layout.entries}")
        logger.info(f"Input Layout: {input_layout.name}")
        is_rigid = ("rigid" in input_layout.name.lower() or "vehicle" in input_layout.name.lower()) and bone_count > 0
        try:
            for i in range(vertex_count):
                for entry in input_layout.entries:
                    stream = vertex_streams[entry.stream]
                    format, size = input_layout_formats[entry.type]
                    #logger.debug(f"Vertex {i} stream {entry.stream} - {entry.type}")
                    #logger.debug(f"Stream at position {stream.tell()}")
                    value = struct.unpack(format, stream.data.read(size))
                    orig_value = value
                    if entry.type == "ubyte4n":
                        value = [(val[0] / 255 * 2) - 1 for val in value]
                    elif entry.type == "Float1":
                        value = value[0]
                    elif entry.type == "D3dcolor":
                        value = [(((value[0] >> i * 8) & 0xFF) / 255 * 2) - 1 for i in range(4)]

                    if entry.usage == LayoutUsage.POSITION:
                        if entry.usage_index not in vertices:
                            vertices[entry.usage_index] = []
                        vertices[entry.usage_index].append(value)
                    elif entry.usage == LayoutUsage.NORMAL:
                        if entry.usage_index not in normals:
                            normals[entry.usage_index] = []
                        normals[entry.usage_index].append(value)
                    elif entry.usage == LayoutUsage.BINORMAL:
                        if entry.usage_index not in binormals:
                            binormals[entry.usage_index] = []
                        binormals[entry.usage_index].append(value)
                        if is_rigid:
                            skin_indices.append([orig_value[3][0], 0, 0, 0])
                    elif entry.usage == LayoutUsage.TANGENT:
                        if entry.usage_index not in tangents:
                            tangents[entry.usage_index] = []
                        tangents[entry.usage_index].append(value)
                        if is_rigid:
                            skin_weights.append([1., 0., 0., 0.])
                    elif entry.usage == LayoutUsage.BLENDWEIGHT:
                        skin_weights.append(value if entry.type != "ubyte4n" else [(val + 1) / 2 for val in value])
                    elif entry.usage == LayoutUsage.BLENDINDICES:
                        skin_indices.append([((orig_value[0] >> i * 8) & 0xFF) for i in range(4)])
                    elif entry.usage == LayoutUsage.TEXCOORD:
                        if entry.usage_index not in uvs:
                            uvs[entry.usage_index] = []
                        uvs[entry.usage_index].append(value)
                    elif entry.usage == LayoutUsage.COLOR:
                        if entry.usage_index not in colors:
                            colors[entry.usage_index] = []
                        colors[entry.usage_index].append(value)
        except struct.error as e:
            logger.error(f"Failed to read data at vertex {i}, format {entry.type}, stream {entry.stream}, stream position {vertex_streams[entry.stream].tell()}")
            raise e

        for entry_index in binormals:
            if entry_index not in normals and len(binormals[entry_index]) > 0 and len(tangents[entry_index]) > 0:
                normals[entry_index] = []
                for binormal, tangent in zip(binormals[entry_index], tangents[entry_index]):
                    b = normalize(binormal)
                    t = normalize(tangent)
                    if len(tangent) == 4:
                        sign = tangent[3]
                    else:
                        sign = -1
                    n = normalize((
                        b[1] * t[2] - b[2] * t[1],
                        b[2] * t[0] - b[0] * t[2],
                        b[0] * t[1] - b[1] * t[0],
                    ))
                    n = [val * sign for val in n]
                    normals[entry_index].append(n)
        
        indices = []
        index_format = "<H"
        if index_size == 0x8000_0004:
            index_size = 4
            index_format = "<I"
        
        index_data = data.read(index_size * index_count)
        
        for index_tuple in struct.iter_unpack(index_format, index_data):
            indices.append(index_tuple[0])
        
        temp_indices = []
        for i in range(0, len(indices), 3):
            temp_indices.extend(indices[i:i+3][::-1])
        indices = temp_indices
        
        return cls(bpv_list, vertex_streams, vertices, colors, normals, binormals, tangents, uvs, skin_weights, skin_indices, index_size, indices, draw_offset, draw_count, bone_count)
        
class D3DXParamType(IntEnum):
    VOID=           0
    BOOL=           1
    INT=            2
    FLOAT=          3
    STRING=         4
    TEXTURE=        5
    TEXTURE1D=      6
    TEXTURE2D=      7
    TEXTURE3D=      8
    TEXTURECUBE=    9
    SAMPLER=        10
    SAMPLER1D=      11
    SAMPLER2D=      12
    SAMPLER3D=      13
    SAMPLERCUBE=    14
    PIXELSHADER=    15
    VERTEXSHADER=   16
    PIXELFRAGMENT=  17
    VERTEXFRAGMENT= 18
    UNSUPPORTED=    19
    FORCE_DWORD=    0x7fffffff

class D3DXParamClass(IntEnum):
    SCALAR=         0
    VECTOR=         1
    MATRIX_ROWS=    2
    MATRIX_COLS=    3
    OBJECT=         4
    STRUCT=         5
    FORCE_DWORD=    0x7fffffff

class Parameter:
    def __init__(self, namehash: int, param_class: D3DXParamClass, param_type: D3DXParamType, data: bytes):
        self.namehash = namehash
        self.name = None
        for group in ParameterGroups:
            if self.namehash in ParameterGroups[group]:
                self.name = ParameterGroups[group][self.namehash].name
                break
        self._class = param_class
        self._type = param_type
        self.value = None
        self.vector = None
        if self._class == D3DXParamClass.VECTOR and self._type == D3DXParamType.FLOAT:
            self.vector = tuple([val[0] for val in struct.iter_unpack("<f", data)])
        elif self._class == D3DXParamClass.VECTOR and self._type == D3DXParamType.INT:
            self.vector = tuple([val[0] for val in struct.iter_unpack("<i", data)])
        elif self._class == D3DXParamClass.VECTOR and self._type == D3DXParamType.BOOL:
            self.vector = tuple([val[0] for val in struct.iter_unpack("<?", data)])
        elif self._type == D3DXParamType.STRING:
            self.value = data.decode("utf-8")
        elif self._type == D3DXParamType.INT:
            self.value = struct.unpack("<i", data)[0]
        elif self._type == D3DXParamType.FLOAT:
            self.value = struct.unpack("<f", data)[0]
        elif self._type == D3DXParamType.BOOL:
            self.value = struct.unpack("<i", data)[0]
        self.data = data
        if self.vector is not None:
            logger.debug(f"    vector: {self.vector}")
        if self.value is not None:
            logger.debug(f"    value:  {self.value}")
        if self._class == D3DXParamClass.OBJECT:
            logger.debug(f"    object: {struct.unpack('<I', data)[0]}")

    def close(self):
        del self.data

    def __len__(self):
        return 16 + len(self.data)

    def serialize(self) -> bytes:
        return struct.pack("<IIII", self.namehash, self._class, self._type, len(self.data)) + self.data

    def __str__(self) -> str:
        return f"DMAT Parameter {self.name + ' ' if self.name else ''}{self._class} {self._type} {repr(self.data) if not self.vector else repr(self.vector) if not self.value else repr(self.value)}"
    
    def __repr__(self):
        return f"Parameter({self.namehash}, {repr(self._class)}, {repr(self._type)}, {repr(self.data) if not self.vector else repr(self.vector) if not self.value else repr(self.value)})"

    @classmethod
    def load(cls, data: BytesIO) -> 'Parameter':
        logger.debug("Loading parameter")
        param_hash, param_class, param_type, length = struct.unpack("<IIII", data.read(16))
        param_data = data.read(length)
        t = D3DXParamType(param_type)
        c = D3DXParamClass(param_class)
        logger.debug(f"    type:   {t.name}")
        logger.debug(f"    class:  {c.name}")

        return cls(param_hash, c, t, param_data)

class Material:
    def __init__(self, namehash: int, definition: int, parameters: List[Parameter]):
        self.namehash = namehash
        self.definition = definition
        self.parameters = parameters
        self.__encoded_parameters = None
    
    def close(self):
        for parameter in self.parameters:
            parameter.close()
        del self.parameters

    def __len__(self):
        return 16 + len(self.encode_parameters())
    
    def data_length(self):
        return 12 + len(self.encode_parameters())
    
    def __str__(self) -> str:
        return f"Material ({self.namehash} {self.definition} params: {len(self.parameters)})"

    def serialize(self) -> bytes:
        return struct.pack("<IIII", self.namehash, self.data_length(), self.definition, len(self.parameters)) + self.encode_parameters
    
    def encode_parameters(self) -> bytes:
        if self.__encoded_parameters is None:
            self.__encoded_parameters = b''.join([param.serialize() for param in self.parameters])
        return self.__encoded_parameters
    
    def name(self) -> Optional[str]:
        if self.definition in MaterialDefinitions:
            return MaterialDefinitions[self.definition].name
        return None
    
    def input_layout(self, material_hash: Optional[int] = None) -> InputLayout:
        input_layout = None
        try:
            usedhash = self.definition if material_hash is None else material_hash
            definition = MaterialDefinitions[usedhash]
            logger.info(f"Material definition '{definition.name}' found (name hash {usedhash})")
        except KeyError:
            logger.warning(f"Material definition not found for definition hash {usedhash}! Defaulting to 'VehicleRigid'")
            definition = MaterialDefinitions[570440238] #"VehicleRigid"
            draw_style = definition.draw_styles[0]
            input_layout = InputLayouts[definition.draw_styles[0].input_layout] #"Vehicle"
        
        if not input_layout:
            try:
                draw_style = definition.draw_styles[0]
                input_layout = InputLayouts[draw_style.input_layout]
                logger.info(f"Input Layout '{input_layout.name}' found (name hash {jenkins.oaat(draw_style.input_layout.encode('utf-8'))})")
            except KeyError:
                logger.warning(f"Input Layout not found for name hash {jenkins.oaat(draw_style.input_layout.encode('utf-8'))}! Defaulting to 'Vehicle'")
                input_layout = InputLayouts[2340912194] #"Vehicle"
        
        return input_layout

    @classmethod
    def load(cls, data: BytesIO) -> 'Material':
        offset = 0
        namehash, length, definition, num_params = struct.unpack("<IIII", data.read(16))
        logger.info(f"Material data - Name hash: 0x{namehash:08X}    Length: {length}    Definition hash: 0x{definition:08X}    Parameter count: {num_params}")
        offset += 16
        parameters: List[Parameter] = []
        for i in range(num_params):
            parameters.append(Parameter.load(data))
            offset += len(parameters[i])
        assert length + 8 == offset, f"Material data length different than stored length ({length + 8} !== {offset})"

        return cls(namehash, definition, parameters)

class DMAT:
    def __init__(self, magic: bytes, version: int, texture_names: List[str], materials: List[Material]):
        self.magic = magic.decode("utf-8")
        assert self.magic == "DMAT", "Not a DMAT chunk"
        self.version = version
        self.textures = texture_names
        self.materials = materials
        self.__encoded_textures = None
        self.__encoded_materials = None
        self.__length = None

    def close(self):
        for material in self.materials:
            material.close()
        del self.materials
        del self.textures

    def serialize(self) -> bytes:
        return struct.pack("<I", len(self)) + self.magic.encode("utf-8") + struct.pack("<I", self.version) + self.encode_textures() + self.encode_materials()
    
    def __len__(self) -> int:
        if self.__length is None:
            self.__length = len(self.magic.encode("utf-8")) + 8 + sum([len(name.encode("utf-8") + b'\x00') for name in self.textures]) + 4 + sum([len(material)] for material in self.materials)
        return self.__length

    def encode_textures(self) -> bytes:
        if self.__encoded_textures is None:
            self.__encoded_textures = b'\x00'.join([name.encode("utf-8") for name in self.textures]) + b'\x00'
            self.__encoded_textures = struct.pack("<I", len(self.__encoded_textures)) + self.__encoded_textures
        return self.__encoded_textures
    
    def encode_materials(self) -> bytes:
        if self.__encoded_materials is None:
            self.__encoded_materials = struct.pack("<I", len(self.materials)) + b''.join([material.serialize() for material in self.materials])
        return self.__encoded_materials
    
    @classmethod
    def load(cls, data: BytesIO, dmat_length: int = None) -> 'DMAT':
        logger.info("Loading DMAT chunk")
        if not dmat_length:
            dmat_length = struct.unpack("<I", data.read(4))[0]
        offset = 0
        magic = data.read(4)
        offset += 4
        assert magic.decode("utf-8") == "DMAT", "Not a DMAT chunk"
        version, filename_length = struct.unpack("<II", data.read(8))
        offset += 8
        assert version == 1, f"Unknown DMAT version {version}"
        
        name_data = data.read(filename_length)
        texture_names = [name for name in name_data.decode("utf-8").strip('\x00').split("\x00") if name != '']
        logger.info("Textures:\n\t" + '\n\t'.join(texture_names))
        offset += filename_length

        material_count = struct.unpack("<I", data.read(4))[0]
        offset += 4
        logger.info(f"Material count: {material_count}")

        materials = []
        for i in range(material_count):
            materials.append(Material.load(data))
            offset += len(materials[i])
        
        assert offset == dmat_length, "Data length does not match stored length!"
        logger.info("DMAT chunk loaded")
        return cls(magic, version, texture_names, materials)

class DME:
    magic: str
    version: int
    dmat: DMAT
    aabb: AABB
    meshes: List[Mesh]
    draw_calls: List[DrawCall]
    bone_map: Dict[int, int]
    bones: List[Bone]
    bone_map_entries: List[BoneMapEntry]
    
    def __init__(self, magic: str, version: int, dmat: DMAT, aabb: AABB, meshes: List[Mesh], draw_calls: List[DrawCall], bone_map: Dict[int, int], bones: List[Bone], bone_map_entries: List[BoneMapEntry], bone_map2: Dict[int, int]):
        assert magic == "DMOD", "Not a DME file"
        assert version == 4, "Unsupported DME version"
        self.magic = magic
        self.version = version
        self.dmat = dmat
        self.aabb = aabb
        self.meshes = meshes
        self.draw_calls = draw_calls
        self.bone_map = bone_map
        self.bones = bones
        self.bone_map_entries = bone_map_entries
        self.bone_map2 = bone_map2

    def close(self):
        self.dmat.close()
        for mesh in self.meshes:
            mesh.close()
        del self.dmat
        del self.meshes
        del self.draw_calls
        del self.bone_map
        del self.bones
    
    def input_layout(self, index):
        if 0 <= index < len(self.dmat.materials):
            return self.dmat.materials[index].input_layout()

    @classmethod
    def load(cls, data: BytesIO, material_hashes: Optional[List[int]] = None, textures_only: bool = False) -> 'DME':
        logger.info("Loading DME file")
        #DMOD block
        magic = data.read(4)
        assert magic.decode("utf-8") == "DMOD", "Not a DME file"
        
        version = struct.unpack("<I", data.read(4))[0]
        assert version == 4, f"Unsupported DME version {version}"
        
        #DMAT block
        dmat = DMAT.load(data)
        
        #MESH block
        minx, miny, minz = struct.unpack("<fff", data.read(12))
        maxx, maxy, maxz = struct.unpack("<fff", data.read(12))
        num_meshes = struct.unpack("<I", data.read(4))[0]
        logger.info(f"{num_meshes} meshes to load...")
        aabb = AABB([(minx, maxx), (miny, maxy), (minz, maxz)])
        meshes = []
        if textures_only:
            logger.info("Skipping mesh loading due to texture only request")
            meshes = [None] * num_meshes
            return cls(magic.decode("utf-8"), version, dmat, aabb, meshes)
        
        for i in range(num_meshes):
            logger.info(f"Loading Mesh {i}")
            material_hash = None
            if material_hashes and i < len(material_hashes):
                material_hash = material_hashes[i]
            mesh = Mesh.load(data, dmat.materials[i].input_layout(material_hash))
            meshes.append(mesh)
            logger.info(f"Mesh {i} loaded")

        #Draw info block
        draw_call_count = struct.unpack("<I", data.read(4))[0]
        logger.info(f"Loading {draw_call_count} draw calls")
        draw_calls: List[DrawCall] = [DrawCall.load(data) for _ in range(draw_call_count)]
        newline = '\n'
        logger.debug(f"Draw calls:\n{newline.join([str(draw_call) for draw_call in draw_calls])}")

        bone_map_entry_count = struct.unpack("<I", data.read(4))[0]
        logger.info(f"Loading {bone_map_entry_count} bone map entries")
        bone_map_entries = [BoneMapEntry.load(data) for _ in range(bone_map_entry_count)]
        bone_map = {}
        bone_map2 = {}
        for entry in bone_map_entries:
            if entry.global_index in bone_map:
                bone_map[entry.global_index + 64] = entry.bone_index
                bone_map2[entry.global_index] = entry.bone_index
            else:
                bone_map[entry.global_index] = entry.bone_index
        #bone_map = {entry.global_index + (64 if i > 63 else 0): entry.bone_index for i, entry in enumerate(bone_map_entries)}

        bones_count = struct.unpack("<I", data.read(4))[0]
        logger.info(f"Loading {bones_count} bones")
        bones: List[Bone] = [Bone() for _ in range(bones_count)]
        for bone in bones:
            matrix = struct.unpack("<ffffffffffff", data.read(48))
            bone.inverse_bind_pose = numpy.matrix([
                [*matrix[ :3], 0,],
                [*matrix[3:6], 0,],
                [*matrix[6:9], 0,],
                [*matrix[9: ], 1,]
            ], dtype=numpy.float32)
        
        for bone in bones:
            bbox = struct.unpack("<ffffff", data.read(24))
            if bbox[0] < bbox[4]:
                bone.bbox = AABB([(bbox[0], bbox[3]), (bbox[1], bbox[4]), (bbox[2], bbox[5])])
            else:
                bone.bbox = AABB()
        
        for bone in bones:
            bone.namehash = struct.unpack("<I", data.read(4))[0]
        
        logger.info("DME file loaded")
        return cls(magic.decode("utf-8"), version, dmat, aabb, meshes, draw_calls, bone_map, bones, bone_map_entries, bone_map2)