import math
import struct
import json
import os
import logging
from io import BytesIO
from typing import List, Tuple, Dict
from enum import IntEnum
from aabbtree import AABB

from . import jenkins
from .data_classes import *

logger = logging.getLogger("dme_loader")
logger.setLevel(logging.DEBUG)

with open(os.path.join(os.path.dirname(__file__), "materials.json")) as f:
    materialsJson: Dict[str, Dict[str, Dict]] = json.load(f)

InputLayouts = {int(key) : InputLayout.from_json(value) for key, value in materialsJson["inputLayouts"].items()}
MaterialDefinitions = {int(key) : MaterialDefinition.from_json(value) for key, value in materialsJson["materialDefinitions"].items()}

input_layout_formats ={
    "Float3":       ("<fff", 12),
    "D3dcolor":     ("<I", 4),
    "Float2":       ("<ff", 8),
    "Float4":       ("<ffff", 16),
    "ubyte4n":      ("<cccc", 4),
    "Float16_2":    ("<ee", 4),
    "float16_2":    ("<ee", 4),
    "Short2":       ("<HH", 4),
    "Float1":       ("<f", 4),
    "Short4":       ("<HHHH", 8)
}

def normalize(vertex: Tuple[float, float, float]):
    length = math.sqrt(vertex[0] ** 2 + vertex[1] ** 2 + vertex[2] ** 2)
    if length > 0:
        return vertex[0] / length, vertex[1] / length, vertex[2] / length
    return vertex

class Bone:
    def __init__(self):
        self.inverse_bind_pose: List[float] = []
        self.bbox: List[float] = []
        self.namehash: int = -1

class BoneMapEntry:
    def __init__(self, bone_index: int, global_index: int):
        self.bone_index = bone_index
        self.global_index = global_index
    
    def serialize(self):
        return struct.pack("<HH", self.bone_index, self.global_index)

    @classmethod
    def load(cls, data: BytesIO) -> 'BoneMapEntry':
        logger.info("Loading bone map entry data")
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

    @classmethod
    def load(cls, data: BytesIO):
        logger.info("Loading draw call data")
        return cls(*struct.unpack("<IIIIIIIII", data.read(36)))

class Mesh:
    def __init__(self, bytes_per_vertex: List[int], vertex_streams: List[VertexStream], vertices: List[Tuple[float]], normals: List[Tuple[float]], 
                    binormals: List[Tuple[float]], tangents: List[Tuple[float]], uvs: Dict[int, List[Tuple[float]]], 
                    skin_weights: List[Tuple[float]], skin_indices: List[Tuple[int]], index_size: int, 
                    indices: List[int], draw_offset: int, draw_count: int, bone_count: int, draw_calls: List[DrawCall], bone_map_entries: Dict[int, int], bones: List[Bone]):
        self.vertex_size = bytes_per_vertex
        self.vertex_streams = vertex_streams
        self.vertices = vertices
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
        self.draw_calls = draw_calls
        self.bone_map = bone_map_entries
        self.bones = bones
        self.__serialized = None

    def __str__(self) -> str:
        return f"Mesh (vertex count {len(self.vertices)} draw calls {len(self.draw_calls)} indices {len(self.indices)})"
    
    def serialize(self) -> bytes:
        if not self.__serialized:
            self.__serialized = (
                struct.pack("<IIII", self.draw_offset, self.draw_count, self.bone_count, 0xffffffff)
                + struct.pack("<IIII", len(self.vertex_streams), self.index_size, len(self.indices), len(self.vertices))
                + b''.join([struct.pack("<I", stream.stride) + stream.data for stream in self.vertex_streams])
                + b''.join([struct.pack("<H" if self.index_size == 2 else "<I", index) for index in self.indices])
                + struct.pack("<I", len(self.draw_calls))
                + b''.join([draw_call.serialize() for draw_call in self.draw_calls])
                + struct.pack("<I", len(self.bone_map))
                + b''.join([struct.pack("<HH", bone_index, global_index) for bone_index, global_index in self.bone_map])
                + b''.join([struct.pack("<ffffffffffff", *bone.inverse_bind_pose[:3], *bone.inverse_bind_pose[4:7], *bone.inverse_bind_pose[8:11], *bone.inverse_bind_pose[12:15]) for bone in self.bones])
                + b''.join([struct.pack("<ffffff", *bone.bbox) for bone in self.bones])
                + b''.join([struct.pack("<I", bone.namehash) for bone in self.bones])
            )
        return self.__serialized
    
    def __len__(self):
        return (
            32 + sum([4 + len(stream.data) for stream in self.vertex_streams]) 
            + self.index_size * len(self.indices) + 4 + len(self.draw_calls) * len(self.draw_calls[0])
            + 4 + 4 * len(self.bone_map) + (48 + 24 + 4) * len(self.bones)
        )
    
    @classmethod
    def load(cls, data: BytesIO, input_layout: InputLayout) -> 'Mesh':
        logger.info("Loading mesh data")
        draw_offset, draw_count, bone_count, unknown = struct.unpack("<IIII", data.read(16))
        assert unknown == 0xFFFFFFFF, "{:x}".format(unknown)
        vert_stream_count, index_size, index_count, vertex_count = struct.unpack("<IIII", data.read(16))
        
        bpv_list = []
        vertices = []
        uvs: Dict[int, List[Tuple]] = {}
        normals = []
        binormals = []
        tangents = []
        vertex_streams: List[VertexStream] = []
        skin_indices = []
        skin_weights = []
        for _ in range(vert_stream_count):
            bytes_per_vertex = struct.unpack("<I", data.read(4))[0]
            bpv_list.append(bytes_per_vertex)
            vertex_streams.append(VertexStream(bytes_per_vertex, data.read(bytes_per_vertex * vertex_count)))
        
        logger.info(f"Loaded {vert_stream_count} vertex streams")
        logger.info(f"Byte strides: {', '.join(map(str, bpv_list))}")

        for _ in range(vertex_count):
            for entry in input_layout.entries:
                stream = vertex_streams[entry.stream]
                format, size = input_layout_formats[entry.type]
                value = struct.unpack(format, stream.data.read(size))
                if entry.type == "ubyte4n":
                    value = [(val[0] / 255 * 2) - 1 for val in value]
                elif entry.type == "Float1" or entry.type == "D3dcolor":
                    value = value[0]

                if entry.usage == LayoutUsage.POSITION:
                    vertices.append(value)
                elif entry.usage == LayoutUsage.NORMAL:
                    normals.append(value)
                elif entry.usage == LayoutUsage.BINORMAL:
                    binormals.append(value)
                elif entry.usage == LayoutUsage.TANGENT:
                    tangents.append(value)
                elif entry.usage == LayoutUsage.BLENDWEIGHT:
                    skin_weights.append(value)
                elif entry.usage == LayoutUsage.BLENDINDICES:
                    skin_indices.append(value)
                elif entry.usage == LayoutUsage.TEXCOORD:
                    if entry.usage_index not in uvs:
                        uvs[entry.usage_index] = []
                    uvs[entry.usage_index].append(value)                    
        
        if len(normals) == 0 and len(binormals) > 0 and len(tangents) > 0:
            for binormal, tangent in zip(binormals, tangents):
                b = normalize(binormal)
                t = normalize(tangent)
                if len(tangent) == 4:
                    sign = -tangent[3]
                else:
                    sign = 1
                n = normalize((
                    b[1] * t[2] - b[2] * t[1],
                    b[2] * t[0] - b[0] * t[2],
                    b[0] * t[1] - b[1] * t[0],
                ))
                n = [val * sign for val in n]
                normals.append(n)
        
        indices = []
        index_format = "<H"
        if index_size == 0x8000_0004:
            index_size = 4
            index_format = "<I"
        
        index_data = data.read(index_size * index_count)
        
        for index_tuple in struct.iter_unpack(index_format, index_data):
            indices.append(index_tuple[0])
        
        draw_call_count = struct.unpack("<I", data.read(4))[0]
        draw_calls = [DrawCall.load(data) for _ in range(draw_call_count)]

        bone_map_entry_count = struct.unpack("<I", data.read(4))[0]
        bone_map_entries = [BoneMapEntry.load(data) for _ in range(bone_map_entry_count)]
        bone_map = {entry.bone_index: entry.global_index for entry in bone_map_entries}

        bones_count = struct.unpack("<I", data.read(4))[0]
        logger.info(f"Loading {bones_count} bones")
        bones = [Bone() for _ in range(bones_count)]
        for bone in bones:
            matrix = struct.unpack("<ffffffffffff", data.read(48))
            bone.inverse_bind_pose = [
                *matrix[ :3], 0,
                *matrix[3:6], 0, 
                *matrix[6:9], 0,
                *matrix[9: ], 1
            ]

        for bone in bones:
            bone.bbox = struct.unpack("<ffffff", data.read(24))
        
        for bone in bones:
            bone.namehash = struct.unpack("<I", data.read(4))[0]
        
        return cls(bpv_list, vertex_streams, vertices, normals, binormals, tangents, uvs, skin_weights, skin_indices, index_size, indices, draw_offset, draw_count, bone_count, draw_calls, bone_map, bones)

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
        self._class = param_class
        self._type = param_type
        if self._class == D3DXParamClass.VECTOR and self._type == D3DXParamType.FLOAT:
            self.vector = tuple([val[0] for val in struct.iter_unpack("<f", data)])
        else:
            self.vector = tuple()
        self.data = data

    def __len__(self):
        return 16 + len(self.data)

    def serialize(self) -> bytes:
        return struct.pack("<IIII", self.namehash, self._class, self._type, len(self.data)) + self.data

    def __str__(self) -> str:
        return f"DMAT Parameter {self._class} {self._type} {repr(self.data) if not self.vector else repr(self.vector)}"
    
    def __repr__(self):
        return f"Parameter({self.namehash}, {repr(self._class)}, {repr(self._type)}, {repr(self.data) if not self.vector else repr(self.vector)})"

    @classmethod
    def load(cls, data: BytesIO) -> 'Parameter':
        logger.info("Loading parameter")
        param_hash, param_class, param_type, length = struct.unpack("<IIII", data.read(16))
        param_data = data.read(length)
        t = D3DXParamType(param_type)
        c = D3DXParamClass(param_class)
        logger.info(f"    type:  {t.name}")
        logger.info(f"    class: {c.name}")

        return cls(param_hash, c, t, param_data)

class Material:
    def __init__(self, namehash: int, definition: int, parameters: List[Parameter]):
        self.namehash = namehash
        self.definition = definition
        self.parameters = parameters
        self.__encoded_parameters = None
    
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

    @classmethod
    def load(cls, data: BytesIO) -> 'Material':
        logger.info("Loading material data")
        offset = 0
        namehash, length, definition, num_params = struct.unpack("<IIII", data.read(16))
        logger.info(f"Name hash: {hex(namehash)}")
        logger.info(f"Definition hash: {hex(definition)}")
        logger.info(f"Parameter count: {num_params}")
        offset += 16
        parameters = []
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
    def load(cls, data: BytesIO) -> 'DMAT':
        logger.info("Loading DMAT chunk")
        dmat_length = struct.unpack("<I", data.read(4))[0]
        offset = 0
        magic = data.read(4)
        offset += 4
        assert magic.decode("utf-8") == "DMAT", "Not a DMAT chunk"
        version, filename_length = struct.unpack("<II", data.read(8))
        offset += 8
        assert version == 1, f"Unknown DMAT version {version}"
        
        name_data = data.read(filename_length)
        texture_names = name_data.decode("utf-8").strip('\x00').split("\x00")
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

        return cls(magic, version, texture_names, materials)

class DME:
    def __init__(self, magic: str, version: int, dmat: DMAT, aabb: AABB, meshes: List[Mesh]):
        assert magic == "DMOD", "Not a DME file"
        assert version == 4, "Unsupported DME version"
        self.magic = magic
        self.version = version
        self.dmat = dmat
        self.aabb = aabb
        self.meshes = meshes

    @classmethod
    def load(cls, data: BytesIO) -> 'DME':
        logger.info("Loading DME file")
        #DMOD block
        magic = data.read(4)
        assert magic.decode("utf-8") == "DMOD", "Not a DME file"
        
        version = struct.unpack("<I", data.read(4))[0]
        assert version == 4, f"Unsupported DME version {version}"
        
        #DMAT block
        dmat = DMAT.load(data)
        material = dmat.materials[0]
        try:
            definition = MaterialDefinitions[material.namehash]
            draw_style = definition.draw_styles[0]
            input_layout = InputLayouts[jenkins.oaat(draw_style.input_layout.encode("utf-8"))]
        except KeyError:
            logger.warning(f"Material definition not found for name hash {hex(material.namehash)}! Defaulting to 'VehicleRigid_PS'")
            definition = MaterialDefinitions[59309762] #"VehicleRigid_PS"
            draw_style = definition.draw_styles[0]
            input_layout = InputLayouts[2340912194] #"Vehicle"
        
        
        #MESH block
        minx, miny, minz = struct.unpack("<fff", data.read(12))
        maxx, maxy, maxz = struct.unpack("<fff", data.read(12))
        num_meshes = struct.unpack("<I", data.read(4))[0]
        aabb = AABB([(minx, maxx), (miny, maxy), (minz, maxz)])
        meshes = [Mesh.load(data, input_layout) for _ in range(num_meshes)]
        
        return cls(magic.decode("utf-8"), version, dmat, aabb, meshes)