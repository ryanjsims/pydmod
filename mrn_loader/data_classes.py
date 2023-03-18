from io import BytesIO, SEEK_CUR, SEEK_END
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from scipy.spatial.transform import Rotation
from enum import IntEnum
from utils import read_cstr

import struct
import logging
import numpy
import sys

from utils import read_cstr

logger = logging.getLogger("MRN loader")

# Skeleton indices start with this value:
#   - first 8 bytes are padding to align the data to a 16 byte address
#   - next 4 bytes are unknown
#   - final 4 bytes indicate the first bone doesn't have a parent (assumption)
SKELETON_START_BYTES = b'\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xff\xff\xff\xff\xff\xff\xff\xff'
# 0xCDCDCDCD as an int32
PADDING_INT32 = -842150451

X64_MAGIC = 0x1A
X32_MAGIC = 0x18

def align(data: BytesIO, alignment: int):
    if data.tell() % alignment != 0:
            data.read(alignment - (data.tell() % alignment))

def sign(rot: Rotation):
    sign = 1.0
    for num in numpy.signbit(rot.as_quat()):
        sign *= -1.0 if num else 1.0
    return sign

@dataclass
class Bone:
    name: str
    index: int
    offset: numpy.ndarray
    rotation: Rotation
    children: List['Bone']
    reoriented: bool = False
    global_offset: numpy.ndarray = None

    def reorient(self):
        if self.reoriented:
            return
        self.reoriented = True
        for child in self.children:
            child.reorient()
        
            child.offset -= self.offset
            child.rotation *= self.rotation.inv()

            child.offset = sign(self.rotation) * self.rotation.inv().as_quat() * child.offset

    def __repr__(self) -> str:
        return f"Bone(name='{self.name}', offset={self.offset}, rotation={self.rotation.as_quat()}, children={self.children})"

@dataclass
class BoneHierarchyEntry:
    unknown: int
    parent: int
    chain_start: int
    chain_length: int

    @classmethod
    def load(cls, data: BytesIO) -> 'BoneHierarchyEntry':
        unknown, parent, chain_start, chain_length = struct.unpack("<iiii", data.read(16))
        return cls(unknown, parent, chain_start, chain_length)

@dataclass
class OrientationHeader:
    header_size: int
    unknown1: int
    total_length: int
    unknown2: int
    unknown3: int
    unknown_bool: bool
    unknown4: int
    unknown5: int
    unknown_bytes: bytes

    @classmethod
    def load(cls, data: BytesIO) -> 'OrientationHeader':
        header_size, unknown1, _, _, total_length, unknown2, unknown3 = struct.unpack("<IIIIIII", data.read(28))
        unknown_bool = bool(data.read(1)[0])
        data.read(3)
        unknown4, _, byteslength, unknown5 = struct.unpack("<IIQQ", data.read(24))
        unknown_bytes = data.read(byteslength)
        return cls(header_size, unknown1, total_length, unknown2, unknown3, unknown_bool, unknown4, unknown5, unknown_bytes)

@dataclass
class OrientationData:
    header: OrientationHeader
    offsets: List[Tuple[float, float, float, float]]
    rotations: List[Tuple[float, float, float, float]]

    @classmethod
    def load(cls, data: BytesIO, skeleton_length: int) -> 'OrientationData':
        header = OrientationHeader.load(data)
        if data.tell() % 16 != 0:
            data.read(16 - (data.tell() % 16))
        offsets = [struct.unpack("<ffff", data.read(16)) for _ in range(skeleton_length)]
        if (len(offsets) * 16) % 64 != 0:
            data.read(64 - ((len(offsets) * 16) % 64))
        rotations = [struct.unpack("<ffff", data.read(16)) for _ in range(skeleton_length)]
        if (len(rotations) * 16) % 64 != 0:
            data.read(64 - ((len(rotations) * 16) % 64))
        return cls(header, offsets, rotations)

@dataclass
class Skeleton:
    rotation: Tuple[float, float, float, float]
    position: Tuple[float, float, float, float]
    chain_count: int
    unknown_array1: List[int]
    unknown_array2: List[int]
    unknown_array3: List[int]
    hierarchy: List[BoneHierarchyEntry]
    bone_count: int
    unknown_array4: List[int]
    indices: List[int]
    variable_length_unknowns: List[int]
    constant_length_unknowns: bytes
    bones: List[Bone]
    transforms: OrientationData

    @classmethod
    def load(cls, data: BytesIO) -> 'Skeleton':
        rotation = struct.unpack("<ffff", data.read(16))
        position = struct.unpack("<ffff", data.read(16))
        chain_count = struct.unpack("<I", data.read(4))[0]
        data.read(4)
        unknown_array1 = struct.unpack("<IIIIIIIIIII", data.read(44))
        data.read(4)
        unknown_array2 = struct.unpack("<III", data.read(12))
        data.read(4)
        unknown_array3 = struct.unpack("<IIII", data.read(16))
        data.read(8)

        hierarchy: List[BoneHierarchyEntry] = [BoneHierarchyEntry.load(data) for _ in range(chain_count)]
        skeleton_length = struct.unpack("<I", data.read(4))[0]
        data.read(4)
        unknown_array4 = struct.unpack("<III", data.read(12))
        indices = [struct.unpack("<i", data.read(4))[0] for _ in range(skeleton_length)]
        variable_length_unknowns = [struct.unpack("<i", data.read(4))[0] for _ in range(skeleton_length)]
        constant_length_unknowns = data.read(20)
        bone_names = [read_cstr(data) for _ in range(skeleton_length)]
        read_data = data.read(1)
        while read_data == b'\xCD':
            read_data = data.read(1)
        read_data += data.read(3)
        assert read_data == b'\xFF\xFF\x0E\x00', f"Offset: {data.tell()}\nData: {read_data}"
        data.read(4) # More 0xCD pad bytes

        transforms = OrientationData.load(data, skeleton_length)
        
        bones = [Bone(bone_names[i], i, numpy.array(transforms.offsets[i], dtype=numpy.float32), Rotation.from_quat(transforms.rotations[i]), []) for i in range(len(bone_names))]
        logger.debug(f"Loaded skeleton {bones[1].name}")
        return cls(rotation, position, chain_count, unknown_array1, unknown_array2, unknown_array3, hierarchy, skeleton_length, unknown_array4, indices, variable_length_unknowns, constant_length_unknowns, bones, transforms)

    def build_recursive(self):
        for entry in self.hierarchy:
            parent = entry.parent
            for i in range(entry.chain_start, entry.chain_start + entry.chain_length):
                if parent != -1:
                    self.bones[parent].children.append(self.bones[i])
                parent = i
    
    def calc_global_offsets(self):
        if len(self.bones[0].children) == 0:
            self.build_recursive()
        self.__calc_offsets(self.bones[0])

    def __calc_offsets(self, root: Bone):
        if root.global_offset is None:
            root.global_offset = root.offset[:3]
        for child in root.children:
            child.global_offset = root.global_offset + child.offset[:3]
            self.__calc_offsets(child)

    def pretty_print(self):
        if len(self.bones[0].children) == 0:
            self.build_recursive()
        self.__print_recursive(self.bones[0], 0)
    
    def __print_recursive(self, root: Bone, depth: int):
        print("    " * depth + root.name)
        for bone in root.children:
            self.__print_recursive(bone, depth + 1)
    
    def name(self) -> str:
        return self.bones[1].name

@dataclass
class Factors:
    q_min: Tuple[float, float, float]
    q_extent: Tuple[float, float, float]

@dataclass
class DeqFactors:
    translation: List[Factors]
    rotation: List[Factors]
    scale: List[Factors]

    def __getitem__(self, index: int):
        if index == 0:
            return self.translation
        elif index == 1:
            return self.rotation
        elif index == 2:
            return self.scale
        raise IndexError
    
    def __setitem__(self, index: int, value: List[Factors]):
        if index == 0:
            self.translation = value
        elif index == 1:
            self.rotation = value
        elif index == 2:
            self.scale = value
        else:
            logger.error(f"Index was {index}")
            raise IndexError
    
    @classmethod
    def load(cls, data: BytesIO, trs_counts: Tuple[int, int, int]) -> 'DeqFactors':
        translation = [Factors(floats[:3], floats[3:]) for floats in struct.iter_unpack("<ffffff", data.read(24 * trs_counts[0]))]
        rotation = [Factors(floats[:3], floats[3:]) for floats in struct.iter_unpack("<ffffff", data.read(24 * trs_counts[1]))]
        scale = [Factors(floats[:3], floats[3:]) for floats in struct.iter_unpack("<ffffff", data.read(24 * trs_counts[2]))]
        return cls(translation, rotation, scale)



@dataclass
class AnimationBoneIndices:
    translation: List[int]
    rotation: List[int]
    scale: List[int]

@dataclass
class AnimationFirstSegment:
    trs_counts: Tuple[int, int, int]
    trs_factors: DeqFactors
    unknown: int
    trs_data: Tuple[bytes, bytes, bytes]
    size: int

    @classmethod
    def load(cls, data: BytesIO, second_segment_offset: int):
        base = data.tell()
        trs_counts = struct.unpack("<III", data.read(12))
        trs_factors = DeqFactors.load(data, (1 if trs_counts[0] > 0 else 0, 1 if trs_counts[1] > 0 else 0, 1 if trs_counts[2] > 0 else 0))
        data.read(4)
        unknown = struct.unpack("<Q", data.read(8))[0]
        trs_offsets = struct.unpack("<QQQ", data.read(24)) + (second_segment_offset - base,)
        logger.debug(f"{trs_offsets=}")
        trs_data = []
        for i in range(len(trs_offsets)):
            if (i + 1) >= len(trs_offsets):
                break
            if trs_offsets[i] == 0:
                trs_data.append(b'')
                continue
            end = 0
            offset = i + 1
            while end == 0 and offset < len(trs_offsets):
                end = trs_offsets[offset]
                offset += 1
            assert end > 0
            data.seek(base + trs_offsets[i])
            try:
                trs_data.append(data.read(6 * trs_counts[i]))
            except MemoryError:
                logger.error(f"Tried to read {6 * trs_counts[i]} bytes at offset {base + trs_offsets[i]}")
                sys.exit(1)
        return cls(trs_counts, trs_factors, unknown, trs_data, second_segment_offset - base)

@dataclass
class InitFactorIndices:
    init_values: Tuple[int, int, int]
    dequantization_factor_indices: Tuple[int, int, int]

    @classmethod
    def load(cls, data: BytesIO) -> 'InitFactorIndices':
        init_values = struct.unpack("<BBB", data.read(3))
        dequantization_factor_indices = struct.unpack("<BBB", data.read(3))
        return cls(init_values, dequantization_factor_indices)

def next_multiple_of_four(bone_count: int) -> int:
    return 4 * ((bone_count // 4) + (1 if (bone_count % 4 != 0) else 0))

def unpack_pos(packed_pos: Tuple[int, int, int], init_factor_indices: InitFactorIndices, pos_factors: List[Factors]) -> Tuple[float, float, float]:
    PRECISION_XY = 2048.0
    PRECISION_Z  = 1024.0
    x_factors = pos_factors[init_factor_indices.dequantization_factor_indices[0]]
    x_quant_factor = x_factors.q_extent[0]
    x_quant_min = x_factors.q_min[0]

    y_factors = pos_factors[init_factor_indices.dequantization_factor_indices[1]]
    y_quant_factor = y_factors.q_extent[1]
    y_quant_min = y_factors.q_min[1]

    z_factors = pos_factors[init_factor_indices.dequantization_factor_indices[2]]
    z_quant_factor = z_factors.q_extent[2]
    z_quant_min = z_factors.q_min[2]

    dequant_x = x_quant_factor * (packed_pos[0] + (init_factor_indices.init_values[0] / 256.0) * PRECISION_XY) + x_quant_min 
    dequant_y = y_quant_factor * (packed_pos[1] + (init_factor_indices.init_values[1] / 256.0) * PRECISION_XY) + y_quant_min
    dequant_z = z_quant_factor * (packed_pos[2] + (init_factor_indices.init_values[2] / 256.0) * PRECISION_Z) + z_quant_min
    return dequant_x, dequant_y, dequant_z

def unpack_rotation(rot_quant: Tuple[int, int, int, int], init_factor_indices: InitFactorIndices, rot_factors: List[Factors]) -> Rotation:
    PRECISION = 65536.0
    x_factors = rot_factors[init_factor_indices.dequantization_factor_indices[0]]
    x_quant_factor = x_factors.q_extent[0]
    x_quant_min = x_factors.q_min[0]

    y_factors = rot_factors[init_factor_indices.dequantization_factor_indices[1]]
    y_quant_factor = y_factors.q_extent[1]
    y_quant_min = y_factors.q_min[1]

    z_factors = rot_factors[init_factor_indices.dequantization_factor_indices[2]]
    z_quant_factor = z_factors.q_extent[2]
    z_quant_min = z_factors.q_min[2]

    dequant_x = x_quant_factor * (rot_quant[0] + (init_factor_indices.init_values[0] / 256.0) * PRECISION) + x_quant_min 
    dequant_y = y_quant_factor * (rot_quant[1] + (init_factor_indices.init_values[1] / 256.0) * PRECISION) + y_quant_min
    dequant_z = z_quant_factor * (rot_quant[2] + (init_factor_indices.init_values[2] / 256.0) * PRECISION) + z_quant_min

    vec_squared = dequant_x * dequant_x + dequant_y * dequant_y + dequant_z * dequant_z
    dequant_w = vec_squared + 1.0

    temp = 2.0 / dequant_w
    output_w = dequant_w / (1.0 - vec_squared)
    output_x = temp * dequant_x
    output_y = temp * dequant_y
    output_z = temp * dequant_z

    return Rotation.from_quat([output_x, output_y, output_z, output_w])


@dataclass
class AnimationSecondSegment:
    sample_count: int
    trs_counts: Tuple[int, int, int]
    trs_data: Tuple[bytes, bytes, bytes]
    trs_factor_indices: Tuple[List[InitFactorIndices], List[InitFactorIndices], List[InitFactorIndices]]
    size: int
    translation: numpy.ndarray = None
    rotation: numpy.ndarray = None
    scale: List[List[numpy.ndarray]] = None

    def dequantize(self, factors: DeqFactors):
        if self.trs_counts[0] > 0:
            translation: List[List[Tuple[float, float, float]]] = []
            translation_bone_count = self.trs_counts[0]
            for sample in range(self.sample_count):
                translation.append([])
                for bone in range(translation_bone_count):
                    data_offset = sample * translation_bone_count * 4 + bone * 4
                    value = struct.unpack("<I", self.trs_data[0][data_offset:data_offset+4])[0]
                    # 11 bits for x and y, 10 bits for z
                    pos_quant = ((value >> 21) & 0x7FF, (value >> 10) & 0x7FF, value & 0x3FF)

                    translation[sample].append(unpack_pos(pos_quant, self.trs_factor_indices[0][bone], factors.translation))
            self.translation = numpy.array(translation, dtype=numpy.float32)
        
        if self.trs_counts[1] > 0:
            rotation_bone_count = self.trs_counts[1]
            shorts_per_rotation = len(self.trs_data[1]) // (self.sample_count * rotation_bone_count * 2)
            # print(shorts_per_rotation)
            # shorts_per_rotation: int = 4 if len(self.trs_data[1]) % 8 == 0 and len(self.trs_data[1]) % 6 != 0 else 3
            rotation: List[List[Tuple[float, float, float, float]]] = []
            for sample in range(self.sample_count):
                rotation.append([])
                for bone in range(rotation_bone_count):
                    data_offset = sample * rotation_bone_count * 2 * shorts_per_rotation + bone * 2 * shorts_per_rotation
                    rot_quant = struct.unpack("<" + "H" * shorts_per_rotation, self.trs_data[1][data_offset : data_offset + 2 * shorts_per_rotation])

                    rotation[sample].append(unpack_rotation(rot_quant, self.trs_factor_indices[1][bone], factors.rotation).as_quat().tolist())
            self.rotation = numpy.array(rotation, dtype=numpy.float32)


    @classmethod
    def load(cls, data: BytesIO, end_data_offset: int):
        base = data.tell()
        frame_count = struct.unpack("<I", data.read(4))[0]
        trs_counts = struct.unpack("<III", data.read(12))
        trs_data_factors_offsets = struct.unpack("<QQQQQQ", data.read(48))
        trs_data = []
        trs_factor_indices = ([], [], [])
        for i in range(0, len(trs_data_factors_offsets), 2):
            if trs_data_factors_offsets[i] == 0:
                trs_data.append(b'')
                continue
            if i + 2 > len(trs_data_factors_offsets):
                end = end_data_offset - base
            else:
                offset = i + 2
                end = 0
                while end == 0:
                    if offset < len(trs_data_factors_offsets):
                        end = trs_data_factors_offsets[offset]
                    else:
                        end = end_data_offset - base
                    offset += 2
            data.seek(base + trs_data_factors_offsets[i])
            trs_data.append(data.read(trs_data_factors_offsets[i + 1] - trs_data_factors_offsets[i]))
            
            length = next_multiple_of_four(trs_counts[len(trs_data) - 1])
            logger.debug(f"Loading {length} init factor indices")
            for _ in range(length):
                trs_factor_indices[len(trs_data) - 1].append(InitFactorIndices.load(data))
        
        return cls(frame_count, trs_counts, trs_data, trs_factor_indices, end_data_offset - base)

@dataclass
class Animation:
    crc32hash: int
    version: int # Not actually sure if this is a version or not
    static_length: int
    alignment: int
    duration: float
    framerate: float
    unknown3: int
    unknown4: int
    static_bones: AnimationBoneIndices
    dynamic_bones: AnimationBoneIndices
    translation_init_factors: Tuple[float, float, float, float, float, float]
    trs_anim_deq_counts: Tuple[int, int, int]
    trs_anim_deq_factors: DeqFactors

    static_data: AnimationFirstSegment
    dynamic_data: AnimationSecondSegment
    end_data: bytes
    size: int
    
    @classmethod
    def load_bytes_offset(cls, data: BytesIO, base: int, offset: int, count: int) -> bytes:
        if offset == 0:
            return None
        data.seek(base + offset)
        return data.read(count)

    @classmethod
    def load(cls, data: BytesIO) -> 'Animation':
        base = data.tell()
        crc32hash = struct.unpack(">I", data.read(4))[0]
        version, _, unknown1, unknown2, unknown_float1, unknown_float2, unknown3, unknown4 = struct.unpack("<IQIIffII", data.read(36))
        static_index_offsets = struct.unpack("<QQQ", data.read(24))
        dynamic_index_offsets = struct.unpack("<QQQ", data.read(24))
        translation_init_factors = struct.unpack("<ffffff", data.read(24))
        data.read(8)
        trs_anim_deq_counts = struct.unpack("<III", data.read(12))
        data.read(4)
        trs_deq_factor_offsets = struct.unpack("<QQQ", data.read(24))
        first_segment_offset, second_segment_offset, end_data_offset = struct.unpack("<QQQ", data.read(24))

        length_bytes = Animation.load_bytes_offset(data, base, static_index_offsets[0], 2)
        if(length_bytes is not None):
            length = struct.unpack("<H", length_bytes)[0]
        else:
            length = 0
        static_translation_bone_indices = [i[0] for i in struct.iter_unpack("<H", data.read(2 * length))]
        
        length_bytes = Animation.load_bytes_offset(data, base, static_index_offsets[1], 2)
        if(length_bytes is not None):
            length = struct.unpack("<H", length_bytes)[0]
        else:
            length = 0
        static_rotation_bone_indices = [i[0] for i in struct.iter_unpack("<H", data.read(2 * length))]

        length_bytes = Animation.load_bytes_offset(data, base, static_index_offsets[2], 2)
        if(length_bytes is not None):
            length = struct.unpack("<H", length_bytes)[0]
        else:
            length = 0
        static_scale_bone_indices = [i[0] for i in struct.iter_unpack("<H", data.read(2 * length))]

        static_bones = AnimationBoneIndices(static_translation_bone_indices, static_rotation_bone_indices, static_scale_bone_indices)

        length_bytes = Animation.load_bytes_offset(data, base, dynamic_index_offsets[0], 2)
        if(length_bytes is not None):
            length = struct.unpack("<H", length_bytes)[0]
        else:
            length = 0
        dynamic_translation_bone_indices = [i[0] for i in struct.iter_unpack("<H", data.read(2 * length))]
        
        length_bytes = Animation.load_bytes_offset(data, base, dynamic_index_offsets[1], 2)
        if(length_bytes is not None):
            length = struct.unpack("<H", length_bytes)[0]
        else:
            length = 0
        dynamic_rotation_bone_indices = [i[0] for i in struct.iter_unpack("<H", data.read(2 * length))]

        length_bytes = Animation.load_bytes_offset(data, base, dynamic_index_offsets[2], 2)
        if(length_bytes is not None):
            length = struct.unpack("<H", length_bytes)[0]
        else:
            length = 0
        dynamic_scale_bone_indices = [i[0] for i in struct.iter_unpack("<H", data.read(2 * length))]

        dynamic_bones = AnimationBoneIndices(dynamic_translation_bone_indices, dynamic_rotation_bone_indices, dynamic_scale_bone_indices)

        trs_anim_deq_factors = DeqFactors([], [], [])
        for i in range(len(trs_deq_factor_offsets)):
            offset = trs_deq_factor_offsets[i]
            data.seek(base + offset) 
            trs_anim_deq_factors[i] = [Factors(floats[:3], floats[3:]) for floats in struct.iter_unpack("<ffffff", data.read(24 * trs_anim_deq_counts[i]))]
        
        if first_segment_offset != 0:
            data.seek(base + first_segment_offset)
            logger.debug(f"Loading static data from offset {first_segment_offset:08x}")
            logger.debug(f"{second_segment_offset=:08x}")
            end_offset = second_segment_offset if second_segment_offset > 0 else end_data_offset
            logger.debug(f"{end_offset=:08x}")
            static_data = AnimationFirstSegment.load(data, base + end_offset)
        else:
            static_data = None

        if second_segment_offset != 0:
            data.seek(base + second_segment_offset)
            logger.debug(f"Loading dynamic data from offset {second_segment_offset:08x}")
            logger.debug(f"{end_data_offset=:08x}")
            dynamic_data = AnimationSecondSegment.load(data, base + end_data_offset)
        else:
            dynamic_data = None

        data.seek(base + end_data_offset)
        end_data = data.read(96)

        return cls(crc32hash, version, unknown1, unknown2, unknown_float1, unknown_float2, unknown3, unknown4, static_bones, dynamic_bones, translation_init_factors, trs_anim_deq_counts, trs_anim_deq_factors, static_data, dynamic_data, end_data, data.tell() - base)


"""
struct string_table {
    u32 string_count;
    u32 string_data_size;
    u32* string_offsets[string_count] : u64 [[pointer_base("std::ptr::relative_to_parent")]];
    string* strings[string_count] : u64 [[pointer_base("std::ptr::relative_to_parent")]];
    padding[string_count * sizeof(u32) + string_data_size];
    padding[while(std::mem::read_unsigned($, 1) == 0xCD)];
};
"""

@dataclass
class StringTable:
    string_offsets: List[int]
    strings: List[str]

    def __getitem__(self, index: int) -> str:
        return self.strings[index]
    
    def __len__(self) -> int:
        return len(self.strings)
    
    def __iter__(self):
        return iter(self.strings)
    
    def __contains__(self, string: str):
        return string in self.strings

    @classmethod
    def load(cls, data: BytesIO) -> 'StringTable':
        base = data.tell()
        string_count, string_data_length, string_offsets_ptr, strings_ptr = struct.unpack("<IIQQ", data.read(24))
        assert (base + string_offsets_ptr) == data.tell()
        string_offsets = [value[0] for value in struct.iter_unpack("<I", data.read(4 * string_count))]
        assert (base + strings_ptr) == data.tell()
        strings = [read_cstr(data) for _ in range(string_count)]
        assert data.tell() - (base + strings_ptr) == string_data_length, f"read_string_length={data.tell() - (base + strings_ptr)} != {string_data_length=}"
        while data.read(1) == b'\xCD':
            pass
        data.seek(-1, SEEK_CUR)
        return cls(string_offsets, strings)

@dataclass
class ExtStringTable:
    string_ids: List[int]
    string_offsets: List[int]
    unknown_ints: List[int]
    strings: List[str]

    def __getitem__(self, index: int) -> str:
        return self.strings[index]
    
    def __len__(self) -> int:
        return len(self.strings)
    
    def __iter__(self):
        return iter(self.strings)
    
    def __contains__(self, string: str):
        return string in self.strings

    @classmethod
    def load(cls, data: BytesIO) -> 'ExtStringTable':
        base = data.tell()
        string_count, data_size = struct.unpack("<II", data.read(8))
        string_ids_ptr, string_offsets_ptr, unknown_ints_ptr, strings_ptr = struct.unpack("<QQQQ", data.read(32))
        if base + string_ids_ptr != data.tell():
            data.seek(base + string_ids_ptr)
        string_ids = [value[0] for value in struct.iter_unpack("<I", data.read(4 * string_count))]
        
        if base + string_offsets_ptr != data.tell():
            data.seek(base + string_offsets_ptr)
        string_offsets = [value[0] for value in struct.iter_unpack("<I", data.read(4 * string_count))]

        if base + unknown_ints_ptr != data.tell():
            data.seek(base + unknown_ints_ptr)
        unknown_ints = [value[0] for value in struct.iter_unpack("<I", data.read(4 * string_count))]

        if base + strings_ptr != data.tell():
            data.seek(base + strings_ptr)
        strings = [read_cstr(data) for _ in range(string_count)]
        assert data.tell() - (base + strings_ptr) == data_size
        return cls(string_ids, string_offsets, unknown_ints, strings)

@dataclass
class Filenames:
    filenames: StringTable
    filetypes: StringTable
    source_filenames: StringTable
    animation_names: StringTable
    crc32hashes: List[int]

    @classmethod
    def load(cls, data: BytesIO):
        base = data.tell()
        filenames_ptr, filetypes_ptr, source_filenames_ptr, animation_names_ptr, crc32hashes_ptr = struct.unpack("<QQQQQ", data.read(40))
        if base + filenames_ptr != data.tell():
            data.seek(base + filenames_ptr)
        filenames = StringTable.load(data)
        if base + filetypes_ptr != data.tell():
            data.seek(base + filetypes_ptr)
        filetypes = StringTable.load(data)
        if base + source_filenames_ptr != data.tell():
            data.seek(base + source_filenames_ptr)
        source_filenames = StringTable.load(data)
        if base + animation_names_ptr != data.tell():
            data.seek(base + animation_names_ptr)
        animation_filenames = StringTable.load(data)

        if base + crc32hashes_ptr != data.tell():
            data.seek(base + crc32hashes_ptr)
        crc32hashes = [value[0] for value in struct.iter_unpack(">I", data.read(4 * len(filenames.strings)))]
        return cls(filenames, filetypes, source_filenames, animation_filenames, crc32hashes)

@dataclass
class SkeletonNames:
    skeletons: ExtStringTable

    def __getitem__(self, index: int) -> str:
        return self.skeletons[index]

    def __len__(self) -> int:
        return len(self.skeletons)
    
    def __iter__(self):
        return iter(self.skeletons)
    
    def __contains__(self, skeleton: str):
        return skeleton in self.skeletons
    
    def index_of(self, skeleton: str) -> int:
        if skeleton not in self.skeletons:
            return -1
        return self.skeletons.strings.index(skeleton)

    @classmethod
    def load(cls, data: BytesIO) -> 'SkeletonNames':
        base = data.tell()
        skeletons_ptr = struct.unpack("<Q", data.read(8))[0]
        assert base + skeletons_ptr == data.tell()
        skeletons = ExtStringTable.load(data)
        return cls(skeletons)

class PacketType(IntEnum):
    skeleton = 0x01
    some_other_data = 0x02
    name2 = 0x03
    unknown = 0x04
    data = 0x07
    maybe_animation_data = 0x0A   # not sure but it has some names in it that sound like it
    name = 0x0C
    file_names = 0x0E       # same as above - has some file name data in it alongside other stuff.
    skeleton_names = 0x0F
    animation_data = 0x10     # Data that has an index rather than namehash in the header


@dataclass
class Header:
    magic: Tuple[int, int]
    packet_type: PacketType
    index_or_namehash: int
    unknown: Tuple[int, int, int, int]
    length: int
    alignment: int

    @classmethod
    def load(cls, data: BytesIO) -> 'Header':
        logger.debug("Loading header...")
        magic = struct.unpack("<II", data.read(8))
        if magic[0] == 0x18:
            raise ValueError("Attempting to load 32bit MRN")
        packet_type = PacketType(struct.unpack("<I", data.read(4))[0])
        index_or_namehash = struct.unpack("<I", data.read(4))[0]
        unknown = struct.unpack("<IIII", data.read(16))
        length, alignment = struct.unpack("<II", data.read(8))
        align(data, alignment)
        return cls(magic, packet_type, index_or_namehash, unknown, length, alignment)

@dataclass
class Packet:
    header: Header
    data: bytes

    @classmethod
    def load(cls, data: BytesIO) -> 'Packet':
        header = Header.load(data)
        data_start = data.tell()
        extra = None
        if header.packet_type == PacketType.skeleton:
            skeleton = Skeleton.load(data)
        elif header.packet_type == PacketType.animation_data:
            animation = Animation.load(data)
        elif header.packet_type == PacketType.file_names:
            files = Filenames.load(data)
        elif header.packet_type == PacketType.skeleton_names:
            skeleton_names = SkeletonNames.load(data)
        else:
            packet_data = data.read(header.length)
        if data.tell() - data_start < header.length:
            extra = data.read(header.length - (data.tell() - data_start))
        Packet.align(data, header)
        if header.packet_type == PacketType.skeleton:
            return SkeletonPacket(header, skeleton, extra)
        if header.packet_type == PacketType.animation_data:
            return AnimationPacket(header, animation, extra)
        if header.packet_type == PacketType.file_names:
            return FilenamePacket(header, files, extra)
        if header.packet_type == PacketType.skeleton_names:
            return SkeletonNamesPacket(header, skeleton_names, extra)
        return cls(header, packet_data)
    
    @classmethod
    def align(cls, data: BytesIO, header: Header):
        if header.magic[0] == X64_MAGIC:
            packet_alignment = 16
        elif header.magic[0] == X32_MAGIC:
            packet_alignment = 4
        else:
            assert False, "Unknown MRN packet magic!"
        align(data, packet_alignment)

@dataclass
class SkeletonPacket:
    header: Header
    skeleton: Skeleton
    extra: bytes

@dataclass
class AnimationPacket:
    header: Header
    animation: Animation
    extra: bytes

"""
struct file_data {
    string_table* filenames : u64 [[pointer_base("std::ptr::relative_to_parent")]];
    string_table* filetypes : u64 [[pointer_base("std::ptr::relative_to_parent")]];
    string_table* source_filenames : u64 [[pointer_base("std::ptr::relative_to_parent")]];
    string_table* animation_names : u64 [[pointer_base("std::ptr::relative_to_parent")]];
    // Turns out imhex does not allow you to dereference pointers to get values.
    // The count here should be filenames->string_count but unfortunately that is not available in imhex
    be u32* crc32hashes[std::mem::read_unsigned(addressof(this) + 40, 4)] : u64 [[pointer_base("std::ptr::relative_to_parent")]];
};
"""

@dataclass
class FilenamePacket:
    header: Header
    files: Filenames
    extra: bytes

@dataclass
class SkeletonNamesPacket:
    header: Header
    skeletons: SkeletonNames
    extra: bytes

@dataclass
class MRN:
    #skeletons: List[Skeleton]
    packets: List[Packet]
    _filenames_index: int
    _skeleton_names_index: int
    _skeleton_packets_indices: List[int]
    _skeleton_packets: List[SkeletonPacket] = None

    def filenames_packet(self) -> Optional[FilenamePacket]:
        if self._filenames_index == -1:
            return None
        return self.packets[self._filenames_index]
    
    def skeleton_names_packet(self) -> Optional[SkeletonNamesPacket]:
        if self._skeleton_names_index == -1:
            return None
        return self.packets[self._skeleton_names_index]

    def skeleton_packets(self) -> List[SkeletonPacket]:
        if self._skeleton_packets is None:
            self._skeleton_packets = [self.packets[index] for index in self._skeleton_packets_indices]
        return self._skeleton_packets

    @classmethod
    def load(cls, data: BytesIO) -> 'MRN':
        logger.info("Loading MRN file...")
        # skeletons = []
        # skeleton = Skeleton.load(data)
        # while skeleton is not None:
        #     skeletons.append(skeleton)
        #     skeleton = Skeleton.load(data)
        # logger.info(f"Loaded {len(skeletons)} skeletons!")

        filenames_index = -1
        skeleton_names_index = -1
        skeleton_indices = []
        packets = []
        data.seek(0, SEEK_END)
        length = data.tell()
        data.seek(0)
        while data.tell() < length:
            logger.debug(f"Loading packet {len(packets) + 1} at offset {data.tell():08x}...")
            packet = Packet.load(data)
            if packet.header.packet_type == PacketType.file_names:
                filenames_index = len(packets)
            elif packet.header.packet_type == PacketType.skeleton_names:
                skeleton_names_index = len(packets)
            elif packet.header.packet_type == PacketType.skeleton:
                skeleton_indices.append(len(packets))
            packets.append(packet)
            logger.debug(f"Loaded packet {len(packets)}")
        logger.info(f"Loaded {len(packets)} MRN packets!")
        return cls(packets, filenames_index, skeleton_names_index, skeleton_indices)