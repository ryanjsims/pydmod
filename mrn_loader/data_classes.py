from io import BytesIO, SEEK_CUR, SEEK_END
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from scipy.spatial.transform import Rotation
from enum import IntEnum

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

@dataclass
class Bone:
    name: str
    offset: numpy.ndarray
    rotation: Rotation
    children: List['Bone']

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
        
        bones = [Bone(bone_names[i], numpy.array(transforms.offsets[i], dtype=numpy.float32), Rotation.from_quat(transforms.rotations[i]), []) for i in range(len(bone_names))]
        logger.debug(f"Loaded skeleton {bones[1].name}")
        return cls(rotation, position, chain_count, unknown_array1, unknown_array2, unknown_array3, hierarchy, skeleton_length, unknown_array4, indices, variable_length_unknowns, constant_length_unknowns, bones, transforms)

    def build_recursive(self):
        for entry in self.hierarchy:
            parent = entry.parent
            for i in range(entry.chain_start, entry.chain_start + entry.chain_length):
                if parent != -1:
                    self.bones[parent].children.append(self.bones[i])
                parent = i

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
    q_max: Tuple[float, float, float]

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

@dataclass
class AnimationSecondSegment:
    frame_count: int
    trs_counts: Tuple[int, int, int]
    trs_data: Tuple[bytes, bytes, bytes]
    trs_factor_indices: Tuple[List[InitFactorIndices], List[InitFactorIndices], List[InitFactorIndices]]
    size: int

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
            parse = "<hh"
            if i // 2 == 1:
                parse = "<hhh"
            trs_data.append(data.read(trs_data_factors_offsets[i + 1] - trs_data_factors_offsets[i]))
            length = end - trs_data_factors_offsets[i + 1]
            length = length - length % 6
            logger.debug(f"Loading factors with {length=}")
            for _ in range(0, length, 6):
                trs_factor_indices[len(trs_data) - 1].append(InitFactorIndices.load(data))
        
        return cls(frame_count, trs_counts, trs_data, trs_factor_indices, end_data_offset - base)

@dataclass
class Animation:
    crc32hash: int
    version: int # Not actually sure if this is a version or not
    unknown1: int
    unknown2: int
    unknown_float1: float
    unknown_float2: float
    unknown3: int
    unknown4: int
    static_bones: AnimationBoneIndices
    dynamic_bones: AnimationBoneIndices
    translation_init_factors: Tuple[float, float, float, float, float, float]
    trs_anim_deq_counts: Tuple[int, int, int]
    trs_anim_dec_factors: DeqFactors

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

        trs_anim_dec_factors = DeqFactors([], [], [])
        for i in range(len(trs_deq_factor_offsets)):
            offset = trs_deq_factor_offsets[i]
            data.seek(base + offset) 
            trs_anim_dec_factors[i] = [Factors(floats[:3], floats[3:]) for floats in struct.iter_unpack("<ffffff", data.read(24 * trs_anim_deq_counts[i]))]
        
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

        return cls(crc32hash, version, unknown1, unknown2, unknown_float1, unknown_float2, unknown3, unknown4, static_bones, dynamic_bones, translation_init_factors, trs_anim_deq_counts, trs_anim_dec_factors, static_data, dynamic_data, end_data, data.tell() - base)


class PacketType(IntEnum):
    skeleton = 0x01
    some_other_data = 0x02
    name2 = 0x03
    unknown = 0x04
    data = 0x07
    maybe_animation_data = 0x0A   # not sure but it has some names in it that sound like it
    name = 0x0C
    file_names = 0x0E       # same as above - has some file name data in it alongside other stuff.
    vehicle_names = 0x0F
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
        else:
            packet_data = data.read(header.length)
        if data.tell() - data_start < header.length:
            extra = data.read(header.length - (data.tell() - data_start))
        Packet.align(data, header)
        if header.packet_type == PacketType.skeleton:
            return SkeletonPacket(header, skeleton, extra)
        if header.packet_type == PacketType.animation_data:
            return AnimationPacket(header, animation, extra)
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

@dataclass
class MRN:
    #skeletons: List[Skeleton]
    packets: List[Packet]

    @classmethod
    def load(cls, data: BytesIO) -> 'MRN':
        logger.info("Loading MRN file...")
        # skeletons = []
        # skeleton = Skeleton.load(data)
        # while skeleton is not None:
        #     skeletons.append(skeleton)
        #     skeleton = Skeleton.load(data)
        # logger.info(f"Loaded {len(skeletons)} skeletons!")

        packets = []
        data.seek(0, SEEK_END)
        length = data.tell()
        data.seek(0)
        while data.tell() < length:
            logger.debug(f"Loading packet {len(packets) + 1} at offset {data.tell():08x}...")
            packets.append(Packet.load(data))
            logger.debug(f"Loaded packet {len(packets)}")
        logger.info(f"Loaded {len(packets)} MRN packets!")
        return cls(packets)