from io import BytesIO, SEEK_CUR, SEEK_END
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from scipy.spatial.transform import Rotation
from enum import IntEnum

import struct
import logging
import numpy

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
        logger.info(f"Loaded skeleton {bones[1].name}")
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


class PacketType(IntEnum):
    skeleton = 0x01
    some_other_data = 0x02
    name2 = 0x03
    data = 0x07
    animation_data = 0x0A   # not sure but it has some names in it that sound like it
    name = 0x0C
    file_names = 0x0E       # same as above - has some file name data in it alongside other stuff.
    vehicle_names = 0x0F
    indexed_data = 0x10     # Data that has an index rather than namehash in the header


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
        logger.info("Loading header...")
        magic = struct.unpack("<II", data.read(8))
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
        if header.packet_type == PacketType.skeleton:
            skeleton = Skeleton.load(data)
        else:
            packet_data = data.read(header.length)
        if data.tell() - data_start < header.length:
            extra = data.read(header.length - (data.tell() - data_start))
        Packet.align(data, header)
        if header.packet_type == PacketType.skeleton:
            return SkeletonPacket(header, skeleton, extra)
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
            packets.append(Packet.load(data))
        return cls(packets)