from io import BytesIO, SEEK_CUR
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from scipy.spatial.transform import Rotation

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
    parent: int
    chain_start: int
    chain_length: int
    ###
    # total_bones contains the number of bones in the skeleton 
    #   *only* on the *last* BoneHierarchyEntry object. Other
    #   entries' total_bones meanings are unknown at this time
    total_bones: int

    @classmethod
    def load(cls, data: BytesIO) -> 'BoneHierarchyEntry':
        parent, chain_start, chain_length, total_bones = struct.unpack("<iiii", data.read(16))
        return cls(parent, chain_start, chain_length, total_bones)

@dataclass
class Skeleton:
    hierarchy: List[BoneHierarchyEntry]
    indices: List[int]
    variable_length_unknowns: List[int]
    constant_length_unknowns: bytes
    bones: List[Bone]
    offsets: List[Tuple[float, float, float, float]]
    rotations: List[Tuple[float, float, float, float]]

    @classmethod
    def load(cls, data: BytesIO) -> 'Skeleton':
        alignment = data.tell() % 16
        if alignment != 8:
            # Skeleton data is aligned to an 8 byte address, so realign before searching
            data.read((24 if alignment > 8 else 8) - alignment)
        read_data = data.read(16)
        while len(read_data) == 16 and read_data != SKELETON_START_BYTES:
            # Skip ALL the unknown animation data that's probably super useful, if we knew how to parse it
            read_data = data.read(16)
        if len(read_data) < 16:
            return None
        data.seek(-4, SEEK_CUR)
        hierarchy: List[BoneHierarchyEntry] = []
        entry = BoneHierarchyEntry.load(data)
        while entry.parent != PADDING_INT32:
            hierarchy.append(entry)
            entry = BoneHierarchyEntry.load(data)
        skeleton_length = hierarchy[-1].total_bones
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

        # This is probably a header of some kind for the transform data, but that needs more research
        read_data = data.read(128)
        if read_data[-4:] != b'\xcd\xcd\xcd\xcd':
            read_data = data.read(16)
            assert b'\xcd\xcd\xcd\xcd' in read_data
        
        # Offset and rotation vectors are in 2 data blocks, each padded out to be a multiple of 64 bytes in size
        start_pos = data.tell()
        offsets = [struct.unpack("<ffff", data.read(16)) for _ in range(skeleton_length)]
        amount_read = data.tell() - start_pos
        while amount_read % 64 != 0:
            read_data = data.read(16)
            assert read_data == b'\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd\xcd', f"Offset: {data.tell()}\nData: {read_data}"
            amount_read += 16
        rotations = [struct.unpack("<ffff", data.read(16)) for _ in range(skeleton_length)]
        
        bones = [Bone(bone_names[i], numpy.array(offsets[i], dtype=numpy.float32), Rotation.from_quat(rotations[i]), []) for i in range(len(bone_names))]
        logger.info(f"Loaded skeleton {bones[1].name}")
        return cls(hierarchy, indices, variable_length_unknowns, constant_length_unknowns, bones, offsets, rotations)

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
class MRN:
    skeletons: List[Skeleton]

    @classmethod
    def load(cls, data: BytesIO) -> 'MRN':
        logger.info("Loading MRN file...")
        skeletons = []
        skeleton = Skeleton.load(data)
        while skeleton is not None:
            skeletons.append(skeleton)
            skeleton = Skeleton.load(data)
        logger.info(f"Loaded {len(skeletons)} skeletons!")
        return cls(skeletons)