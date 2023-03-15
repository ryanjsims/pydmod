from mrn_loader import MRN, PacketType, AnimationPacket, SkeletonPacket, InitFactorIndices, Factors
import struct, math
from typing import Tuple, List

import logging

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

logging.basicConfig(level=logging.DEBUG, handlers=[handler])

with open("export/mrn/WheeledVehicleX64.mrn", "rb") as mrn_file:
    mrn = MRN.load(mrn_file)


packet: AnimationPacket = mrn.packets[8]

print("Translation")
for dec_factors in packet.animation.trs_anim_dec_factors.translation:
    print(dec_factors)

print("----\nRotation")
for dec_factors in packet.animation.trs_anim_dec_factors.rotation:
    print(dec_factors)

print("----\nScale")
for dec_factors in packet.animation.trs_anim_dec_factors.scale:
    print(dec_factors)

print(f"Static translation data len: {len(packet.animation.static_data.trs_data[0])}")
print(f"Static translation bones len: {len(packet.animation.static_bones.translation)}")
print(len(packet.animation.static_data.trs_data[0]) / len(packet.animation.static_bones.translation))
print(len(packet.animation.static_data.trs_data[0]) % len(packet.animation.static_bones.translation))

print(f"Dynamic translation data len: {len(packet.animation.dynamic_data.trs_data[0])}")
print(f"Dynamic translation bones len: {len(packet.animation.dynamic_bones.translation)}")
print(f"Dynamic data frames: {packet.animation.dynamic_data.sample_count}")
print(len(packet.animation.dynamic_data.trs_data[0]) / len(packet.animation.dynamic_bones.translation) / packet.animation.dynamic_data.sample_count)
print(len(packet.animation.dynamic_data.trs_data[0]) % len(packet.animation.dynamic_bones.translation))

skeleton_packet: SkeletonPacket = mrn.packets[27]
skeleton = skeleton_packet.skeleton
translating_bones, rotating_bones = [], []
print("Translating bones:")
for index in packet.animation.dynamic_bones.translation:
    print(skeleton.bones[index + 1].name)
    translating_bones.append(skeleton.bones[index + 1].name)

print("\nRotating bones:")
for index in packet.animation.dynamic_bones.rotation:
    print(skeleton.bones[index + 1].name)
    rotating_bones.append(skeleton.bones[index + 1].name)
print()


def unpack_pos(packed_pos: Tuple[int, int, int], init_factor_indices: InitFactorIndices, frame: int, pos_factors: List[Factors]) -> Tuple[float, float, float]:
    PRECISION_XY = 2048.0
    PRECISION_Z  = 1024.0
    x_factors = pos_factors[init_factor_indices.dequantization_factor_indices[0]]
    x_quant_factor = x_factors.q_max[0]
    x_quant_min = x_factors.q_min[0]

    y_factors = pos_factors[init_factor_indices.dequantization_factor_indices[1]]
    y_quant_factor = y_factors.q_max[1]
    y_quant_min = y_factors.q_min[1]

    z_factors = pos_factors[init_factor_indices.dequantization_factor_indices[2]]
    z_quant_factor = z_factors.q_max[2]
    z_quant_min = z_factors.q_min[2]

    dequant_x = x_quant_factor * (packed_pos[0] + (init_factor_indices.init_values[0] / 256.0) * PRECISION_XY) + x_quant_min 
    dequant_y = y_quant_factor * (packed_pos[1] + (init_factor_indices.init_values[1] / 256.0) * PRECISION_XY) + y_quant_min
    dequant_z = z_quant_factor * (packed_pos[2] + (init_factor_indices.init_values[2] / 256.0) * PRECISION_Z) + z_quant_min
    return dequant_x, dequant_y, dequant_z

tbone_count = len(packet.animation.dynamic_bones.translation)
rbone_count = len(packet.animation.dynamic_bones.rotation)
translation_factor_indices = packet.animation.dynamic_data.trs_factor_indices[0]
tchunk_size = len(packet.animation.dynamic_data.trs_data[0]) / len(translation_factor_indices)
for frame in range(packet.animation.dynamic_data.sample_count):
    print(f"{frame=}")
    for bone in range(tbone_count):
        if not (0 <= bone < 4):
            continue 
        data_offset = bone * tbone_count * 4 + frame * 4
        value = struct.unpack("<I", packet.animation.dynamic_data.trs_data[0][data_offset : data_offset + 4])[0]
        # Bit field representing x y and z as 11, 11, and 10 bit quantized ints
        # 31                                     0
        #  xxxx xxxx xxxy yyyy yyyy yyzz zzzz zzzz
        value = (value >> 21, (value >> 10) & 0x7ff, value & 0x3ff)

        index = int(data_offset / tchunk_size)
        deltas = unpack_pos(value, translation_factor_indices[index], frame, packet.animation.trs_anim_dec_factors[0])
        print(f"{translating_bones[bone]} --- ({deltas[0]: .5f}, {deltas[1]: .5f}, {deltas[2]: .5f})")

    for bone in range(rbone_count):
        if not (0 <= bone < 2):
            continue
        value = struct.unpack("<HHH", packet.animation.dynamic_data.trs_data[1][frame * rbone_count * 6 + bone * 6 : frame * rbone_count * 6 + bone * 6 + 6])
        # for index in range(6):
        #     byte = packet.animation.dynamic_data.trs_data[1][frame * rbone_count * 6 + bone * 6 + index]
        #     if byte & 0x80:
        #         byte = -(256 - int(byte))
        #     else:
        #         byte = int(byte)
        #     translation_factors = packet.animation.trs_anim_dec_factors.translation
        #     value.append(byte)
        print(f"{rotating_bones[bone]} --- ({value[0]: 6d}, {value[1]: 6d}, {value[2]: 6d})")
    print()


print("\nTranslation init and indices")
for data in packet.animation.dynamic_data.trs_factor_indices[0]:
    print(data.init_values, data.dequantization_factor_indices)

print("\nRotation init and indices")
for data in packet.animation.dynamic_data.trs_factor_indices[1]:
    print(data.init_values, data.dequantization_factor_indices)