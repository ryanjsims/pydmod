from mrn_loader import MRN, PacketType, AnimationPacket, SkeletonPacket, InitFactorIndices, Factors, FilenamePacket
import struct, math, json
from typing import Tuple, List

import logging

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

logging.basicConfig(level=logging.DEBUG, handlers=[handler])

with open("export/mrn/TurretAlternatingX64.mrn", "rb") as mrn_file:
    mrn = MRN.load(mrn_file)

files = None

for data_packet in reversed(mrn.packets):
    if data_packet.header.packet_type == PacketType.file_names:
        data_packet: FilenamePacket
        files = data_packet.files
        break

bones_per_dequant_count = {}
for i in range(len(files.filenames.strings)):
    packet: AnimationPacket = mrn.packets[i]
    if packet.animation.trs_anim_deq_counts[0] == 0:
        continue
    if packet.animation.trs_anim_deq_counts[0] not in bones_per_dequant_count:
        bones_per_dequant_count[packet.animation.trs_anim_deq_counts[0]] = {"count": 0, "total": 0, "indices": [], "names": [], "values": []}
    
    bones_per_dequant_count[packet.animation.trs_anim_deq_counts[0]]["count"] += 1
    bones_per_dequant_count[packet.animation.trs_anim_deq_counts[0]]["total"] += len(packet.animation.dynamic_bones.translation)
    bones_per_dequant_count[packet.animation.trs_anim_deq_counts[0]]["indices"].append(i)
    bones_per_dequant_count[packet.animation.trs_anim_deq_counts[0]]["names"].append(files.animation_names.strings[i])
    bones_per_dequant_count[packet.animation.trs_anim_deq_counts[0]]["values"].append(len(packet.animation.dynamic_bones.translation))
    # if packet.animation.trs_anim_deq_counts[0] == 2:
    #     print(f"Packet {i}: {files.animation_names.strings[i]} has:")
    #     print(f"\t{len(packet.animation.dynamic_bones.translation)} translating bones {packet.animation.dynamic_bones.translation}")
    #     #print(f"\t{len(packet.animation.dynamic_bones.rotation)} rotating bones    {packet.animation.dynamic_bones.rotation}")

for key in bones_per_dequant_count:
    bones_per_dequant_count[key]["average"] = bones_per_dequant_count[key]["total"] / bones_per_dequant_count[key]["count"]

#print(json.dumps(bones_per_dequant_count, indent=4))

if 1:
    packet: AnimationPacket = mrn.packets[100]
    print(files.animation_names.strings[100])
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

    skeleton_packet: SkeletonPacket = mrn.packets[219]
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

    tbone_count = len(packet.animation.dynamic_bones.translation)
    rbone_count = len(packet.animation.dynamic_bones.rotation)
    translation_factor_indices = packet.animation.dynamic_data.trs_factor_indices[0]
    tchunk_size = len(packet.animation.dynamic_data.trs_data[0]) / len(translation_factor_indices)
    for frame in range(packet.animation.dynamic_data.sample_count):
        print(f"{frame=}")
        for bone in range(tbone_count):
            # if not (0 <= bone < 4):
            #     continue 
            data_offset = bone * tbone_count * 4 + frame * 4
            value = struct.unpack("<I", packet.animation.dynamic_data.trs_data[0][data_offset : data_offset + 4])[0]
            # Bit field representing x y and z as 11, 11, and 10 bit quantized ints
            # 31                                     0
            #  xxxx xxxx xxxy yyyy yyyy yyzz zzzz zzzz
            value = (value >> 21 & 0x7ff, (value >> 10) & 0x7ff, value & 0x3ff)

            deltas = unpack_pos(value, translation_factor_indices[bone], frame, packet.animation.trs_anim_dec_factors[0])
            print(f"{translating_bones[bone]} --- ({value[0]:4d}, {value[1]:4d}, {value[2]:4d}) -> ({deltas[0]: .5f}, {deltas[1]: .5f}, {deltas[2]: .5f})")

        # for bone in range(rbone_count):
        #     if not (0 <= bone < 2):
        #         continue
        #     value = struct.unpack("<HHH", packet.animation.dynamic_data.trs_data[1][frame * rbone_count * 6 + bone * 6 : frame * rbone_count * 6 + bone * 6 + 6])
        #     # for index in range(6):
        #     #     byte = packet.animation.dynamic_data.trs_data[1][frame * rbone_count * 6 + bone * 6 + index]
        #     #     if byte & 0x80:
        #     #         byte = -(256 - int(byte))
        #     #     else:
        #     #         byte = int(byte)
        #     #     translation_factors = packet.animation.trs_anim_dec_factors.translation
        #     #     value.append(byte)
        #     print(f"{rotating_bones[bone]} --- ({value[0]: 6d}, {value[1]: 6d}, {value[2]: 6d})")
        print()


    print("\nTranslation init and indices")
    for data in packet.animation.dynamic_data.trs_factor_indices[0]:
        print(data.init_values, data.dequantization_factor_indices)

    print("\nRotation init and indices")
    for data in packet.animation.dynamic_data.trs_factor_indices[1]:
        print(data.init_values, data.dequantization_factor_indices)