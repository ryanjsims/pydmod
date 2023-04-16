import logging
import sys

from argparse import ArgumentParser
from pygltflib import *
from pathlib import *
import numpy
from scipy.spatial.transform import Rotation

from mrn_loader import *

logger = logging.getLogger("MRN Converter")

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

def add_skeleton_to_gltf(gltf: GLTF2, skeleton: Skeleton, skeleton_name: str, bone_map: dict = None) -> bytes:
    joints = []
    matrices_bin = b''
    if bone_map is not None:
        remapped_bones = [Bone.FAKE()] * len(skeleton.bones)
        for bone in skeleton.bones:
            bone.index = bone_map[bone.name]["end"]
            remapped_bones[bone_map[bone.name]["end"]] = bone
        skeleton.bones = remapped_bones

    for bone in skeleton.bones:
        joints.append(len(gltf.nodes))
        node = Node(
            translation=bone.offset.tolist()[:3],
            rotation=bone.rotation.as_quat().tolist(),
            scale=[1, 1, 1],
            name=f"{bone.name.upper()}",
            children=[child.index for child in bone.children]
        )
        
        gltf.nodes.append(node)
        bind_matrix = numpy.matrix(bone.global_transform, dtype=numpy.float32).I
        matrices_bin += bind_matrix.flatten().tobytes()
    
    gltf.skins.append(Skin(name=skeleton_name, inverseBindMatrices=len(gltf.accessors), joints=joints))
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews),
        componentType=FLOAT,
        count=len(gltf.nodes),
        type=MAT4,
    ))
    gltf.bufferViews.append(BufferView(
        buffer=len(gltf.buffers),
        byteOffset=0,
        byteLength=len(matrices_bin)
    ))
    gltf.buffers.append(Buffer(
        byteLength=len(matrices_bin)
    ))

    return matrices_bin

skips = [
    "R_TOE", "L_TOE", "HEAD_END", 
    #"L_middle_end".upper(), "L_ring_end".upper(), "L_pinky_end".upper(), "L_index_end".upper(), "L_thumb_end".upper(), 
    #"R_middle_end".upper(), "R_ring_end".upper(), "R_pinky_end".upper(), "R_index_end".upper(), "R_thumb_end".upper()
]

def add_dynamic_animation(gltf: GLTF2, gltf_animation: Animation, skeleton: Skeleton, animation_pkt: AnimationPacket, offset: int) -> bytes:
    animation_data = b''
    animation = animation_pkt.animation

    bone_offset = 1
    logger.debug(f"{bone_offset=}")
    
    sample_times = numpy.array([index / animation.framerate for index in range(animation.dynamic_data.sample_count)], dtype=numpy.float32)
    time_accessor = len(gltf.accessors)
    animation_data += sample_times.tobytes()
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews),
        componentType=FLOAT,
        count=len(sample_times),
        type=SCALAR,
        min=[float(sample_times[0])],
        max=[float(sample_times[-1])]
    ))
    gltf.bufferViews.append(BufferView(
        buffer=len(gltf.buffers),
        byteOffset=offset,
        byteLength=len(animation_data)
    ))

    offset += len(animation_data)
    
    animation.dynamic_data.dequantize(animation.trs_anim_deq_factors, animation.translation_init_factors)
    if len(animation.dynamic_bones.translation) > 0:
        logger.debug("Dynamic Translation Bones:")
        for j, bone in enumerate(animation.dynamic_bones.translation):
            logger.debug(f"    {bone+bone_offset}: {skeleton.bones[bone+bone_offset].name}")
            gltf_animation.channels.append(AnimationChannel(
                sampler=len(gltf_animation.samplers),
                target=AnimationChannelTarget(
                    node=bone + bone_offset,
                    path="translation"
                ),
                extras = {
                    "bone_name": f"{skeleton.bones[bone+bone_offset].name}"
                }
            ))

            data_accessor = len(gltf.accessors)
            #print(animation.dynamic_data.translation[:, j, :] + skeleton.bones[bone].global_offset)
            translation_data = animation.dynamic_data.translation[:, j, :].flatten().tobytes()
            gltf.accessors.append(Accessor(
                bufferView=len(gltf.bufferViews),
                componentType=FLOAT,
                count=len(sample_times),
                type=VEC3
            ))
            gltf.bufferViews.append(BufferView(
                buffer=len(gltf.buffers),
                byteOffset=offset,
                byteLength=len(translation_data)
            ))

            animation_data += translation_data
            offset += len(translation_data)

            gltf_animation.samplers.append(AnimationSampler(
                input=time_accessor,
                output=data_accessor
            ))
    
    bone_offset = 1

    if len(animation.dynamic_bones.rotation) > 0:
        logger.debug("Dynamic Rotation Bones:")
        for j, bone in enumerate(animation.dynamic_bones.rotation):
            logger.debug(f"    {bone+bone_offset}: {skeleton.bones[bone+bone_offset].name}")
            #print(animation.dynamic_data.rotation[:, j, :])
            rotation = Rotation.from_quat(animation.dynamic_data.rotation[:, j, :])
            initial_rotations = Rotation.from_quat(animation.dynamic_data.initial_rotations)
            local_rotation = initial_rotations[j]
            gltf_animation.channels.append(AnimationChannel(
                sampler=len(gltf_animation.samplers),
                target=AnimationChannelTarget(
                    node=bone + bone_offset,
                    path="rotation"
                ),
                extras = {
                    "bone_name": f"{skeleton.bones[bone+bone_offset].name}",
                }
            ))

            data_accessor = len(gltf.accessors)
            #print(animation.dynamic_data.translation[:, j, :] + skeleton.bones[bone].global_offset)
            logger.debug(f"Bone {j}: {skeleton.bones[bone+bone_offset].name}")
            logger.debug(f"\tFirst frame inv: {rotation[0].inv().as_quat()}")
            logger.debug(f"\tBone rotation:   {skeleton.bones[bone+bone_offset].rotation.as_quat()}")
            logger.debug(f"\tOffset rotation: {local_rotation.as_quat()}")
            
            transformed = local_rotation * rotation
            # rotation_data = animation.dynamic_data.rotation[:, j, [0, 1, 2, 3]]
            # rotation_data = rotation_data.flatten().tobytes()
            rotation_data = numpy.array(transformed.as_quat().tolist(), dtype=numpy.float32).flatten().tobytes()
            gltf.accessors.append(Accessor(
                bufferView=len(gltf.bufferViews),
                componentType=FLOAT,
                count=len(sample_times),
                type=VEC4
            ))
            gltf.bufferViews.append(BufferView(
                buffer=len(gltf.buffers),
                byteOffset=offset,
                byteLength=len(rotation_data)
            ))

            animation_data += rotation_data
            offset += len(rotation_data)

            gltf_animation.samplers.append(AnimationSampler(
                input=time_accessor,
                output=data_accessor
            ))

    return animation_data

def add_static_pose(gltf: GLTF2, gltf_animation: Animation, skeleton: Skeleton, animation_pkt: AnimationPacket, offset: int) -> bytes:
    animation_data = b''
    animation = animation_pkt.animation

    sample_times = numpy.array([0], dtype=numpy.float32)
    time_accessor = len(gltf.accessors)
    animation_data += sample_times.tobytes()
    gltf.accessors.append(Accessor(
        bufferView=len(gltf.bufferViews),
        componentType=FLOAT,
        count=len(sample_times),
        type=SCALAR,
        min=[float(sample_times[0])],
        max=[float(sample_times[-1])]
    ))
    gltf.bufferViews.append(BufferView(
        buffer=len(gltf.buffers),
        byteOffset=offset,
        byteLength=len(animation_data)
    ))

    offset += len(animation_data)

    animation.static_data.dequantize()
    bone_offset = 1
    if len(animation.static_bones.translation) > 0:
        logger.debug("Static Translation Bones:")
        for j, bone in enumerate(animation.static_bones.translation):
            logger.debug(f"    {bone+bone_offset}: {skeleton.bones[bone+bone_offset].name}")
            translation_data = animation.static_data.translation[j, :].flatten().tobytes()
            gltf_animation.channels.append(AnimationChannel(
                sampler=len(gltf_animation.samplers),
                target=AnimationChannelTarget(
                    node=bone + bone_offset,
                    path="translation"
                ),
                extras = {
                    "bone_name": f"{skeleton.bones[bone+bone_offset].name}"
                }
            ))

            data_accessor = len(gltf.accessors)
            #print(animation.dynamic_data.translation[:, j, :] + skeleton.bones[bone].global_offset)
            gltf.accessors.append(Accessor(
                bufferView=len(gltf.bufferViews),
                componentType=FLOAT,
                count=1,
                type=VEC3
            ))
            gltf.bufferViews.append(BufferView(
                buffer=len(gltf.buffers),
                byteOffset=offset,
                byteLength=len(translation_data)
            ))

            animation_data += translation_data
            offset += len(translation_data)

            gltf_animation.samplers.append(AnimationSampler(
                input=time_accessor,
                output=data_accessor
            ))

    bone_offset = 1

    if len(animation.static_bones.rotation) > 0:
        logger.debug("Static Rotation Bones:")
        for j, bone in enumerate(animation.static_bones.rotation):
            logger.debug(f"    {bone+bone_offset}: {skeleton.bones[bone+bone_offset].name}")
            rotation = animation.static_data.rotation[j, :]
            gltf_animation.channels.append(AnimationChannel(
                sampler=len(gltf_animation.samplers),
                target=AnimationChannelTarget(
                    node=bone + bone_offset,
                    path="rotation"
                ),
                extras = {
                    "bone_name": f"{skeleton.bones[bone+bone_offset].name}",
                }
            ))

            data_accessor = len(gltf.accessors)

            rotation_data = rotation.flatten().tobytes()
            gltf.accessors.append(Accessor(
                bufferView=len(gltf.bufferViews),
                componentType=FLOAT,
                count=1,
                type=VEC4
            ))
            gltf.bufferViews.append(BufferView(
                buffer=len(gltf.buffers),
                byteOffset=offset,
                byteLength=len(rotation_data)
            ))

            animation_data += rotation_data
            offset += len(rotation_data)

            gltf_animation.samplers.append(AnimationSampler(
                input=time_accessor,
                output=data_accessor
            ))
    
    return animation_data

def main():
    parser = ArgumentParser(description="MRN to GLTF Animation exporter")
    parser.add_argument("input_file", type=str, help="Name of the input MRN animation file")
    parser.add_argument("--list", "-l", action="store_true", help="List the available skeletons to export, then exit")
    parser.add_argument("--list-anims", "-a", action="store_true", help="List the available animations to export, then exit")
    parser.add_argument("--skeleton", "-s", type=str, default="", help="Name of the skeleton to export")
    parser.add_argument("--export-anims", "-e", nargs="*", type=str, help="Specific animation to export")
    parser.add_argument("--output-file", "-o", type=str, help="Where to store the converted file. If not provided, will use the skeleton name")
    parser.add_argument("--format", "-f", choices=["gltf", "glb"], help="The output format to use, required for conversion")
    parser.add_argument("--verbose", "-v", help="Increase log level, can be specified multiple times", action="count", default=0)
    parser.add_argument("--remap", "-r", type=str, help="Use a bone map to remap the indices bones are placed at. Used for ThirdPersonX64.mrn since its skeleton does not have the correct indices.")
    args = parser.parse_args()

    logging.basicConfig(level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG), handlers=[handler])

    with open(args.input_file, "rb") as in_file:
        mrn = MRN.load(in_file)
    
    if mrn.skeleton_names_packet() is None:
        logger.error("No skeleton names packet in MRN!")
        sys.exit(-1)
    
    if args.list:
        print(f"{len(mrn.skeleton_names_packet().skeletons)} available skeletons:")
        for name in mrn.skeleton_names_packet().skeletons:
            print(f"\t{name}")

    if args.list_anims:
        print(f"{len(mrn.filenames_packet().files.animation_names)} available animations:")
        for name in mrn.filenames_packet().files.animation_names:
            print(f"\t{name}")
    
    if args.list or args.list_anims:
        sys.exit(0)

    if args.skeleton == "":
        logger.error(f"Skeleton name is required to export an animation set")
        sys.exit(-2)

    if not args.skeleton in mrn.skeleton_names_packet().skeletons:
        logger.error(f"Skeleton: '{args.skeleton}' not present in MRN!")
        sys.exit(-3)
    
    bone_remap = None
    if args.remap:
        with open(args.remap) as f:
            bone_remap = json.load(f)

    skeleton_index = mrn.skeleton_names_packet().skeletons.index_of(args.skeleton)
    skeleton = mrn.skeleton_packets()[skeleton_index].skeleton
    skeleton.calc_global_transforms()
    #skeleton.pretty_print()
    path = Path(args.output_file if args.output_file else "export/animations/" + args.skeleton + ".gltf")

    gltf = GLTF2()
    blob = add_skeleton_to_gltf(gltf, skeleton, path.stem, bone_remap)

    animations_added = set()

    logger.info(f"Exporting animations for {args.skeleton}...")
    for i, name in enumerate(mrn.filenames_packet().files.animation_names):
        if args.export_anims and name not in args.export_anims or not args.export_anims and not name.split("_")[0] == args.skeleton:
            continue
        
        if name in animations_added:
            continue

        animations_added.add(name)

        logger.info(f"\t{i: 3d}: {name}")
        

        animation_packet: AnimationPacket = mrn.packets[i]
        animation = animation_packet.animation

        gltf_animation = Animation(name=name)

        animation_data = b''
        if animation.static_data is not None:
            animation_data += add_static_pose(gltf, gltf_animation, skeleton, animation_packet, len(animation_data))
            

        if animation.dynamic_data is not None:
            animation_data += add_dynamic_animation(gltf, gltf_animation, skeleton, animation_packet, len(animation_data))
        
        gltf.buffers.append(Buffer(
            byteLength=len(animation_data),
            uri=name + ".bin"
        ))

        with open(path.parent / str(gltf.buffers[-1].uri), "wb") as anim_file:
            anim_file.write(animation_data)
        gltf.animations.append(gltf_animation)

    
    gltf.buffers[0].uri = path.with_suffix(".bin").name
    gltf.save_json(str(path))
    with open(path.with_suffix(".bin"), "wb") as f:
        f.write(blob)
        
        

        




if __name__ == "__main__":
    main()