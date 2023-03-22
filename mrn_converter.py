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

def add_skeleton_to_gltf(gltf: GLTF2, skeleton: Skeleton) -> bytes:
    joints = []
    matrices_bin = b''
    for bone in skeleton.bones[2:]:
        joints.append(len(gltf.nodes))
        gltf.nodes.append(Node(
            translation=bone.offset.tolist()[:3],
            rotation=bone.rotation.as_quat().tolist(),
            scale=[1, 1, 1],
            name=f"{bone.name.upper()}",
            children=[child.index - 2 for child in bone.children]
        ))
        bind_matrix = numpy.matrix(bone.global_transform, dtype=numpy.float32).I
        matrices_bin += bind_matrix.flatten().tobytes()
    
    gltf.skins.append(Skin(inverseBindMatrices=len(gltf.accessors), joints=joints))
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

def main():
    parser = ArgumentParser(description="MRN to GLTF Animation exporter")
    parser.add_argument("input_file", type=str, help="Name of the input MRN animation file")
    parser.add_argument("--list", "-l", action="store_true", help="List the available skeletons to export, then exit")
    parser.add_argument("--list-anims", "-a", action="store_true", help="List the available animations to export, then exit")
    parser.add_argument("--skeleton", "-s", type=str, default="", help="Name of the skeleton to export")
    parser.add_argument("--export-anim", "-e", type=str, default="", help="Specific animation to export")
    parser.add_argument("--output-file", "-o", type=str, help="Where to store the converted file. If not provided, will use the input filename and change the extension")
    parser.add_argument("--format", "-f", choices=["gltf", "glb"], help="The output format to use, required for conversion")
    parser.add_argument("--verbose", "-v", help="Increase log level, can be specified multiple times", action="count", default=0)
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

    skeleton_index = mrn.skeleton_names_packet().skeletons.index_of(args.skeleton)
    skeleton = mrn.skeleton_packets()[skeleton_index].skeleton
    skeleton.calc_global_transforms()
    #skeleton.pretty_print()

    gltf = GLTF2()
    blob = add_skeleton_to_gltf(gltf, skeleton)

    path = Path(args.output_file if args.output_file else "export/animations/" + args.skeleton + ".gltf")

    logger.info(f"Exporting animations for {args.skeleton}...")
    for i, name in enumerate(mrn.filenames_packet().files.animation_names):
        if args.export_anim != "" and args.export_anim != name or args.export_anim == "" and not name.split("_")[0] == args.skeleton:
            continue
        logger.info(f"\t{i: 3d}: {name}")
        animation_data = b''
        offset = 0

        animation_packet: AnimationPacket = mrn.packets[i]
        animation = animation_packet.animation
        if animation.dynamic_data is None:
            continue

        gltf_animation = Animation(name=name)
        
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

        offset = len(animation_data)
        
        animation.dynamic_data.dequantize(animation.trs_anim_deq_factors, animation.translation_init_factors)
        if len(animation.dynamic_bones.translation) > 0:
            for j, bone in enumerate(animation.dynamic_bones.translation):
                gltf_animation.channels.append(AnimationChannel(
                    sampler=len(gltf_animation.samplers),
                    target=AnimationChannelTarget(
                        node=bone - 1,
                        path="translation"
                    ),
                    extras = {
                        "bone_name": f"{skeleton.bones[bone+1].name}"
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
                offset = len(animation_data)

                gltf_animation.samplers.append(AnimationSampler(
                    input=time_accessor,
                    output=data_accessor
                ))
        
        if len(animation.dynamic_bones.rotation) > 0:
            for j, bone in enumerate(animation.dynamic_bones.rotation):
                #print(animation.dynamic_data.rotation[:, j, :])
                gltf_animation.channels.append(AnimationChannel(
                    sampler=len(gltf_animation.samplers),
                    target=AnimationChannelTarget(
                        node=bone - 1,
                        path="rotation"
                    ),
                    extras = {
                        "bone_name": f"{skeleton.bones[bone+1].name}"
                    }
                ))

                data_accessor = len(gltf.accessors)
                #print(animation.dynamic_data.translation[:, j, :] + skeleton.bones[bone].global_offset)
                rotation = Rotation.from_quat(animation.dynamic_data.rotation[:, j, :])
                local_rotation = skeleton.bones[bone+1].rotation * rotation[0].inv()
                
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
                offset = len(animation_data)

                gltf_animation.samplers.append(AnimationSampler(
                    input=time_accessor,
                    output=data_accessor
                ))

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