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
    combined_translation = numpy.array([0, 0, 0], dtype=numpy.float32)
    combined_rotation = Rotation.from_quat([0, 0, 0, 1])
    for bone in skeleton.bones[2:]:
        joints.append(len(gltf.nodes))
        gltf.nodes.append(Node(
            translation=bone.offset.tolist()[:3],
            rotation=bone.rotation.as_quat().tolist(),
            scale=[1, 1, 1],
            name=f"{bone.index - 1} {bone.name.upper()}",
            children=[child.index - 2 for child in bone.children]
        ))
        combined_translation -= bone.offset[:3]
        combined_rotation *= bone.rotation.inv()
        bind_matrix = numpy.matrix(numpy.empty((4, 4)), dtype=numpy.float32)
        bind_matrix[:3, :3] = combined_rotation.as_matrix()
        bind_matrix[3, :3] = combined_translation
        bind_matrix[:, 3] = numpy.atleast_2d([0, 0, 0, 1]).T
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
    parser.add_argument("--skeleton", "-s", type=str, default="", help="Name of the skeleton to export")
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
        sys.exit(0)

    if args.skeleton == "":
        logger.error(f"Skeleton name is required to export an animation set")
        sys.exit(-2)

    if not args.skeleton in mrn.skeleton_names_packet().skeletons:
        logger.error(f"Skeleton: '{args.skeleton}' not present in MRN!")
        sys.exit(-3)

    skeleton_index = mrn.skeleton_names_packet().skeletons.index_of(args.skeleton)
    skeleton = mrn.skeleton_packets()[skeleton_index].skeleton
    skeleton.calc_global_offsets()
    #skeleton.bones[0].reorient()

    gltf = GLTF2()
    blob = add_skeleton_to_gltf(gltf, skeleton)

    path = Path(args.output_file if args.output_file else "export/animations/" + args.skeleton + ".gltf")

    logger.info(f"Exporting animations for {args.skeleton}...")
    for i, name in enumerate(mrn.filenames_packet().files.animation_names):
        animation_data = b''
        offset = 0
        if not name.split("_")[0] == args.skeleton:
            continue
        logger.info(f"\t{i: 3d}: {name}")

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
        
        if len(animation.dynamic_bones.translation) > 0:
            animation.dynamic_data.dequantize(animation.trs_anim_deq_factors)
            for j, bone in enumerate(animation.dynamic_bones.translation):
                gltf_animation.channels.append(AnimationChannel(
                    sampler=len(gltf_animation.samplers),
                    target=AnimationChannelTarget(
                        node=bone - 1,
                        path="translation"
                    )
                ))

                data_accessor = len(gltf.accessors)
                print(animation.dynamic_data.translation[:, j, :] + skeleton.bones[bone].global_offset)
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