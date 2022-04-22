import sys
import os
from argparse import ArgumentParser

from dme_loader import DME

def main():
    parser = ArgumentParser(description="DME v4 to OBJ converter")
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()
    try:
        with open(args.input_file, "rb") as in_file:
            dme = DME.load(in_file)
        
        with open(args.output_file + ".tmp", "w") as output:
            for mesh in dme.meshes:
                for vertex in mesh.vertices:
                    output.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

                for normal in mesh.normals:
                    output.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
                
                uv_offset = 0
                for uvs in mesh.uvs:
                    for uv in uvs:
                        output.write(f"vt {uv[0]} {uv[1]}\n")
                
                    for i in range(0, len(mesh.indices), 3):
                        i0, i1, i2 = mesh.indices[i] + 1, mesh.indices[i+1] + 1, mesh.indices[i+2] + 1
                        output.write(f"f {i1}/{i1 + uv_offset} {i0}/{i0 + uv_offset} {i2}/{i2 + uv_offset}\n")
                    uv_offset += len(uvs)

        os.replace(args.output_file + ".tmp", args.output_file)
    except FileNotFoundError:
        print(f"Error: Could not find {args.input_file}", file=sys.stderr)
    except AssertionError as e:
        print(f"Error: {e}", file=sys.stderr)
    


if __name__ == "__main__":
    main()