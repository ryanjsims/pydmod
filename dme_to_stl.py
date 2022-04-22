import sys
import numpy
from stl import mesh

from dme_loader import DME

def main():
    if len(sys.argv) < 3:
        print("Usage: dme_to_stl.py <input> <output>", file=sys.stderr)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    with open(input_file, "rb") as f_in:
        data = DME.load(f_in)

    faces = []
    for i in range(0, len(data.meshes[0].indices), 3):
        faces.append((data.meshes[0].indices[i+2], data.meshes[0].indices[i+1], data.meshes[0].indices[i]))
    faces = numpy.array(faces)
    vertex_array = numpy.array(data.meshes[0].vertices)
    stl_mesh = mesh.Mesh(numpy.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = vertex_array[f[j], :]

    stl_mesh.save(output_file)
    print("Saved to {}".format(output_file))

main()
