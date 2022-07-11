# Forgelight DME model converter

This repo contains several scripts that allow the conversion of forgelight DME models to STL, GLTF, or OBJ models for use in model viewing and rendering programs, as well as a python library that allows parsing of DME models which powers the conversion script. I made this to convert models to .glb format for import into Blender. 

This software still has several issues. The actual hash function used for material names seems to have changed since the last time people did research on the DME format, so the script will default to loading the model using the PlayerStudio Vehicle material. If a different material is desired, you can provide specific name hashes on the commandline via the `-m` option. If the data strides listed in the DME file do not match the memory layout of the provided material, the tool will list materials that do match, and you may select the one to use depending on the mesh(es) being saved.

If you're finding this on github you probably know how to obtain DME models, if not there are several repositories dedicated to opening and saving game assets from .pack2 files.

## Usage

```
python dme_converter.py [-h] [--output-file OUTPUT_FILE] [--format {stl,gltf,obj,glb}] [--material-hashes MATERIAL_HASHES [MATERIAL_HASHES ...]] [--new-materials] [--dump-textures] [--embed-textures] [--verbose] input_file

DME v4 to GLTF/OBJ/STL converter

positional arguments:
  input_file            Name of the input DME model

optional arguments:
  -h, --help            show this help message and exit
  --output-file OUTPUT_FILE, -o OUTPUT_FILE
                        Where to store the converted file. If not provided, will use the input filename and change the extension
  --format {stl,gltf,obj,glb}, -f {stl,gltf,obj,glb}
                        The output format to use, required for conversion
  --material-hashes MATERIAL_HASHES [MATERIAL_HASHES ...], -m MATERIAL_HASHES [MATERIAL_HASHES ...]
                        The name hash(es) of the materials to use for each mesh when loading the DME data
  --new-materials, -n   Use a more recently generated materials.json file. Not super helpful yet since the hash function seems to be changed
  --dump-textures, -t   Dump the filenames of the textures used by the model to stdout and exit
  --embed-textures, -e  Embed the texture filenames used by the model in the output file, saving the textures alongside the output (GLTF/GLB only)
  --verbose, -v         Increase log level, can be specified multiple times
```

## 