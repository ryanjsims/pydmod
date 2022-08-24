# Forgelight DME model converter

This repo contains several scripts that allow the conversion of forgelight DME models to STL, GLTF, or OBJ models for use in model viewing and rendering programs, as well as a python library that allows parsing of DME models which powers the conversion script. I made this to convert models to .glb format for import into Blender. 

This software still has several issues. The actual hash function used for material names seems to have changed since the last time people did research on the DME format, so the script will default to loading the model using the PlayerStudio Vehicle material. If a different material is desired, you can provide specific name hashes on the commandline via the `-m` option. If the data strides listed in the DME file do not match the memory layout of the provided material, the tool will list materials that do match, and you may select the one to use depending on the mesh(es) being saved.

If you're finding this on github you probably know how to obtain DME models, if not there are several repositories dedicated to opening and saving game assets from .pack2 files.

## Installation (Linux)

1. After cloning the repository, run `git submodule update --init` to pull in all the needed submodules.
2. Create a virtual environment for the repository and activate it with `python3 -m venv venv && . venv/bin/activate`
    * Windows: You'll need to run `./venv/bin/Activate.ps1` if you use powershell
3. Run `pip install -r requirements.txt`
4. Navigate to `./dbg-pack/` and run `pip install .`
5. If you need terrain loading, navigate to `./cnk_loader/` and run `pip install .`
    * Windows: check below for extra instructions

## Compiling `cnk_loader` from Windows

1. Install Visual Studio with the Desktop development with C++ and Python development workloads (VS2019 used when testing)
    * Python native development tools should be enabled
2. Under the start menu, navigate to the `Visual Studio` folder for your install
3. Launch `x64 Native Tools Command Prompt for VS 2019`
4. Clone the repository wherever you plan to use it, and navigate there in the Command Prompt
5. Navigate to `./cnk_loader/`
6. Run `pip install .` to install the library

## Usage
```
usage: adr_converter.py [-h] [--format {gltf,glb}] [--live] [--verbose]
                        input_file output_file

Actor Runtime (.adr) to gltf/glb converter

positional arguments:
  input_file            Path of the input ADR file
  output_file           Path of the output file

optional arguments:
  -h, --help            show this help message and exit
  --format {gltf,glb}, -f {gltf,glb}
                        The output format to use, required for conversion
  --live, -l            Load assets from live server rather than test
  --verbose, -v         Increase log level, can be specified multiple times
```

```
usage: dme_converter.py [-h] [--output-file OUTPUT_FILE]
                        [--format {stl,gltf,obj,glb}]
                        [--material-hashes MATERIAL_HASHES [MATERIAL_HASHES ...]]
                        [--dump-textures] [--verbose]
                        input_file

DME v4 to GLTF/OBJ/STL converter

positional arguments:
  input_file            Name of the input DME model

optional arguments:
  -h, --help            show this help message and exit
  --output-file OUTPUT_FILE, -o OUTPUT_FILE
                        Where to store the converted file. If not provided,
                        will use the input filename and change the extension
  --format {stl,gltf,obj,glb}, -f {stl,gltf,obj,glb}
                        The output format to use, required for conversion
  --material-hashes MATERIAL_HASHES [MATERIAL_HASHES ...], -m MATERIAL_HASHES [MATERIAL_HASHES ...]
                        The name hash(es) of the materials to use for each
                        mesh when loading the DME data
  --dump-textures, -t   Dump the filenames of the textures used by the model
                        to stdout and exit
  --verbose, -v         Increase log level, can be specified multiple times
```
