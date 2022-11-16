# Forgelight DME model converter

## Table of Contents

1. [Description](#description)
2. [Installation](#installation)
    1. [Compiling `cnk_loader` with Windows](#compiling-cnk_loader-from-windows)
3. [Usage](#usage)

## Description

This repo contains several scripts that allow the conversion of forgelight DME/ADR/Zone models to STL, GLTF, or OBJ models for use in model viewing and rendering programs, as well as a python library that allows parsing of DME models which powers the conversion script. I made this to convert models to .glb format for import into Blender. 

This software still has several issues. Only the basic C/N/S texture maps are used in 95% of exports, with a few atlases added in specific cases for control consoles and computer screens. Detail maps are not used. If the data strides listed in the DME file do not match the memory layout of the provided material, the tool will list materials that do match, and you may select the one to use depending on the mesh(es) being saved.

Zones can take a while to export due to their size, and then importing the resulting GLTF files will take time as well. Since LOD0 meshes are exported, memory requirements for even small continents (IE Nexus or VR) can be large in Blender.

## Installation

1. After cloning the repository, run `git submodule update --init` to pull in all the needed submodules.
2. Create a virtual environment for the repository with `python3 -m venv venv`
3. Activate the virtual environment:
    * Windows: `& ./venv/bin/Activate.ps1`
    * Linux: `. venv/bin/activate`
4. Run `pip install -r requirements.txt`
5. Navigate to `./dbg-pack/` and run `pip install .`
6. If you need terrain loading, navigate to `./cnk_loader/` and run `pip install .`
    * Windows: check below for extra instructions

### Compiling `cnk_loader` from Windows

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
