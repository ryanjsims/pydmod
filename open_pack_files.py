import logging

from glob import glob
from multiprocessing import Pool
from pathlib import Path
from export_manager import ExportManager
from argparse import ArgumentParser

from io import BytesIO

logger = logging.getLogger("Open Pack Files")

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

server = "C:/Users/Public/Daybreak Game Company/Installed Games/PlanetSide 2 Test/Resources/Assets/"

paths = ([]
    + [Path(p) for p in glob(server + "assets_x64*.pack2")]
    + [Path(p) for p in glob(server + "data_x64*.pack2")]
    + [Path(p) for p in glob(server + "Nexus_x64*.pack2")]
    + [Path(p) for p in glob(server + "ui_x64*.pack2")]
)
        
manager = None

def main():
    global manager, paths, handler
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    parser = ArgumentParser(description="Opens pack files for interactive use")
    parser.add_argument("-n", "--namelist", default=None, type=str, help="Namelist file to load for additional file names")
    args = parser.parse_args()

    namelist = None
    if args.namelist is not None:
        with open(args.namelist) as f:
            namelist = f.read().split("\n")

    with Pool(8) as pool:
        manager = ExportManager(paths, p=pool, namelist=namelist)
        print("Loading packs...")
        manager.loaded.wait()
        print("Loaded.")

if __name__ == "__main__":
    main()
