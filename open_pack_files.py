import logging

from glob import glob
from multiprocessing import Pool
from pathlib import Path
from export_manager import ExportManager

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
    with Pool(8) as pool:
        manager = ExportManager(paths, p=pool)
        print("Loading packs...")
        manager.loaded.wait()
        print("Loaded.")

if __name__ == "__main__":
    main()