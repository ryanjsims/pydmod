import logging
logger = logging.getLogger("Export Manager")

import DbgPack
from pathlib import Path
from PIL import Image
from typing import List, Optional
import magic
from adr_converter import load_adr, get_base_model, get_animation_network
from io import BytesIO

class ExportManager(DbgPack.AssetManager):
    def save_png(self, name: str, directory: Path = Path("./export/images")) -> bool:
        asset = self.get_raw(name)
        if asset is None:
            logger.error(f"{name} not found in assets")
            return False
        directory.mkdir(parents=True, exist_ok=True)
        filename = Path(name).with_suffix(".png")

        image = Image.open(BytesIO(asset.get_data()))
        try:
            image.save(directory / filename)
        except ValueError:
            print("Could not determine file type from filename??? Should not happen.")
            return False
        except OSError as e:
            print(e)
            return False
        return True

    def print(self, name: str, size_limit: int=5120):
        asset = self.get_raw(name)
        if asset is None:
            logger.error(f"{name} not found in assets")
            return
        print(str(asset.get_data()[:size_limit], encoding="utf-8").strip())
    
    def generate_namelist(self) -> List[str]:
        def filenames(base_name):
            return set([base_name + extension for extension in [".adr"]])
        to_return = set()
        for pack in self.packs:
            pack: DbgPack.Pack2
            for asset in pack.raw_assets.values():
                try:
                    data = asset.get_data()
                except Exception as e:
                    print(e)
                file_type = magic.from_buffer(data[:1024], mime=True)
                if file_type == 'text/plain' and b'ActorRuntime' in data[:16]:
                    adr = load_adr(BytesIO(data))
                    result = get_base_model(adr)
                    if result and result[0] and result[0] not in self:
                        model_name, _ = result
                        base_name = model_name[:model_name.lower().find("_lod0.dme")]
                        logger.info(f"adding {base_name}")
                        to_return |= filenames(base_name)
                    anim = get_animation_network(adr)
                    if anim:
                        to_return |= set([Path(anim).stem + "X64.mrn"])

        return sorted(list(to_return))
    
    def get_by_namehash(self, name_hash: int) -> Optional[DbgPack.Asset2]:
        if not self.loaded.is_set():
            return None
        for pack in self.packs:
            assert type(pack) == DbgPack.Pack2
            if name_hash in pack.raw_assets:
                return pack.raw_assets[name_hash]
        return None