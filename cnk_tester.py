import logging
import multiprocessing
from time import sleep

from cnk_loader import ForgelightChunk
from io import BytesIO
from zone_converter import get_manager

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logging.basicConfig(level=logging.DEBUG, handlers=[handler])


def main():
    with multiprocessing.Pool(8) as pool:
        manager = get_manager(pool)
        manager.loaded.wait()
    chunk = ForgelightChunk.load(BytesIO(manager.get_raw("Nexus_-32_-32.cnk0").get_data()))
    print("Past chunk loading")
    sleep(5)

if __name__ == "__main__":
    main()