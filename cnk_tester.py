import logging

from cnk_loader import ForgelightChunk

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))
logging.basicConfig(level=logging.INFO, handlers=[handler])


with open("dbg-pack/export/Oshur_8_0.cnk0", "rb") as oshur_chunk:
    chunk = ForgelightChunk.load(oshur_chunk)