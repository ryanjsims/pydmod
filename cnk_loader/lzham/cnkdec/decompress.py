from . import cnkdec

class Decompressor:
    def __init__(self):
        self._handler = cnkdec.lzham_decompress_init()
    
    def decompress(self, data: bytes, uncompressed_size: int) -> bytes:
        return cnkdec.lzham_decompress(self._handler, data, uncompressed_size)
    
    def version(self):
        return cnkdec.get_lzham_version()