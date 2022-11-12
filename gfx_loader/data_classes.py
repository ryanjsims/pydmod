from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional, Tuple

import bitstruct
import logging
import math
import os
import struct
import zlib

from utils import read_cstr

def read_netstr(data: BytesIO) -> str:
    count, length = 0, 0
    currbyte = data.read(1)[0]
    while currbyte & 0x80:
        length |= (currbyte & 0x7f) << 7 * count
        currbyte = data.read(1)[0]
        count += 1
    length |= (currbyte & 0x7f) << 7 * count
    to_decode = data.read(length)
    try:
        return str(to_decode, encoding='utf-8')
    except UnicodeDecodeError as e:
        logging.error(f"Could not decode bytes {to_decode} as UTF-8")
        raise e


def flatten(to_flatten: List[Tuple[int]]) -> List[int]:
    return [x for tuples in to_flatten for x in tuples]

@dataclass
class Rect:
    nbits: int
    xrange: Tuple[int, int]
    yrange: Tuple[int, int]

    @classmethod
    def load(cls, data: BytesIO) -> 'Rect':
        firstbyte = data.read(1)
        nbits = bitstruct.unpack("u5p3", firstbyte)[0]
        remaining = math.ceil((nbits*4 - 3) / 8)
        rect_bytes = firstbyte + data.read(remaining)
        xmin, xmax, ymin, ymax = bitstruct.unpack(("p5" + f"s{nbits}" * 4 + "p3"), rect_bytes)
        return cls(nbits, (xmin, xmax), (ymin, ymax))

@dataclass
class Header:
    magic: bytes
    version: int
    size: int
    frame_size: Rect
    frame_rate: float
    frame_count: int


    @classmethod
    def load(cls, data: BytesIO) -> 'Header':
        magic = data.read(3)
        assert magic == b'FWS', f"{magic}: Not an uncompressed SWF file"
        version, size = struct.unpack('<BI', data.read(5))
        frame_size = Rect.load(data)
        frame_rate, frame_count = struct.unpack("HH", data.read(4))
        frame_rate = frame_rate / 256
        return cls(magic, version, size, frame_size, frame_rate, frame_count)

@dataclass
class Tag:
    code: int
    len: int

    @classmethod
    def load(cls, data: BytesIO) -> 'Tag':
        code, len = bitstruct.unpack("u10u6", bitstruct.byteswap('2', data.read(2)))
        if len == 0x3f:
            len = struct.unpack("<I", data.read(4))[0]
        return cls(code, len)

@dataclass
class Generic(Tag):
    data: bytes

    @classmethod
    def load(cls, data: BytesIO, code: int, size: int) -> 'Generic':
        return cls(code, size, data.read(size))
    
    def __str__(self) -> str:
        return f"Generic(code={self.code}, size={self.len})"

    def __repr__(self) -> str:
        return f"Generic(code={self.code}, size={self.len})"
    

@dataclass
class ExporterInfo(Tag):
    version: int
    flags: Optional[int]
    bitmap_format: int
    prefix: bytes
    swf_name: str
    code_offsets: Optional[List[int]]

    @classmethod
    def load(cls, data: BytesIO, size: int) -> 'ExporterInfo':
        start = data.tell()
        version = struct.unpack("H", data.read(2))[0]
        if version > 0x10a:
            flags = struct.unpack("I", data.read(4))[0]
        else:
            flags = None
        bitmap_format, prefix_len = struct.unpack("HB", data.read(3))
        prefix = struct.unpack(f"{prefix_len}s", data.read(prefix_len))[0]
        swf_name = read_netstr(data)
        if data.tell() - start < size:
            code_offsets_len = struct.unpack("H", data.read(2))[0]
            code_offsets = flatten(list(struct.iter_unpack("I", data.read(code_offsets_len * 4))))
        else:
            code_offsets = []
        return cls(1000, size, version, flags, bitmap_format, prefix, swf_name, code_offsets)

@dataclass
class DefineExternalImage2(Tag):
    character_id: int
    bitmap_format: int
    target_width: int
    target_height: int
    export_name: str
    file_name: str
    extra_data: Optional[bytes]

    @classmethod
    def load(cls, data: BytesIO, size: int) -> 'DefineExternalImage2':
        start = data.tell()
        character_id, bitmap_format, target_width, target_height = struct.unpack("IHHH", data.read(10))
        export_name = read_netstr(data)
        file_name = read_netstr(data)
        if data.tell() - start < size:
            extra_data = data.read(size - (data.tell() - start))
        else:
            extra_data = None

        return cls(1009, size, character_id, bitmap_format, target_width, target_height, export_name, file_name, extra_data)

@dataclass
class CFX:
    header: Header
    tags: List[Tag]

    @classmethod
    def load(cls, gfx: BytesIO) -> 'CFX':
        magic = gfx.read(3)
        assert magic == b'CFX', f"{magic}: Not a ZLIB compressed GFX file"
        version_size = gfx.read(5)
        _, size = struct.unpack('<BI', version_size)
        data = zlib.decompress(gfx.read(), bufsize=size)

        swf = BytesIO(b'FWS' + version_size + data)
        header = Header.load(swf)

        tags = []
        
        tag = Tag.load(swf)
        if tag.code == 1000:
            exporter_info = ExporterInfo.load(swf, tag.len)
        else:
            logging.error("First tag was not an ExporterInfo tag!")
            return 1
        
        tags.append(exporter_info)
        tag = Tag.load(swf)
        while tag.code != 0:
            logging.debug(tag)
            if tag.code == 1009:
                tags.append(DefineExternalImage2.load(swf, tag.len))
            else:
                tags.append(Generic.load(swf, tag.code, tag.len))
            tag = Tag.load(swf)

        return cls(header, tags)