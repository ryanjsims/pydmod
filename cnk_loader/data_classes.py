import struct
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import List, Tuple, Union
from cnkdec import Decompressor

logger = logging.getLogger("cnk_loader")

@dataclass
class Header:
    magic: bytes
    version: int

    @classmethod
    def load(cls, data: BytesIO) -> 'Header':
        magic = data.read(4)
        assert magic in (b'CNK0', b'CNK1', b'CNK2'), "Unsupported file type!"
        version = struct.unpack("<I", data.read(4))[0]
        return cls(magic, version)
    
    def serialize(self) -> bytes:
        return self.magic + struct.pack("<I", self.version)

@dataclass
class Layer:
    unk1: int
    unk2: int

    @classmethod
    def load(cls, data: BytesIO) -> 'Layer':
        unk1, unk2 = struct.unpack("<II", data.read(8))
        return cls(unk1, unk2)

@dataclass
class Flora:
    layers: List[Layer]

    @classmethod
    def load(cls, data: BytesIO) -> 'Flora':
        layer_count = struct.unpack("<I", data.read(4))[0]
        logger.debug(f"Loading {layer_count} layers...")
        layers = [Layer.load(data) for _ in range(layer_count)]
        return cls(layers)

@dataclass
class Eco:
    id: int
    floras: List[Flora]

    @classmethod
    def load(cls, data: BytesIO) -> 'Eco':
        id, flora_count = struct.unpack("<II", data.read(8))
        logger.debug(f"Loading {flora_count} floras...")
        floras = [Flora.load(data) for _ in range(flora_count)]
        return cls(id, floras)

@dataclass
class Tile:
    x: int
    y: int
    unk1: int
    unk2: int
    ecos: List[Eco]
    index: int
    unk3: int
    image_data: bytes
    layer_textures: List[int]

    @classmethod
    def load(cls, data: BytesIO) -> 'Tile':
        x, y, unk1, unk2, eco_count = struct.unpack("<iiiiI", data.read(20))
        logger.debug(f"Loading {eco_count} ecos...")
        ecos = [Eco.load(data) for _ in range(eco_count)]
        index, unk3, img_length = struct.unpack("<III", data.read(12))
        image_data = data.read(img_length)
        layer_length = struct.unpack("<I", data.read(4))[0]
        layer_textures = list(map(lambda x: x[0], struct.iter_unpack("<B", data.read(layer_length))))
        return cls(x, y, unk1, unk2, ecos, index, unk3, image_data, layer_textures)
    
    def __repr__(self) -> str:
        return f"Tile(x={self.x}, y={self.y}, unk1={self.unk1}, unk2={self.unk2}, ecos=Eco[{len(self.ecos)}], index={self.index}, image_data={self.image_data[:4]} + {len(self.image_data[4:])} bytes, layer_textures={self.layer_textures})"

@dataclass
class Tiles:
    tiles: List[Tile]

    @classmethod
    def load(cls, data: BytesIO) -> 'Tiles':
        tile_count = struct.unpack("<I", data.read(4))[0]
        logger.info(f"Loading {tile_count} tiles...")
        tiles = [Tile.load(data) for _ in range(tile_count)]
        return cls(tiles)

    def __iter__(self):
        return iter(self.tiles)

    def __str__(self) -> str:
        return str(self.tiles)

    def __repr__(self) -> str:
        return repr(self.tiles)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, index) -> Tile:
        return self.tiles[index]

@dataclass
class Vertex:
    x: int
    y: int
    height_far: int
    height_near: int
    color1: int
    color2: int

    @classmethod
    def load(cls, data: BytesIO) -> 'Vertex':
        return cls(*struct.unpack("<hhhhII", data.read(16)))

@dataclass
class RenderBatch:
    index_offset: int
    index_count: int
    vertex_offset: int
    vertex_count: int

    @classmethod
    def load(cls, data: BytesIO) -> 'RenderBatch':
        return cls(*struct.unpack("<IIII", data.read(16)))

@dataclass
class OptimizedDraw:
    data: bytes

    @classmethod
    def load(cls, data: BytesIO) -> 'OptimizedDraw':
        return cls(data.read(320))

@dataclass
class TileOccluderInfo:
    data: bytes

    @classmethod
    def load(cls, data: BytesIO) -> 'TileOccluderInfo':
        return cls(data.read(64))

@dataclass
class CNK0:
    header: Header
    tiles: Tiles
    unk1: int
    unkArr1: List[Tuple[int, int, int]]
    indices: List[int]
    vertices: List[Vertex]
    render_batches: List[RenderBatch]
    optimized_draw: List[OptimizedDraw]
    unk_shorts: List[int]
    unk_vectors: List[Tuple[float, float, float]]
    tile_occluder_info: List[TileOccluderInfo]

    @classmethod
    def load(cls, data: BytesIO) -> 'CNK0':
        logger.info("Loading CNK0 file...")
        header = Header.load(data)
        if header.magic != b'CNK0':
            raise ValueError("Not a CNK0 file!")
        tiles = Tiles.load(data)
        unk1, unkarr_len = struct.unpack("<II", data.read(8))
        unkArr1 = [struct.unpack("<hBB", data.read(4)) for _ in range(unkarr_len)]
        indices_len = struct.unpack("<I", data.read(4))[0]
        indices = [struct.unpack("<H", data.read(2))[0] for _ in range(indices_len)]
        vertices_len = struct.unpack("<I", data.read(4))[0]
        vertices = [Vertex.load(data) for _ in range(vertices_len)]
        render_batches_len = struct.unpack("<I", data.read(4))[0]
        render_batches = [RenderBatch.load(data) for _ in range(render_batches_len)]
        optimized_draw_len = struct.unpack("<I", data.read(4))[0]
        optimized_draw = [OptimizedDraw.load(data) for _ in range(optimized_draw_len)]
        unk_shorts_len = struct.unpack("<I", data.read(4))[0]
        unk_shorts = list(map(lambda x: x[0], struct.iter_unpack("<H", data.read(2 * unk_shorts_len))))
        unk_vectors_len = struct.unpack("<I", data.read(4))[0]
        unk_vectors = list(struct.iter_unpack("<fff", data.read(12 * unk_vectors_len)))
        tile_occluder_info_len = struct.unpack("<I", data.read(4))[0]
        tile_occluder_info = [TileOccluderInfo.load(data) for _ in range(tile_occluder_info_len)]
        logger.info("CNK0 file loaded")
        return cls(header, tiles, unk1, unkArr1, indices, vertices, render_batches, optimized_draw, unk_shorts, unk_vectors, tile_occluder_info)

@dataclass
class CNK1:
    header: Header

    @classmethod
    def load(cls, data: BytesIO) -> 'CNK1':
        header = Header.load(data)
        if header.magic != b'CNK1':
            raise ValueError("Not a CNK1 file!")
        return cls(header)

@dataclass
class ForgelightChunk:
    chunk: Union[CNK0, CNK1]

    @classmethod
    def load(cls, data: BytesIO, compressed: bool = True) -> Union[CNK0, CNK1]:
        header = Header.load(data)
        if compressed:
            logger.info("Decompressing chunk data...")
            decompressor = Decompressor()
            logger.info(f"Decompressor lib version: {decompressor.version()}")
            decompressed_size, compressed_size = struct.unpack("<II", data.read(8))
            compressed_data = data.read()
            logger.info(f"Decompressed size: {decompressed_size}")
            logger.info(f"Compressed size: {compressed_size}")
            if len(compressed_data) != compressed_size:
                logger.warning(f"Compressed data length {len(compressed_data)} != file value {compressed_size}")
            decompressed_data = decompressor.decompress(compressed_data, decompressed_size)
            if len(decompressed_data) != decompressed_size:
                logger.warning("Decompressed data length different from listed size!")
            data = BytesIO(header.serialize() + decompressed_data)
        data.seek(0)
        
        if header.magic == b'CNK0':
            chunk = CNK0.load(data)
        elif header.magic == b'CNK1':
            chunk = CNK1.load(data)
        elif header.magic == b'CNK2':
            raise NotImplementedError("CNK2 not implemented yet!")
        else:
            assert False, "This should never happen"


        return chunk