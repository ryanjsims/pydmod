import struct
import logging
from dataclasses import dataclass
from io import SEEK_END, SEEK_SET, BytesIO
from typing import List, Tuple, Union
from cnkdec import Decompressor
from PIL import Image, ImageChops

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
    image_data: bytes
    layer_textures: List[int]

    @classmethod
    def load(cls, data: BytesIO) -> 'Tile':
        logger.debug(f"offset: {hex(data.tell())}")
        to_unpack = data.read(20)
        logger.debug(f"Attempting to unpack {len(to_unpack)} bytes... ({to_unpack})")
        x, y, unk1, unk2, eco_count = struct.unpack("<iiiiI", to_unpack)
        logger.debug(f"Loading {eco_count} ecos...")
        ecos = [Eco.load(data) for _ in range(eco_count)]
        index, img_id = struct.unpack("<II", data.read(8))
        logger.debug(f"Index: {index} img_id: {img_id}")
        image_data = None
        if img_id:
            img_length = struct.unpack("<I", data.read(4))[0]
            image_data = data.read(img_length)
        layer_length = struct.unpack("<I", data.read(4))[0]
        logger.debug(f"Loading layer textures of length {layer_length}")
        layer_textures = list(map(lambda x: x[0], struct.iter_unpack("<B", data.read(layer_length))))
        return cls(x, y, unk1, unk2, ecos, index, image_data, layer_textures)
    
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
    verts: List[Tuple[float, float, float]]
    uvs: List[Tuple[float, float]]
    triangles: List[List[int]]
    aabb: Tuple[Tuple[float, float, float], Tuple[float, float, float]]

    def calculate_verts(self):
        minimum, maximum = None, None
        for i in range(len(self.render_batches)):
            for j in range(self.render_batches[i].vertex_count):
                offset = self.render_batches[i].vertex_offset + j
                x = float(self.vertices[offset].x - (((len(self.render_batches) - 1 - i) % 4) + 1) * 64)
                z = float(self.vertices[offset].y - (((len(self.render_batches) - 1 - i) >> 2) + 1) * 64)
                heightNear = float(self.vertices[offset].height_near / 32)

                self.verts.append((x, heightNear, z))
                if minimum is None or self.verts[-1][0] < minimum[0] and self.verts[-1][1] < minimum[1] and self.verts[-1][2] < minimum[2]:
                    minimum = self.verts[-1]
                if maximum is None or self.verts[-1][0] > maximum[0] and self.verts[-1][1] > maximum[1] and self.verts[-1][2] > maximum[2]:
                    maximum = self.verts[-1]
                self.uvs.append((((z / 128) / 2) + 1, (-(1 - x / 128) + 1) / 2) - 1)
            
            self.triangles.append([0] * self.render_batches[i].index_count)
            for j in range(self.render_batches[i].index_count):
                self.triangles[i][self.render_batches[i].index_count - 1 - j] = self.indices[j + self.render_batches[i].index_offset] + self.render_batches[i].vertex_offset
        self.aabb = (minimum if minimum is not None else (0, 0, 0), maximum if maximum is not None else (0, 0, 0))

    @classmethod
    def load(cls, data: BytesIO) -> 'CNK0':
        data.seek(0, SEEK_END)
        logger.info(f"Loading CNK0 file of length {data.tell()}...")
        data.seek(0, SEEK_SET)
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
        return cls(header, tiles, unk1, unkArr1, indices, vertices, render_batches, optimized_draw, unk_shorts, unk_vectors, tile_occluder_info, [], [], [], ())

@dataclass
class Texture:
    color_nx_map: Image.Image
    spec_ny_map: Image.Image
    extra1: bytes
    extra2: bytes
    extra3: bytes
    extra4: bytes

    @classmethod
    def load(cls, data: BytesIO) -> 'Texture':
        color_len = struct.unpack("<I", data.read(4))[0]
        logger.debug(f"Reading color texture of length {color_len}")
        color_nx_map = Image.open(BytesIO(data.read(color_len)))
        
        spec_len = struct.unpack("<I", data.read(4))[0]
        logger.debug(f"Reading specular texture of length {spec_len}")
        spec_ny_map = Image.open(BytesIO(data.read(spec_len)))
        
        extra1_len = struct.unpack("<I", data.read(4))[0]
        logger.debug(f"Reading extra data of length {extra1_len}")
        extra1 = data.read(extra1_len)
        
        extra2_len = struct.unpack("<I", data.read(4))[0]
        logger.debug(f"Reading extra data of length {extra2_len}")
        extra2 = data.read(extra2_len)

        extra3_len = struct.unpack("<I", data.read(4))[0]
        logger.debug(f"Reading extra data of length {extra3_len}")
        extra3 = data.read(extra3_len)

        extra4_len = struct.unpack("<I", data.read(4))[0]
        logger.debug(f"Reading extra data of length {extra4_len}")
        extra4 = data.read(extra4_len)
        
        return cls(color_nx_map, spec_ny_map, extra1, extra2, extra3, extra4)

    def __repr__(self) -> str:
        return f"Texture([{self.color_nx_map[:16]}{'...' if len(self.color_nx_map) > 16 else ''}], [{self.spec_ny_map[:16]}{'...' if len(self.spec_ny_map) > 16 else ''}], [{self.extra1[:16]}{'...' if len(self.extra1) > 16 else ''}], [{self.extra2[:16]}{'...' if len(self.extra2) > 16 else ''}], [{self.extra3[:16]}{'...' if len(self.extra3) > 16 else ''}], [{self.extra4[:16]}{'...' if len(self.extra4) > 16 else ''}])"

@dataclass
class CNK1:
    header: Header
    textures: List[Texture]

    @classmethod
    def load(cls, data: BytesIO) -> 'CNK1':
        logger.info("Loading CNK1 file...")
        header = Header.load(data)
        if header.magic != b'CNK1':
            raise ValueError("Not a CNK1 file!")
        textures_len = struct.unpack("<I", data.read(4))[0]
        logger.info(f"Loading {textures_len} textures...")
        textures = [Texture.load(data) for _ in range(textures_len)]
        logger.info(f"Loaded {len(textures)} textures")
        logger.info("CNK1 file loaded")
        return cls(header, textures)

    def color(self) -> Image.Image:
        combined_color = Image.new("RGB", (self.textures[0].color_nx_map.width * 2, self.textures[0].color_nx_map.height * 2))
        width = self.textures[0].color_nx_map.width
        height = self.textures[0].color_nx_map.height
        combined_color.paste(self.textures[0].color_nx_map, (0, 0))
        combined_color.paste(self.textures[1].color_nx_map, (width, 0))
        combined_color.paste(self.textures[2].color_nx_map, (0, height))
        combined_color.paste(self.textures[3].color_nx_map, (width, height))
        return combined_color
    
    def specular(self) -> Image.Image:
        temp = Image.new("RGB", (self.textures[0].spec_ny_map.width * 2, self.textures[0].spec_ny_map.height * 2))
        width = self.textures[0].spec_ny_map.width
        height = self.textures[0].spec_ny_map.height
        temp.paste(self.textures[0].spec_ny_map, (0, 0))
        temp.paste(self.textures[1].spec_ny_map, (width, 0))
        temp.paste(self.textures[2].spec_ny_map, (0, height))
        temp.paste(self.textures[3].spec_ny_map, (width, height))
        r, g, b = temp.split()
        g = ImageChops.invert(g)
        b = ImageChops.invert(b)
        combined_specular = Image.merge("RGBA", (r, g, b, r))
        return combined_specular
    
    def normal(self) -> Image.Image:
        width = self.textures[0].spec_ny_map.width
        height = self.textures[0].spec_ny_map.height
        x, y, z = Image.new("RGB", (width * 2, height * 2), (0, 0, 255)).split()
        x.paste(self.textures[0].color_nx_map.getchannel("A"), (0, 0))
        x.paste(self.textures[1].color_nx_map.getchannel("A"), (width, 0))
        x.paste(self.textures[2].color_nx_map.getchannel("A"), (0, height))
        x.paste(self.textures[3].color_nx_map.getchannel("A"), (width, height))
        y.paste(self.textures[0].spec_ny_map.getchannel("A"), (0, 0))
        y.paste(self.textures[1].spec_ny_map.getchannel("A"), (width, 0))
        y.paste(self.textures[2].spec_ny_map.getchannel("A"), (0, height))
        y.paste(self.textures[3].spec_ny_map.getchannel("A"), (width, height))
        return Image.merge("RGB", (x, y, z))
        
    
@dataclass
class ForgelightChunk:
    chunk: Union[CNK0, CNK1]

    @classmethod
    def decompress(cls, data: BytesIO) -> BytesIO:
        header = Header.load(data)
        logger.debug("Decompressing chunk data...")
        decompressor = Decompressor()
        logger.debug(f"Decompressor lib version: {decompressor.version()}")
        decompressed_size, compressed_size = struct.unpack("<II", data.read(8))
        compressed_data = data.read()
        logger.debug(f"Decompressed size: {decompressed_size}")
        logger.debug(f"Compressed size: {compressed_size}")
        if len(compressed_data) != compressed_size:
            logger.warning(f"Compressed data length {len(compressed_data)} != file value {compressed_size}")
        decompressed_data = decompressor.decompress(compressed_data, decompressed_size)
        if len(decompressed_data) != decompressed_size:
            logger.warning("Decompressed data length different from listed size!")
        return BytesIO(header.serialize() + decompressed_data)

    @classmethod
    def load(cls, data: BytesIO, compressed: bool = True) -> Union[CNK0, CNK1]:
        if compressed:
            data = cls.decompress(data)
        header = Header.load(data)
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