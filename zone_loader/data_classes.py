from enum import IntEnum
import struct
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import List, Tuple

logger = logging.getLogger("zone_loader")

@dataclass
class Color:
    r: int
    g: int
    b: int
    a: int

    @classmethod
    def load(cls, data: BytesIO, argb: bool = False) -> 'Color':
        values = struct.unpack("<BBBB", data.read(4))
        if not argb:
            return cls(*values)
        else:
            return cls(values[1], values[2], values[3], values[0])

    def serialize(self, argb: bool = False) -> bytes:
        if not argb:
            return struct.pack("<cccc", self.r, self.g, self.b, self.a)
        else:
            return struct.pack("<cccc", self.a, self.r, self.g, self.b)

@dataclass
class Float4:
    x: float
    y: float
    z: float
    w: float

    @classmethod
    def load(cls, data: BytesIO) -> 'Float4':
        values = struct.unpack("<ffff", data.read(16))
        return cls(*values)

@dataclass
class Offsets:
    ecos: int
    floras: int
    invis_walls: int
    objects: int
    lights: int
    unknowns: int
    decals: int

    @classmethod
    def load(cls, data: BytesIO, version: int) -> 'Offsets':
        ecos, floras, invis_walls, objects, lights, unknown0 = struct.unpack("<IIIIII", data.read(24))
        if version in (4, 5):
            decals = struct.unpack("<I", data.read(4))[0]
        else:
            decals = None
        return cls(ecos, floras, invis_walls, objects, lights, unknown0, decals)

@dataclass
class Header:
    magic: bytes
    version: int
    unknown: int
    offsets: Offsets
    quads_per_tile: int
    tile_size: float
    tile_height: float
    verts_per_tile: int
    tiles_per_chunk: int
    start_x: int
    start_y: int
    chunks_x: int
    chunks_y: int

    @classmethod
    def load(cls, data: BytesIO) -> 'Header':
        magic = data.read(4)
        assert magic == b'ZONE'

        version = struct.unpack("<I", data.read(4))[0]
        if version in (4, 5):
            unknown = struct.unpack("<I", data.read(4))[0]
        else:
            unknown = None
        offsets = Offsets.load(data, version)

        quads_per_tile, tile_size, tile_height, verts_per_tile = struct.unpack("<IffI", data.read(16))
        tiles_per_chunk, start_x, start_y, chunks_x, chunks_y = struct.unpack("<IiiII", data.read(20))
        return cls(
            magic, version, unknown,
            offsets, quads_per_tile,
            tile_size, tile_height,
            verts_per_tile, tiles_per_chunk,
            start_x, start_y,
            chunks_x, chunks_y
        )

def read_cstr(data: BytesIO) -> str:
    value = data.read(1)
    while value[-1] != 0:
        value += data.read(1)
    try:
        string = str(value.strip(b'\0'), encoding='utf-8')
    except UnicodeDecodeError as e:
        display = value.strip(b'\0')
        logger.error(f"Failed to decode bytes {display}")
        raise e
    return string

@dataclass
class TexturePart:
    name: str
    color_nx_map: str
    spec_blend_ny_map: str
    detail_repeat: int
    blend_strength: float
    spec_min: float
    spec_max: float
    spec_smoothness_min: float
    spec_smoothness_max: float
    physics_material: str

    @classmethod
    def load(cls, data: BytesIO) -> 'TexturePart':
        name = read_cstr(data)
        color_nx_map = read_cstr(data)
        spec_blend_ny_map = read_cstr(data)
        detail_repeat, blend_strength, spec_min, spec_max = struct.unpack("<Ifff", data.read(16))
        spec_smoothness_min, spec_smoothness_max = struct.unpack("<ff", data.read(8))
        physics_material = read_cstr(data)
        return cls(
            name, color_nx_map, spec_blend_ny_map, 
            detail_repeat, blend_strength, 
            spec_min, spec_max, 
            spec_smoothness_min, spec_smoothness_max, 
            physics_material
        )

@dataclass
class EcoTint:
    color_rgba: Color
    strength: int

    @classmethod
    def load(cls, data: BytesIO) -> 'EcoTint':
        color_rgba = Color.load(data)
        strength = struct.unpack("<I", data.read(4))[0]
        return cls(color_rgba, strength)

@dataclass
class EcoLayer:
    density: float
    min_scale: float
    max_scale: float
    slope_peak: float
    slope_extent: float
    min_elevation: float
    max_elevation: float
    min_alpha: int
    flora_name: str
    tints: List[EcoTint]

    @classmethod
    def load(cls, data: BytesIO) -> 'EcoLayer':
        density, min_scale, max_scale, slope_peak, slope_extent = struct.unpack("<fffff", data.read(20))
        min_elevation, max_elevation, min_alpha = struct.unpack("<ffc", data.read(9))
        flora_name = read_cstr(data)
        tint_count = struct.unpack("<I", data.read(4))[0]
        tints: List[EcoTint] = [EcoTint.load(data) for _ in range(tint_count)]
        return cls(density, min_scale, max_scale, slope_peak, slope_extent, min_elevation, max_elevation, min_alpha, flora_name, tints)
    
@dataclass
class FloraPart:
    layers: List[EcoLayer]

    @classmethod
    def load(cls, data: BytesIO) -> 'FloraPart':
        layer_count = struct.unpack("<I", data.read(4))[0]
        layers = [EcoLayer.load(data) for _ in range(layer_count)]
        return cls(layers)
    
@dataclass
class Eco:
    index: int
    texture_part: TexturePart
    flora_part: FloraPart
    
    @classmethod
    def load(cls, data: BytesIO) -> 'Eco':
        index = struct.unpack("<I", data.read(4))[0]
        texture_part = TexturePart.load(data)
        flora_part = FloraPart.load(data)
        return cls(index, texture_part, flora_part)

@dataclass
class Ecos:
    ecos: List[Eco]

    @classmethod
    def load(cls, data: BytesIO) -> 'Ecos':
        eco_count = struct.unpack("<I", data.read(4))[0]
        logger.info(f"Loading {eco_count} ecos...")
        ecos = [Eco.load(data) for _ in range(eco_count)]
        return cls(ecos)
    
    def __iter__(self):
        return iter(self.ecos)

    def __str__(self) -> str:
        return str(self.ecos)
    
    def __repr__(self) -> str:
        return repr(self.ecos)
    
    def __len__(self):
        return len(self.ecos)
    
    def __getitem__(self, index) -> Eco:
        return self.ecos[index]

@dataclass
class Flora:
    name: str
    texture: str
    model: str
    unk_bool: bool
    unk_float0: float
    unk_float1: float
    unk_data: bytes

    @classmethod
    def load(cls, data: BytesIO, version: int) -> 'Flora':
        name = read_cstr(data)
        texture = read_cstr(data)
        model = read_cstr(data)
        unk_bool, unk_float0, unk_float1 = struct.unpack("<Bff", data.read(9))
        if version in (4, 5):
            unk_data = data.read(12)
        else:
            unk_data = None
        return cls(name, texture, model, unk_bool, unk_float0, unk_float1, unk_data)

@dataclass
class Floras:
    floras: List[Flora]

    @classmethod
    def load(cls, data: BytesIO, version: int) -> 'Floras':
        flora_count = struct.unpack("<I", data.read(4))[0]
        logger.info(f"Loading {flora_count} floras...")
        floras = [Flora.load(data, version) for _ in range(flora_count)]
        return cls(floras)
    
    def __str__(self) -> str:
        return str(self.floras)
    
    def __repr__(self) -> str:
        return repr(self.floras)

    def __iter__(self):
        return iter(self.floras)
    
    def __len__(self):
        return len(self.floras)
    
    def __getitem__(self, index):
        return self.floras[index]

@dataclass
class InvisWall:
    
    @classmethod
    def load(cls, data: BytesIO) -> 'InvisWall':
        raise NotImplementedError("No binary template for invis walls")

@dataclass
class InvisWalls:
    walls: List[InvisWall]

    @classmethod
    def load(cls, data: BytesIO) -> 'InvisWalls':
        count = struct.unpack("<I", data.read(4))[0]
        logger.info(f"Loading {count} invisible walls...")
        walls = [InvisWall.load(data) for _ in range(count)]
        return cls(walls)

    def __str__(self) -> str:
        return str(self.walls)
    
    def __repr__(self) -> str:
        return repr(self.walls)

    def __iter__(self):
        return iter(self.walls)
    
    def __len__(self):
        return len(self.walls)
    
    def __getitem__(self, index):
        return self.walls[index]

@dataclass
class UIntPair:
    key: int
    value: int

    @classmethod
    def load(cls, data: BytesIO) -> 'UIntPair':
        key, value = struct.unpack("<II", data.read(8))
        return cls(key, value)

@dataclass
class FloatPair:
    key: int
    value: float

    @classmethod
    def load(cls, data: BytesIO) -> 'FloatPair':
        key, value = struct.unpack("<If", data.read(8))
        return cls(key, value)

@dataclass
class Vector4Pair:
    key: int
    value: Float4

    @classmethod
    def load(cls, data: BytesIO) -> 'Vector4Pair':
        key = struct.unpack("<I", data.read(4))[0]
        value = Float4.load(data)
        return cls(key, value)

@dataclass
class Instance:
    translation: Float4
    rotation: Float4
    scale: Float4
    unk_data: bytes
    unk_float: float
    uint_pairs: List[UIntPair]
    float_pairs: List[FloatPair]
    unk_int: int
    vector4_pairs: List[Vector4Pair]
    unk_data2: bytes
    unk_byte: int
    unk_byte2: int
    
    @classmethod
    def load(cls, data: BytesIO, version: int) -> 'Instance':
        translation = Float4.load(data)
        rotation = Float4.load(data)
        scale = Float4.load(data)
        if version == 4:
            unk_data = data.read(29)
            unk_float = None
            uint_pairs = None
            float_pairs = None
            unk_int = None
            vector4_pairs = None
            unk_data2 = None
            unk_byte = None
            unk_byte2 = None
        elif version == 5:
            unk_data = data.read(5)
            unk_float, uint_pair_count = struct.unpack("<fI", data.read(8))
            uint_pairs = [UIntPair.load(data) for _ in range(uint_pair_count)]
            float_pair_count = struct.unpack("<I", data.read(4))[0]
            float_pairs = [FloatPair.load(data) for _ in range(float_pair_count)]
            unk_int, vector4_pair_count = struct.unpack("<II", data.read(8))[0]
            vector4_pairs = [Vector4Pair.load(data) for _ in range(vector4_pair_count)]
            unk_data2 = data.read(5)
            unk_byte = None
            unk_byte2 = None
        elif version == 2:
            unk_data = None
            uint_pairs = None
            float_pairs = None
            vector4_pairs = None
            unk_data2 = None
            unk_int, unk_byte, unk_byte2, unk_float = struct.unpack("<IBBf", data.read(10))
        else:
            unk_data = None
            uint_pairs = None
            float_pairs = None
            vector4_pairs = None
            unk_data2 = None
            unk_int, unk_byte, unk_float = struct.unpack("<IBf", data.read(9))
            unk_byte2 = None
        
        return cls(translation, rotation, scale, unk_data, unk_float, uint_pairs, float_pairs, unk_int, vector4_pairs, unk_data2, unk_byte, unk_byte2)
        
@dataclass
class RuntimeObject:
    actor_file: str
    unk_float: float
    instances: List[Instance]

    @classmethod
    def load(cls, data: BytesIO, version: int) -> 'RuntimeObject':
        actor_file = read_cstr(data)
        logger.debug(f"Actor file: {actor_file}")
        unk_float, instance_count = struct.unpack("<fI", data.read(8))
        instances = [Instance.load(data, version) for _ in range(instance_count)]
        return cls(actor_file, unk_float, instances)

@dataclass
class RuntimeObjects:
    runtime_objects: List[RuntimeObject]

    @classmethod
    def load(cls, data: BytesIO, version: int) -> 'RuntimeObjects':
        object_count = struct.unpack("<I", data.read(4))[0]
        logger.info(f"Loading {object_count} runtime objects...")
        runtime_objects = [RuntimeObject.load(data, version) for _ in range(object_count)]
        return cls(runtime_objects)

    def __str__(self) -> str:
        return str(self.runtime_objects)
    
    def __repr__(self) -> str:
        return repr(self.runtime_objects)

    def __iter__(self):
        return iter(self.runtime_objects)

    def __len__(self):
        return len(self.runtime_objects)
    
    def __getitem__(self, index):
        return self.runtime_objects[index]

class LightType(IntEnum):
    Point = 1
    Spot = 2

@dataclass
class Light:
    name: str
    color_name: str
    type: LightType
    unknown_bool: bool
    translation: Float4
    rotation: Float4
    unk_floats: Tuple[float, float]
    color_val: Color
    unk_chunk: bytes

    @classmethod
    def load(cls, data: BytesIO) -> 'Light':
        name = read_cstr(data)
        color_name = read_cstr(data)
        type = LightType(struct.unpack("<I", data.read(4))[0])
        unknown_bool = struct.unpack("<B", data.read(1))[0]
        translation = Float4.load(data)
        rotation = Float4.load(data)
        unk_floats = struct.unpack("<ff", data.read(8))
        color_val = Color.load(data, argb=True)
        unk_chunk = data.read(26)
        return cls(name, color_name, type, unknown_bool, translation, rotation, unk_floats, color_val, unk_chunk)

@dataclass
class Lights:
    lights: List[Light]

    @classmethod
    def load(cls, data: BytesIO) -> 'Lights':
        light_count = struct.unpack("<I", data.read(4))[0]
        lights = [Light.load(data) for _ in range(light_count)]
        return cls(lights)
    
    def __str__(self) -> str:
        return str(self.lights)
    
    def __repr__(self) -> str:
        return repr(self.lights)

    def __iter__(self):
        return iter(self.lights)

    def __len__(self):
        return len(self.lights)
    
    def __getitem__(self, index):
        return self.lights[index]

@dataclass
class Zone:
    header: Header
    ecos: Ecos
    floras: Floras
    invis_walls: InvisWalls
    objects: RuntimeObjects
    lights: Lights
    unknown_data: bytes

    @classmethod
    def load(cls, data: BytesIO) -> 'Zone':
        logger.info("Loading Zone file...")
        header = Header.load(data)
        logger.info(f"Zone {header}")
        assert data.tell() == header.offsets.ecos
        ecos = Ecos.load(data)
        assert data.tell() == header.offsets.floras
        floras = Floras.load(data, header.version)
        assert data.tell() == header.offsets.invis_walls
        invis_walls = InvisWalls.load(data)
        assert data.tell() == header.offsets.objects
        objects = RuntimeObjects.load(data, header.version)
        assert data.tell() == header.offsets.lights
        lights = Lights.load(data)
        assert data.tell() == header.offsets.unknowns
        unknown_data = data.read()
        logger.info("Zone file loaded!")
        return cls(header, ecos, floras, invis_walls, objects, lights, unknown_data)