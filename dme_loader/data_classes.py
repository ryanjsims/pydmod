from typing import Dict, List, Optional
from io import BytesIO, SEEK_END
from enum import Enum
from . import jenkins

input_layout_formats ={
    "Float3":       ("<fff", 12),
    "D3dcolor":     ("<I", 4),
    "Float2":       ("<ff", 8),
    "Float4":       ("<ffff", 16),
    "ubyte4n":      ("<cccc", 4),
    "Float16_2":    ("<ee", 4),
    "float16_2":    ("<ee", 4),
    "Short2":       ("<HH", 4),
    "Float1":       ("<f", 4),
    "Short4":       ("<HHHH", 8)
}

class LayoutUsage(str, Enum):
    POSITION = "Position"
    NORMAL = "Normal"
    BINORMAL = "Binormal"
    TANGENT = "Tangent"
    BLENDWEIGHT = "BlendWeight"
    BLENDINDICES = "BlendIndices"
    TEXCOORD = "Texcoord"
    COLOR = "Color"

class VertexStream:
    def __init__(self, stride: int, data: bytes):
        self.stride = stride
        self.data = BytesIO(data)
    
    def __len__(self) -> int:
        pos = self.data.tell()
        self.data.seek(0, SEEK_END)
        length = self.data.tell()
        self.data.seek(pos)
        return length

    def tell(self) -> int:
        return self.data.tell()

class InputLayoutEntry:
    def __init__(self, stream: int, _type: str, usage: LayoutUsage, usage_index: int):
        self.stream = stream
        self.type = _type
        self.usage = usage
        self.usage_index = usage_index
    
    @classmethod
    def from_json(cls, data: Dict) -> 'InputLayoutEntry':
        stream = data["stream"]
        _type = data["type"]
        usage = LayoutUsage(data["usage"])
        usage_index = data["usageIndex"]
        return cls(stream, _type, usage, usage_index)
    
    def __repr__(self) -> str:
        return f"InputLayoutEntry({self.stream}, {self.type}, {self.usage}, {self.usage_index})"
 
class InputLayout:
    def __init__(self, name: str, name_hash: int, entries: List[InputLayoutEntry], sizes: Optional[List[int]]):
        self.name = name
        self.entries = entries
        self.sizes = sizes
        self.__name_hash = name_hash

    def __hash__(self) -> int:
        return self.__name_hash

    @classmethod
    def from_json(cls, data: Dict, hash: Optional[str] = None) -> 'InputLayout':
        name: str = data["name"]
        if "hash" in data:
            name_hash = data["hash"]
        elif hash is not None:
            name_hash = int(hash)
        else:
            name_hash = jenkins.oaat(name.encode("utf-8"))
        entries = [InputLayoutEntry.from_json(entry) for entry in data["entries"]]
        if "sizes" in data:
            sizes = [data["sizes"][str(i)] for i in range(len(data["sizes"]))]
        else:
            temp_sizes = {}
            for i in range(len(entries)):
                if entries[i].stream not in temp_sizes:
                    temp_sizes[entries[i].stream] = 0
                temp_sizes[entries[i].stream] += input_layout_formats[entries[i].type][1]
            sizes = [temp_sizes[i] for i in range(len(temp_sizes))]

        return cls(name, name_hash, entries, sizes)

class DrawStyle:
    def __init__(self, name: str, hash: int, input_layout: str):
        self.name = name
        self.hash = hash
        self.input_layout = input_layout
    
    @classmethod
    def from_json(cls, data: Dict):
        name = data["name"]
        hash = data["hash"]
        input_layout = data["inputLayout"]
        return cls(name, hash, input_layout)

class MaterialDefinition:
    def __init__(self, name: str, hash: int, draw_styles: List[DrawStyle]):
        self.name = name
        self.hash = hash
        self.draw_styles = draw_styles
    
    @classmethod
    def from_json(cls, data: Dict) -> 'MaterialDefinition':
        name = data["name"]
        hash = data["hash"]
        draw_styles = [DrawStyle.from_json(draw_style) for draw_style in data["drawStyles"]]
        return cls(name, hash, draw_styles)