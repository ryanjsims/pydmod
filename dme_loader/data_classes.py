from typing import Dict, List
from io import BytesIO
from enum import Enum

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
 
class InputLayout:
    def __init__(self, name: str, entries: List[InputLayoutEntry]):
        self.name = name
        self.entries = entries

    @classmethod
    def from_json(cls, data: Dict) -> 'InputLayout':
        name = data["name"]
        entries = [InputLayoutEntry.from_json(entry) for entry in data["entries"]]
        return cls(name, entries)

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