import csv
from datetime import datetime
from threading import Thread
import glfw
import OpenGL.GL as gl
from OpenGLContext.texture import Texture

from io import BytesIO
import logging
from typing import Dict, List, Tuple, Union
import dearpygui.dearpygui as dpg
from pathlib import Path
from glob import glob
from multiprocessing import Event, Pool
from export_manager import ExportManager
from DbgPack import Asset2, Pack1, Pack2
from functools import cmp_to_key
from PIL import Image, UnidentifiedImageError
from PIL.Image import Transpose
import magic
import imgui
import numpy
from imgui.integrations.glfw import GlfwRenderer
from imgui_extensions import FilePicker
from dme_loader import DME
from dme_loader.data_classes import input_layout_formats
from adr_converter import dme_from_adr, to_glb, to_gltf
import re

logger = logging.getLogger("Pack Inspector")

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    fmt="[%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
))

server = "C:/Users/Public/Daybreak Game Company/Installed Games/PlanetSide 2 Test/Resources/Assets/"

paths = ([]
    + [Path(p) for p in glob(server + "assets_x64*.pack2")]
    + [Path(p) for p in glob(server + "data_x64*.pack2")]
    + [Path(p) for p in glob(server + "Nexus_x64*.pack2")]
    + [Path(p) for p in glob(server + "ui_x64*.pack2")]
)

def load_manager():
    pool = Pool(8)
    return ExportManager(paths, p=pool)

def populate_names(sender, app_data, user_data):
    print("Test")

def preview_item(asset: Asset2) -> Union[Image.Image, DME, str, None]:
    if asset.name.endswith("txt") or asset.name.endswith("xml") or asset.name.endswith("adr") or asset.name == "{NAMELIST}":
        return str(asset.get_data(), encoding="utf-8")
    if asset.name.endswith("dme"):
        return DME.load(BytesIO(asset.get_data()))
    identifier = magic.from_buffer(asset.get_data())
    if "Microsoft DirectDraw Surface" or "PNG" in identifier:
        base = Image.new("RGBA", (800, 800))
        try:
            img = Image.open(BytesIO(asset.get_data()))
        except UnidentifiedImageError:
            return
        if img.size[0] > 800 or img.size[1] > 800:
            img.thumbnail((800, 800))
        base.paste(img, (int((base.size[0] - img.size[0]) / 2), int((base.size[1] - img.size[1]) / 2)))
        return base


def pack_sort(a: Union[Pack1, Pack2], b: Union[Pack1, Pack2]):
    name, _, number = zip(a.name.split("_"), b.name.split("_"))
    return -1 if name[0].lower() < name[1].lower() else 1 if name[0].lower() > name[1].lower() else int(number[0]) - int(number[1])

def render_dme_to_texture(model: DME, framebuffer: int):
    input_layout = model.input_layout(0)
    if input_layout is None:
        logger.error("Model does not have a material")
        return
    for vertex_stream in model.meshes[0].vertex_streams:
        vertex_stream.rewind()
    
    VAO = gl.glGenVertexArrays(1)
    VBO = gl.glGenBuffers(1)
    EBO = gl.glGenBuffers(1)
    return
    
    for entry in input_layout.entries:
        format, size = input_layout_formats[entry.type]

    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)
    gl.glViewport(0, 0, 800, 800)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

def update_namelist(namelist_path, manager: ExportManager) -> Thread:
    def work():
        with open(namelist_path) as namelist:
            namelist_data = namelist.read().split()
            for pack in manager.packs:
                pack: Pack2
                pack.namelist = namelist_data
    return Thread(target=work)

def export_selected(path: str, data: Tuple[ExportManager, str]) -> Thread:
    def work():
        manager = data[0]
        to_export = data[1]
        logger.info(f"Exporting {to_export} to {path}...")
        if to_export.lower().endswith(".adr") and (path.lower().endswith(".glb") or path.lower().endswith(".gltf")):
            dme = dme_from_adr(manager, to_export)
            if path.lower().endswith(".glb"):
                to_glb(dme, path, manager, dme.name)
            elif path.lower().endswith(".gltf"):
                to_gltf(dme, path, manager, dme.name)
        else:
            with open(path, "wb") as output:
                output.write(manager[to_export].get_data())
    return Thread(target=work)

def main():
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    manager = load_manager()
    imgui.create_context()
    window = impl_glfw_init(1280, 840, "Pack Inspector")
    impl = GlfwRenderer(window)
    preview_texture = Texture(Image.new("RGBA", (800, 800), (0, 0, 0, 255)))
    preview_text = None
    preview_model = None

    # TODO: Move this stuff to a class wrapping DME for rendering
    framebuffer = gl.glGenFramebuffers(1)
    render_texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, render_texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, 800, 800, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, [])
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    render_buffer = gl.glGenRenderbuffers(1)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, render_buffer)
    gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, 800, 800)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)
    gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, render_buffer)
    gl.glFramebufferTexture(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, render_texture, 0)
    gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, 0)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    # ###

    name_filter = re.compile(".*", re.IGNORECASE)
    matched_names: Dict[str, List[str]] = {}
    file_picker = None
    file_picker_callback = None
    file_picker_thread = None
    selected = None
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", enabled=True):
                clicked, visible = imgui.menu_item("Load namelist...")
                if clicked:
                    file_picker = FilePicker("Select namelist", user_data=manager)
                    file_picker_callback = update_namelist
                clicked, visible = imgui.menu_item("Export selected")
                if clicked:
                    file_picker_thread = export_selected(str(Path(".") / "export" / selected), (manager, selected))
                    file_picker_thread.start()

                clicked, visible = imgui.menu_item("Export selected as...")
                if clicked:
                    file_picker = FilePicker(f"Exporting {selected}...", user_data=(manager, selected))
                    file_picker_callback = export_selected
                imgui.end_menu()
            imgui.end_main_menu_bar()
        
        imgui.set_next_window_size(imgui.get_io().display_size[0], imgui.get_io().display_size[1] - imgui.get_frame_height())
        imgui.begin("Pack Inspector", True, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)
        imgui.set_window_position(0, imgui.get_frame_height())
        imgui.begin_child("Controls", 0, 25)
        imgui.text("Filter:")
        imgui.same_line(spacing=10)
        imgui.push_item_width(imgui.get_window_width() * 0.25)
        changed, text_val = imgui.input_text("##filter", "", 64)
        imgui.same_line(spacing=20)
        imgui.text(f"Selected: {selected}")
        imgui.end_child()
        if changed and text_val == "":
            name_filter = re.compile(".*", re.IGNORECASE)
        elif changed:
            try:
                new_filter = re.compile(text_val, re.IGNORECASE)
                name_filter = new_filter
                matched_names = {}
            except re.error:
                pass

        imgui.begin_child("Pack contents", int(0.3 * imgui.get_io().display_size[0]), 0)
        if not manager.loaded.is_set():
            imgui.text(f"Loading packs{'.' * (int(datetime.now().timestamp() * 10) % 4)}")
        else:
            for pack in sorted(manager.packs, key=cmp_to_key(pack_sort)):
                expanded, visible = imgui.collapsing_header(pack.name)
                if expanded and pack.name not in matched_names:
                    matched_names[pack.name] = []
                    for asset in pack:
                        if asset.name == '':
                            continue
                        if not name_filter.match(asset.name):
                            continue
                        matched_names[pack.name].append(asset.name)
                
                if expanded and pack.name in matched_names:
                    for name in matched_names[pack.name]:
                        imgui.text(name)
                        if imgui.is_item_clicked():
                            selected = name
                            to_preview = preview_item(manager[name])
                            if type(to_preview) == Image.Image:
                                preview_texture.update((0, 0), (800, 800), to_preview.tobytes())
                                preview_text = None
                                preview_model = None
                            elif type(to_preview) == str:
                                if to_preview.startswith("#*"):
                                    reader = csv.reader(to_preview.splitlines(), delimiter="^")
                                    preview_text = list(reader)
                                else:
                                    preview_text = to_preview
                                preview_model = None
                            elif type(to_preview) == DME:
                                render_dme_to_texture(to_preview, framebuffer)
                                preview_model = to_preview
                                preview_text = None
                
        imgui.end_child()
        imgui.same_line()
        imgui.begin_child("Preview")
        if preview_text is not None and type(preview_text) == str:
            imgui.text_unformatted(preview_text)
        elif preview_text is not None and type(preview_text) == list:
            header = preview_text[0]
            imgui.columns(len(header))
            imgui.separator()
            for line in preview_text:
                if len(line) > 14 and header[14] == "MODEL_NAME" and line[14] == "":
                    continue
                for i, item in enumerate(line):
                    imgui.text(item)
                    imgui.next_column()
                imgui.separator()
        elif preview_model is not None:
            imgui.image(render_texture, 800, 800)
        else:
            imgui.image(preview_texture.texture, 800, 800)
        imgui.end_child()
        
        imgui.end()

        if file_picker:
            file_picker.tick()
            if not file_picker.active:
                if file_picker.selected is not None:
                    file_picker_thread = file_picker_callback(file_picker.selected, file_picker.user_data)
                    file_picker_thread.start()
                    file_picker_callback = None
                file_picker = None
        
        if file_picker_thread and not file_picker_thread.is_alive():
            file_picker_thread = None
            manager.refresh_assets()

        gl.glClearColor(0, 0, 0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)
    
    impl.shutdown()
    glfw.terminate()


def impl_glfw_init(width: int, height: int, window_name: str):
    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(
        int(width), int(height), window_name, None, None
    )
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window
    

if __name__ == "__main__":
    main()