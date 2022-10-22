from datetime import datetime
import glfw
import OpenGL.GL as gl
from OpenGLContext.texture import Texture

from io import BytesIO
import logging
from typing import Union
import dearpygui.dearpygui as dpg
from pathlib import Path
from glob import glob
from multiprocessing import Pool
from export_manager import ExportManager
from DbgPack import Asset2, Pack1, Pack2
from functools import cmp_to_key
from PIL import Image
from PIL.Image import Transpose
import magic
import imgui
import numpy
from imgui.integrations.glfw import GlfwRenderer
from dme_loader import DME
from dme_loader.data_classes import input_layout_formats
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
        img = Image.open(BytesIO(asset.get_data()))
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

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()
        imgui.set_next_window_size(*imgui.get_io().display_size)
        imgui.begin("", True, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)
        imgui.set_window_position(0, 0)
        imgui.begin_child("Pack contents", 300, 0)
        if not manager.loaded.is_set():
            imgui.text(f"Loading packs{'.' * (int(datetime.now().timestamp() * 10) % 4)}")
        else:
            for pack in sorted(manager.packs, key=cmp_to_key(pack_sort)):
                expanded, visible = imgui.collapsing_header(pack.name)
                if expanded:
                    for asset in pack:
                        if asset.name == '':
                            continue
                        imgui.text(asset.name)
                        if imgui.is_item_clicked():
                            to_preview = preview_item(asset)
                            if type(to_preview) == Image.Image:
                                preview_texture.update((0, 0), (800, 800), to_preview.tobytes())
                                preview_text = None
                                preview_model = None
                            elif type(to_preview) == str:
                                preview_text = to_preview
                                preview_model = None
                            elif type(to_preview) == DME:
                                render_dme_to_texture(to_preview, framebuffer)
                                print(to_preview.meshes[0].skin_indices[:64])
                                print(to_preview.meshes[0].skin_weights[:64])
                                preview_model = to_preview
                                preview_text = None

        imgui.end_child()
        imgui.same_line()
        imgui.begin_child("Preview")
        if preview_text is not None:
            imgui.text_unformatted(preview_text)
        elif preview_model is not None:
            imgui.image(render_texture, 800, 800)
        else:
            imgui.image(preview_texture.texture, 800, 800)
        imgui.end_child()
        imgui.end()

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
    
def dpg_main(manager: ExportManager):
    dpg.create_context()
    dpg.create_viewport(title="Pack Inspector", height=832)
    dpg.setup_dearpygui()

    texture_data = BytesIO()
    Image.new("RGBA", (800, 800), (0, 0, 0, 255)).save("temp.png")
    width, height, channels, data = dpg.load_image("temp.png")

    with dpg.texture_registry(show=True):
        dpg.add_dynamic_texture(width=width, height=height, default_value=data, tag="preview_texture")

    with dpg.window(no_move=True, no_resize=True, no_title_bar=True, width=1280, height=800):
        with dpg.child_window(width=320, tracked=True, tag="file_list"):
            if len(manager) == 0:
                dpg.add_text("Loading...", tag="loading_text")
            for name in manager:
                dpg.add_text(name)
        with dpg.child_window(pos=[321, 0], width=1280-320):
            dpg.add_image("preview_texture", width=width, height=height, pos = [(1280-320-800) / 2, 0])
    
        
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        if manager and manager.loaded.is_set() and manager.pool is not None:
            manager.pool.close()
            manager.pool = None
            logger.info("Assets loaded.")
            dpg.delete_item("loading_text")
            def pack_sort(a: Union[Pack1, Pack2], b: Union[Pack1, Pack2]):
                name, _, number = zip(a.name.split("_"), b.name.split("_"))
                return -1 if name[0].lower() < name[1].lower() else 1 if name[0].lower() > name[1].lower() else int(number[0]) - int(number[1])
            for pack in sorted(manager.packs, key=cmp_to_key(pack_sort)):
                with dpg.collapsing_header(label=pack.name, default_open=False, parent="file_list", user_data=pack):
                    for item in sorted(list(pack), key=lambda x: x.name):
                        if item.name == '':
                            continue
                        dpg.add_button(label="\t" + item.name, callback=preview_item, user_data=item)


        dpg.render_dearpygui_frame()
    

if __name__ == "__main__":
    main()