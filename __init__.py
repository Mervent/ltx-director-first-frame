from .ltx_director import LTXDirectorFirstFrame
from .ltx_director_guide import LTXDirectorGuide, LTXDirectorCropGuides
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


class LTXDirectorFirstFrameExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LTXDirectorFirstFrame]


async def comfy_entrypoint() -> LTXDirectorFirstFrameExtension:
    return LTXDirectorFirstFrameExtension()


NODE_CLASS_MAPPINGS = {
    "LTXDirectorFirstFrameGuide": LTXDirectorGuide,
    "LTXDirectorFirstFrameCropGuides": LTXDirectorCropGuides,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXDirectorFirstFrameGuide": "LTX Director Guide (First Frame)",
    "LTXDirectorFirstFrameCropGuides": "LTX Director Crop Guides (First Frame)",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
