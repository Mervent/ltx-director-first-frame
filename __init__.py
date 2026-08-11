from .ltx_director import LTXDirectorFirstFrame
from .ltx_director_guide import LTXDirectorGuide, LTXDirectorCropGuides
from .nodes.nodes_minimax_h3_director import MiniMaxH3DirectorFirstFrame
from .nodes.nodes_minimax_h3_director_guide import MiniMaxH3DirectorFirstFrameGuide
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override


class LTXDirectorFirstFrameExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [LTXDirectorFirstFrame]


async def comfy_entrypoint() -> LTXDirectorFirstFrameExtension:
    return LTXDirectorFirstFrameExtension()


NODE_CLASS_MAPPINGS = {
    "LTXDirectorFirstFrame": LTXDirectorFirstFrame,
    "LTXDirectorFirstFrameGuide": LTXDirectorGuide,
    "LTXDirectorFirstFrameCropGuides": LTXDirectorCropGuides,
    "MiniMaxH3DirectorFirstFrame": MiniMaxH3DirectorFirstFrame,
    "MiniMaxH3DirectorFirstFrameGuide": MiniMaxH3DirectorFirstFrameGuide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXDirectorFirstFrame": "LTX Director (First Frame Input)",
    "LTXDirectorFirstFrameGuide": "LTX Director Guide (First Frame)",
    "LTXDirectorFirstFrameCropGuides": "LTX Director Crop Guides (First Frame)",
    "MiniMaxH3DirectorFirstFrame": "MiniMax H3 Director (First Frame Input)",
    "MiniMaxH3DirectorFirstFrameGuide": "MiniMax H3 Director Guide (First Frame)",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
