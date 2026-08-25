"""Seed-keyed gate that freezes its upstream branch while the key stays unchanged."""
import torch

from .helper_logging import log_dasiwa

_LAST_RUN = {}


def _connected_inputs(prompt, unique_id):
    inputs = ((prompt or {}).get(str(unique_id)) or {}).get("inputs", {})
    return {name for name, value in inputs.items() if isinstance(value, list)}


def _pack_images(images, store_uint8):
    if images is None or not store_uint8:
        return images
    return (images.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)


def _unpack_images(images):
    if images is None or images.dtype != torch.uint8:
        return images
    return images.to(torch.float32) / 255.0


class SeedFreezeGate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "key": ("INT", {"forceInput": True, "tooltip": "Wire your seed here. While this value stays the same, the stored batch is served and the branch upstream of this node is not executed at all."}),
                "enabled": ("BOOLEAN", {"default": True, "tooltip": "Off = transparent pass-through: upstream always runs and the stored batch is left untouched."}),
                "store_uint8": ("BOOLEAN", {"default": True, "tooltip": "Store frames as uint8 (4x less RAM). Lossless for video encoding, which quantizes to 8 bit anyway."}),
            },
            "optional": {
                "images": ("IMAGE", {"lazy": True, "tooltip": "Expensive image batch to freeze (e.g. sampler + VAE decode output)."}),
                "audio": ("AUDIO", {"lazy": True, "tooltip": "Audio produced by the same upstream branch. Route it through this gate too, or its consumer will pull the sampler anyway."}),
                "text": ("STRING", {"forceInput": True, "lazy": True, "tooltip": "Text produced by the same upstream branch (e.g. a caption/prompt). Route it through this gate too, or its consumer will pull the sampler anyway."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"},
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "text")
    FUNCTION = "gate"
    CATEGORY = "DaSiWa Nodes/Utils"

    # Returning [] tells ComfyUI the lazy inputs are not needed, so the whole
    # upstream branch (sampler/VAE) is never staged for execution. This works
    # independently of the output cache, surviving RAM-pressure eviction.
    def check_lazy_status(self, key, enabled, store_uint8, images=None, audio=None, text=None, unique_id=None, prompt=None):
        entry = _LAST_RUN.get(unique_id)
        if enabled and entry is not None and entry[0] == key:
            return []
        connected = _connected_inputs(prompt, unique_id)
        return [name for name, value in (("images", images), ("audio", audio), ("text", text)) if value is None and name in connected]

    def gate(self, key, enabled, store_uint8, images=None, audio=None, text=None, unique_id=None, prompt=None):
        entry = _LAST_RUN.get(unique_id)
        if images is None and audio is None and text is None:
            if entry is None:
                raise ValueError("SeedFreezeGate has nothing frozen yet: wire images, audio or text into it and run once")
            log_dasiwa("Seed Freeze Gate", f"key={key} unchanged; serving frozen batch, upstream skipped")
            return _unpack_images(entry[1]), entry[2], entry[3]
        if not enabled:
            return images, audio, text
        same_key = entry is not None and entry[0] == key
        stored_images = _pack_images(images, store_uint8) if images is not None else (entry[1] if same_key else None)
        stored_audio = audio if audio is not None else (entry[2] if same_key else None)
        stored_text = text if text is not None else (entry[3] if same_key else None)
        _LAST_RUN[unique_id] = (key, stored_images, stored_audio, stored_text)
        size_mb = stored_images.numel() * stored_images.element_size() / 1e6 if stored_images is not None else 0.0
        log_dasiwa("Seed Freeze Gate", f"froze batch for key={key} ({size_mb:.0f} MB held in RAM)")
        return (
            images if images is not None else _unpack_images(stored_images),
            audio if audio is not None else stored_audio,
            text if text is not None else stored_text,
        )


NODE_CLASS_MAPPINGS = {"SeedFreezeGate": SeedFreezeGate}
NODE_DISPLAY_NAME_MAPPINGS = {"SeedFreezeGate": "Seed Freeze Gate"}
