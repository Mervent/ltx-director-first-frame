"""MiniMax H3 Director guide node."""
import json

from .helper_logging import log_dasiwa
from .helper_minimax_h3_director import (
    align_frame_count, assemble_prompt, audio_duration, load_audio,
    load_embedded_video_audio, load_image, load_video, normalize_guide,
    validate_reference_limits,
)
from .helper_minimax_h3_prompt_builder import (
    build_prompt, default_builder_state, normalize_ref_schema, validate_builder_state,
)

BASE_MODES = {"T2VA", "I2VA", "FL2VA", "L2VA"}


def _describe_model(model) -> str:
    if model is None:
        return "none"
    model_type = type(model)
    return f"{model_type.__module__}.{model_type.__name__}"


class MiniMaxH3DirectorFirstFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["T2VA", "I2VA", "FL2VA", "L2VA", "REF2VA"], {"default": "FL2VA"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32}),
                "duration": ("INT", {"default": 5, "min": 1, "max": 1000}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "timeline_data": ("STRING", {"default": "{\"version\":1,\"items\":[],\"prompt_blocks\":[]}", "multiline": False, "hidden": True}),
                "builder_state": ("STRING", {"default": "", "multiline": False, "hidden": True}),
            },
            "optional": {
                "fl2va_model": ("MODEL", {"lazy": True}),
                "ref2va_model": ("MODEL", {"lazy": True}),
                "first_frame_image": ("IMAGE", {"tooltip": "Optional ComfyUI IMAGE wired straight in as the video's first frame, bypassing the timeline editor. Overrides any timeline first-frame slot. Used only in the image modes (T2VA is auto-promoted to I2VA/FL2VA); ignored in REF2VA."}),
                "last_frame_image": ("IMAGE", {"tooltip": "Optional ComfyUI IMAGE wired straight in as the video's last frame, bypassing the timeline editor. Overrides any timeline last-frame slot. Used only in the image modes; ignored in REF2VA."}),
            },
        }

    RETURN_TYPES = ("MINIMAX_H3_DIRECTOR_GUIDE", "INT", "STRING", "INT", "INT", "MODEL", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("guide", "duration", "positive_prompt", "width", "height", "model", "fl2va_requested", "ref2va_requested")
    FUNCTION = "build_guide"
    CATEGORY = "DaSiWa Nodes/MiniMax H3"

    def check_lazy_status(self, mode, prompt, width, height, duration, ref_image_size, timeline_data, builder_state,
                          fl2va_model=None, ref2va_model=None, first_frame_image=None, last_frame_image=None):
        selected_name = "ref2va_model" if mode == "REF2VA" else "fl2va_model"
        selected_model = ref2va_model if mode == "REF2VA" else fl2va_model
        return [selected_name] if selected_model is None else []

    def build_guide(self, mode, prompt, width, height, duration, ref_image_size, timeline_data, builder_state="",
                    fl2va_model=None, ref2va_model=None, first_frame_image=None, last_frame_image=None):
        # Preserve direct Python callers that used the pre-builder positional model argument.
        if builder_state is not None and not isinstance(builder_state, str):
            if fl2va_model is None:
                fl2va_model = builder_state
                builder_state = ""
            else:
                raise ValueError("builder_state must be JSON text")
        if mode not in BASE_MODES | {"REF2VA"}:
            raise ValueError(f"unsupported MiniMax Director mode: {mode}")
        length = align_frame_count(int(duration) * 24)
        try:
            state = json.loads(timeline_data or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"MiniMax Director timeline_data is invalid JSON: {exc}") from exc
        if not isinstance(state, dict):
            raise ValueError("MiniMax Director timeline_data must contain an object")
        try:
            builder = json.loads(builder_state) if builder_state else state.get("builder_state", {})
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"MiniMax Director builder_state is invalid JSON: {exc}") from exc
        if not isinstance(builder, dict):
            builder = {}
        merged = default_builder_state(mode)
        merged.update(builder)
        merged["ref"] = {**default_builder_state(mode)["ref"], **(builder.get("ref") or {})}
        normalize_ref_schema(merged["ref"])
        merged["mode"] = mode
        merged["duration"] = duration

        items = sorted(enumerate(state.get("items", [])), key=lambda pair: (int(pair[1].get("order", pair[0])), pair[0]))
        items = [pair for pair in items if pair[1].get("enabled", True)]
        first_frame = last_frame = None
        ref_images, ref_videos, ref_video_audios, ref_audios = {}, {}, {}, {}
        images, videos, audios = [], [], []
        try:
            import folder_paths
            input_directory = folder_paths.get_input_directory()
        except (ImportError, AttributeError):
            input_directory = None

        if mode in BASE_MODES:
            image_items = sorted((pair for pair in items if pair[1].get("type") == "image"), key=lambda pair: (pair[1].get("slot", pair[0]), pair[0]))
            if mode == "T2VA":
                image_items = []
            elif mode in {"I2VA", "L2VA"}:
                image_items = image_items[:1]
            else:
                image_items = image_items[:2]
            for index, (_, item) in enumerate(image_items):
                value = item.get("value", item.get("tensor"))
                if isinstance(value, str) and input_directory:
                    value = load_image(value, input_directory)
                if mode == "I2VA":
                    first_frame = value
                elif mode == "L2VA" or (mode == "FL2VA" and item.get("slot", index) == 1):
                    last_frame = value
                else:
                    first_frame = value
        else:
            type_order = {"image": 0, "video": 1, "audio": 2}
            for _, item in sorted(items, key=lambda pair: (type_order.get(pair[1].get("type"), 3), pair[1].get("slot", pair[0]), pair[0])):
                kind, value = item.get("type"), item.get("value", item.get("tensor"))
                if value is None:
                    continue
                trim_start = float(item.get("trim_start", 0))
                trim_end = item.get("trim_end")
                trim_end = float(trim_end) if trim_end is not None else None
                video_mode = item.get("media_mode", "video")
                if kind == "image":
                    value = load_image(value, input_directory) if isinstance(value, str) and input_directory else value
                    ref_images[f"ref_image_{len(ref_images) + 1}"] = value
                    images.append(item)
                elif kind == "audio":
                    value = load_audio(value, input_directory, trim_start=trim_start, trim_end=trim_end) if isinstance(value, str) and input_directory else value
                    ref_audios[f"ref_audio_{len(ref_audios) + 1}"] = value
                    audios.append({**item, "duration": audio_duration(value) if isinstance(value, dict) else item.get("duration")})
                elif kind == "video":
                    if video_mode not in {"video", "audio", "video_audio"}:
                        raise ValueError(f"unsupported video media mode: {video_mode}")
                    if video_mode in {"video", "video_audio"}:
                        video = load_video(value, input_directory, trim_start=trim_start, trim_end=trim_end) if isinstance(value, str) and input_directory else value
                        ref_videos[f"ref_video_{len(ref_videos) + 1}"] = video
                        video_duration = float(video.shape[0]) / 24.0 if hasattr(video, "shape") else item.get("duration")
                        videos.append({**item, "duration": video_duration})
                    if video_mode in {"audio", "video_audio"}:
                        audio = load_embedded_video_audio(value, input_directory, trim_start=trim_start, trim_end=trim_end) if isinstance(value, str) and input_directory else item.get("audio")
                        if video_mode == "video_audio":
                            ref_video_audios[f"ref_video_audio_{len(ref_videos)}"] = audio
                        else:
                            ref_audios[f"ref_audio_{len(ref_audios) + 1}"] = audio
                        audios.append({**item, "duration": audio_duration(audio) if isinstance(audio, dict) else item.get("duration")})
                    attached_audio = item.get("audio")
                    if attached_audio is not None and video_mode not in {"audio", "video_audio"}:
                        if isinstance(attached_audio, str) and input_directory:
                            attached_audio = load_audio(attached_audio, input_directory, trim_start=trim_start, trim_end=trim_end)
                        if video_mode == "video" and ref_videos:
                            ref_video_audios[f"ref_video_audio_{len(ref_videos)}"] = attached_audio
                        else:
                            ref_audios[f"ref_audio_{len(ref_audios) + 1}"] = attached_audio
                        audios.append({**item, "duration": audio_duration(attached_audio) if isinstance(attached_audio, dict) else item.get("duration")})
            validate_reference_limits(images=images, videos=videos, audios=audios)

        # A wired IMAGE socket overrides the timeline slot. H3's native node has no strength
        # or hold-duration knob, so (unlike the LTX fork) only the raw image is carried.
        if mode in BASE_MODES:
            if first_frame_image is not None and first_frame_image.numel() > 0:
                ff = first_frame_image[0:1]
                first_frame = (ff[..., :3] if ff.shape[-1] > 3 else ff).float()
            if last_frame_image is not None and last_frame_image.numel() > 0:
                lf = last_frame_image[0:1]
                last_frame = (lf[..., :3] if lf.shape[-1] > 3 else lf).float()
            # T2VA drops images from the prompt header; promote so a wired frame is actually used.
            if mode == "T2VA" and (first_frame is not None or last_frame is not None):
                if first_frame is not None and last_frame is not None:
                    mode = "FL2VA"
                elif last_frame is not None:
                    mode = "L2VA"
                else:
                    mode = "I2VA"
                merged["mode"] = mode
                log_dasiwa("MiniMax H3 Director", f"first/last-frame IMAGE wired in T2VA; promoting mode to {mode}")
        elif first_frame_image is not None or last_frame_image is not None:
            log_dasiwa("MiniMax H3 Director", f"first/last-frame IMAGE sockets are ignored in {mode} mode")

        blocks = state.get("prompt_blocks", [])
        resolved = build_prompt(merged)
        if not any(str(merged.get(key) or "").strip() for key in ("imd", "soundscape")) and mode != "REF2VA":
            resolved = assemble_prompt(prompt, blocks)
        for issue in validate_builder_state(merged):
            log_dasiwa("MiniMax H3 Director", f"[{issue['level'].upper()}] {issue['msg']}")
        guide = {
            "version": 2, "mode": mode, "prompt": prompt, "prompt_blocks": blocks, "resolved_prompt": resolved,
            "width": width, "height": height, "length": length, "ref_image_size": ref_image_size,
            "first_frame": first_frame, "last_frame": last_frame, "ref_images": ref_images, "ref_videos": ref_videos,
            "ref_video_audios": ref_video_audios, "ref_audios": ref_audios, "builder_state": merged,
            "timeline": [{key: item.get(key) for key in ("id", "type", "start", "duration", "order", "trim_start", "trim_end") if key in item} for _, item in items],
            "prompt_payload": {"mode": mode, "full_prompt": resolved, "is_ref_mode": mode == "REF2VA", "subject_definitions": merged["ref"]["subject_defs"], "summary": merged["ref"]["summary_text"], "retention_analysis": merged["ref"]["retention"], "detailed_description": {"style_line": merged["ref"]["style_line"], "detail": merged["ref"]["detail"]}, "overall_soundscape": merged["ref"]["soundscape"] if mode == "REF2VA" else merged["soundscape"], "non_diegetic_music": merged["ref"]["music"] if mode == "REF2VA" else merged["music"], "imd": merged.get("imd", ""), "p2_shot": merged.get("p2_shot", ""), "last_shot": merged.get("last_shot", "")},
        }
        normalize_guide(guide)
        selected_model = ref2va_model if mode == "REF2VA" else fl2va_model
        log_dasiwa("MiniMax H3 Director", f"mode={mode}; requested_model={'ref2va_model' if mode == 'REF2VA' else 'fl2va_model'}; passed_model={_describe_model(selected_model)}; canvas={width}x{height}; frames={length}; refs=images:{len(ref_images)},videos:{len(ref_videos)},video_audio:{len(ref_video_audios)},audio:{len(ref_audios)}; timeline_items={len(items)}")
        return guide, length, resolved, int(width), int(height), selected_model, mode in BASE_MODES, mode == "REF2VA"


NODE_CLASS_MAPPINGS = {"MiniMaxH3DirectorFirstFrame": MiniMaxH3DirectorFirstFrame}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3DirectorFirstFrame": "MiniMax H3 Director (First Frame Input)"}
