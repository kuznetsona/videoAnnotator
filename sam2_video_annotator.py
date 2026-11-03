#!/usr/bin/env python3
"""Tkinter GUI for semi-automatic video annotation with SAM2."""

from __future__ import annotations

import argparse
import bisect
import contextlib
import json
import os
import queue
import shutil
import tempfile
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT_DIR / "models" / "sam2" / "sam2.1_hiera_tiny.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"


@dataclass(frozen=True)
class PromptPoint:
    frame_idx: int
    x: float
    y: float
    label: int
    object_id: int


@dataclass(frozen=True)
class ExtractedFramesInfo:
    fps: float
    width: int
    height: int
    frame_count: int
    sam_scale: float
    source_frame_indices: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, help="Optional video to open on startup.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu", "mps"))
    parser.add_argument("--object-id", type=int, default=1)
    parser.add_argument("--max-display-width", type=int, default=980)
    parser.add_argument(
        "--max-sam-width",
        type=int,
        default=0,
        help="Resize frames for SAM2 to this width before inference. 0 keeps original size.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=85, help="Temporary frame JPEG quality, 1..100.")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Run SAM2 on every Nth source frame. Missing boxes are linearly interpolated.",
    )
    parser.add_argument(
        "--vos-optimized",
        action="store_true",
        help="Enable SAM2 torch.compile video optimization. First run can be slower while compiling.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Annotate only the first N source frames (before --frame-step sampling). 0 uses the whole video.",
    )
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must be 0 or greater")
    return args


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class Sam2Worker:
    def __init__(
        self,
        checkpoint: Path,
        config: str,
        device: str,
        max_sam_width: int = 0,
        jpeg_quality: int = 85,
        vos_optimized: bool = False,
        frame_step: int = 1,
        max_frames: int = 0,
    ):
        self.checkpoint = checkpoint
        self.config = config
        self.device = device
        self.max_sam_width = max(0, int(max_sam_width))
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self.vos_optimized = bool(vos_optimized)
        self.frame_step = max(1, int(frame_step))
        self.max_frames = max(0, int(max_frames))

    def run(
        self,
        video_path: Path,
        prompts: list[PromptPoint],
        output_dir: Path,
        progress: queue.Queue,
    ) -> None:
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {self.checkpoint}\n"
                "Run: python3 download_sam2_model.py --variant tiny"
            )
        if not prompts:
            raise ValueError("Add at least one positive point on the object.")

        from sam2.build_sam import build_sam2_video_predictor

        frame_dir = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
        try:
            frame_info = extract_video_frames(
                video_path,
                frame_dir,
                progress,
                max_sam_width=self.max_sam_width,
                jpeg_quality=self.jpeg_quality,
                frame_step=self.frame_step,
                max_frames=self.max_frames,
            )
            torch_context, autocast_context = self._torch_contexts()
            with torch_context, autocast_context:
                if self.vos_optimized:
                    progress.put(("status", "Loading SAM2 model with VOS optimization..."))
                else:
                    progress.put(("status", "Loading SAM2 model..."))
                predictor = build_sam2_video_predictor(
                    self.config,
                    str(self.checkpoint),
                    device=self.device,
                    vos_optimized=self.vos_optimized,
                )
                if self.device == "cpu":
                    configure_sam2_for_cpu(predictor)
                progress.put(("status", "Initializing SAM2 video state..."))
                inference_state = predictor.init_state(video_path=str(frame_dir))
                predictor.reset_state(inference_state)

                frame_prompts: dict[tuple[int, int], list[PromptPoint]] = {}
                for prompt in prompts:
                    sampled_idx = nearest_sampled_position(
                        frame_info.source_frame_indices,
                        prompt.frame_idx,
                    )
                    frame_prompts.setdefault((sampled_idx, prompt.object_id), []).append(prompt)

                for frame_idx, object_id in sorted(frame_prompts):
                    items = frame_prompts[(frame_idx, object_id)]
                    points = np.array(
                        [[p.x * frame_info.sam_scale, p.y * frame_info.sam_scale] for p in items],
                        dtype=np.float32,
                    )
                    labels = np.array([p.label for p in items], dtype=np.int32)
                    source_idx = frame_info.source_frame_indices[frame_idx]
                    progress.put(("status", f"Adding prompts for object {object_id} near source frame {source_idx}..."))
                    predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=object_id,
                        points=points,
                        labels=labels,
                    )

                sampled_masks: dict[int, dict[int, np.ndarray]] = {}
                sampled_boxes: dict[int, dict[int, tuple[int, int, int, int]]] = {}
                progress.put(("status", "Propagating masks through video..."))
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                    sampled_idx = int(out_frame_idx)
                    source_idx = frame_info.source_frame_indices[sampled_idx]
                    mask_logits = out_mask_logits.detach().cpu().numpy()
                    for obj_pos, obj_id in enumerate(out_obj_ids):
                        object_id = int(obj_id)
                        mask = (mask_logits[obj_pos] > 0.0).astype(np.uint8)
                        if mask.ndim == 3:
                            mask = mask[0]
                        if frame_info.sam_scale != 1.0:
                            mask = cv2.resize(
                                mask,
                                (frame_info.width, frame_info.height),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        sampled_masks.setdefault(object_id, {})[source_idx] = mask
                        box = bbox_from_mask(mask)
                        if box is not None:
                            sampled_boxes.setdefault(object_id, {})[source_idx] = box
                    progress.put(("progress", source_idx + 1, frame_info.frame_count))

            object_ids = sorted({prompt.object_id for prompt in prompts})
            masks_by_object = {
                object_id: fill_masks_from_nearest(sampled_masks.get(object_id, {}), frame_info.frame_count)
                for object_id in object_ids
            }
            boxes_by_object = {
                object_id: interpolate_boxes(
                    sampled_boxes.get(object_id, {}),
                    frame_info.frame_count,
                    tuple(sorted(sampled_masks.get(object_id, {}))),
                )
                for object_id in object_ids
            }

            output_dir.mkdir(parents=True, exist_ok=True)
            stem = video_path.stem
            masks_npz = output_dir / f"{stem}_sam2_masks.npz"
            np.savez_compressed(
                masks_npz,
                **{
                    f"object_{object_id}_frame_{frame_idx}": mask
                    for object_id, frames in masks_by_object.items()
                    for frame_idx, mask in frames.items()
                },
            )
            annotations_json = output_dir / f"{stem}_sam2_annotations.json"
            write_annotations_json(
                annotations_json,
                video_path,
                frame_info.width,
                frame_info.height,
                frame_info.fps,
                frame_info.frame_count,
                boxes_by_object,
                {object_id: set(sampled_boxes.get(object_id, {})) for object_id in object_ids},
            )
            preview_mp4 = output_dir / f"{stem}_sam2_preview.mp4"
            write_preview_video(
                video_path,
                preview_mp4,
                masks_by_object,
                boxes_by_object,
                frame_info.fps,
                progress,
                max_frames=frame_info.frame_count,
            )
            progress.put(("done", annotations_json, masks_npz, preview_mp4, masks_by_object, boxes_by_object))
        finally:
            shutil.rmtree(frame_dir, ignore_errors=True)

    def _torch_contexts(self):
        try:
            import torch
        except Exception:
            return contextlib.nullcontext(), contextlib.nullcontext()
        if self.device == "cuda" and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16)
        return torch.inference_mode(), contextlib.nullcontext()


def configure_sam2_for_cpu(predictor) -> None:
    """Keep SAM2 video-memory tensors in float32 on CPU.

    Upstream SAM2 stores these tensors as bfloat16 to save accelerator memory.
    CPU inference then feeds them to float32 linear layers, which can raise a
    ``mat1 and mat2 must have the same dtype`` error on later video frames.
    """
    predictor.float()

    original_run_memory_encoder = predictor._run_memory_encoder

    def run_memory_encoder_float32(_self, *args, **kwargs):
        maskmem_features, maskmem_pos_enc = original_run_memory_encoder(*args, **kwargs)
        return maskmem_features.float(), maskmem_pos_enc

    predictor._run_memory_encoder = types.MethodType(run_memory_encoder_float32, predictor)

    original_run_single_frame = predictor._run_single_frame_inference

    def run_single_frame_float32(_self, *args, **kwargs):
        current_out, pred_masks = original_run_single_frame(*args, **kwargs)
        maskmem_features = current_out.get("maskmem_features")
        if maskmem_features is not None:
            current_out["maskmem_features"] = maskmem_features.float()
        return current_out, pred_masks

    predictor._run_single_frame_inference = types.MethodType(run_single_frame_float32, predictor)


def extract_video_frames(
    video_path: Path,
    frame_dir: Path,
    progress: queue.Queue,
    max_sam_width: int = 0,
    jpeg_quality: int = 85,
    frame_step: int = 1,
    max_frames: int = 0,
) -> ExtractedFramesInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames > 0:
        frame_count = min(frame_count, max_frames) if frame_count > 0 else max_frames
    sam_scale = 1.0
    if max_sam_width > 0 and width > max_sam_width:
        sam_scale = max_sam_width / float(width)
    source_idx = 0
    saved_count = 0
    source_frame_indices: list[int] = []
    if sam_scale == 1.0:
        progress.put(("status", "Extracting video frames for SAM2..."))
    else:
        progress.put(("status", f"Extracting resized frames for SAM2 ({sam_scale:.2f}x)..."))
    while max_frames <= 0 or source_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if source_idx % frame_step == 0:
            if sam_scale != 1.0:
                frame = cv2.resize(
                    frame,
                    (max(1, int(width * sam_scale)), max(1, int(height * sam_scale))),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imwrite(
                str(frame_dir / f"{saved_count:06d}.jpg"),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            source_frame_indices.append(source_idx)
            saved_count += 1
        source_idx += 1
        if source_idx % 25 == 0:
            progress.put(("progress", source_idx, frame_count or source_idx))
    cap.release()
    if not source_frame_indices:
        raise RuntimeError(f"Video contains no readable frames: {video_path}")
    return ExtractedFramesInfo(
        fps=fps,
        width=width,
        height=height,
        frame_count=source_idx,
        sam_scale=sam_scale,
        source_frame_indices=tuple(source_frame_indices),
    )


def nearest_sampled_position(source_indices: tuple[int, ...], target: int) -> int:
    """Return the SAM2-frame position nearest to a source-video frame."""
    pos = bisect.bisect_left(source_indices, target)
    if pos == 0:
        return 0
    if pos == len(source_indices):
        return len(source_indices) - 1
    before = source_indices[pos - 1]
    after = source_indices[pos]
    return pos - 1 if target - before <= after - target else pos


def fill_masks_from_nearest(
    sampled_masks: dict[int, np.ndarray],
    frame_count: int,
) -> dict[int, np.ndarray]:
    """Fill only frames inside the observed range with the nearest SAM2 mask."""
    if not sampled_masks:
        return {}
    sampled_indices = tuple(sorted(sampled_masks))
    return {
        frame_idx: sampled_masks[sampled_indices[nearest_sampled_position(sampled_indices, frame_idx)]]
        for frame_idx in range(sampled_indices[0], min(frame_count, sampled_indices[-1] + 1))
    }


def interpolate_boxes(
    sampled_boxes: dict[int, tuple[int, int, int, int]],
    frame_count: int,
    observed_frame_indices: tuple[int, ...] | None = None,
) -> dict[int, tuple[int, int, int, int]]:
    """Interpolate only across consecutive observations where the object is visible."""
    if not sampled_boxes:
        return {}
    observed_indices = observed_frame_indices or tuple(sorted(sampled_boxes))
    observed_indices = tuple(idx for idx in observed_indices if 0 <= idx < frame_count)
    result = {
        frame_idx: box
        for frame_idx, box in sampled_boxes.items()
        if 0 <= frame_idx < frame_count
    }
    for left_idx, right_idx in zip(observed_indices, observed_indices[1:]):
        if left_idx not in sampled_boxes or right_idx not in sampled_boxes:
            continue
        for frame_idx in range(left_idx + 1, right_idx):
            alpha = (frame_idx - left_idx) / float(right_idx - left_idx)
            result[frame_idx] = tuple(
                int(round((1.0 - alpha) * left + alpha * right))
                for left, right in zip(sampled_boxes[left_idx], sampled_boxes[right_idx])
            )
    return result


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def write_annotations_json(
    path: Path,
    video_path: Path,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    boxes_by_object: dict[int, dict[int, tuple[int, int, int, int]]],
    sampled_frame_indices: dict[int, set[int]] | None = None,
) -> None:
    sampled_frame_indices = sampled_frame_indices or {
        object_id: set(boxes) for object_id, boxes in boxes_by_object.items()
    }
    rows = []
    for object_id, boxes_by_frame in sorted(boxes_by_object.items()):
        for frame_idx, (x1, y1, x2, y2) in sorted(boxes_by_frame.items()):
            rows.append(
                {
                    "video_path": str(video_path),
                    "frame_width": width,
                    "frame_height": height,
                    "fps": fps,
                    "frame_count": frame_count,
                    "frame_id": frame_idx,
                    "object_id": object_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "visible": 1,
                    "source": (
                        "sam2"
                        if frame_idx in sampled_frame_indices.get(object_id, set())
                        else "sam2_interpolated"
                    ),
                }
            )
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_preview_video(
    video_path: Path,
    output_path: Path,
    masks_by_object: dict[int, dict[int, np.ndarray]],
    boxes_by_object: dict[int, dict[int, tuple[int, int, int, int]]],
    fps: float,
    progress: queue.Queue,
    max_frames: int = 0,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames > 0:
        total = min(total, max_frames) if total > 0 else max_frames
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps or 30.0,
        (width, height),
    )
    idx = 0
    progress.put(("status", "Writing preview video..."))
    while max_frames <= 0 or idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        for object_id, masks_by_frame in sorted(masks_by_object.items()):
            mask = masks_by_frame.get(idx)
            if mask is not None:
                frame = overlay_mask(frame, mask, object_color(object_id))
            box = boxes_by_object.get(object_id, {}).get(idx)
            if box is not None:
                draw_labeled_box(frame, box, object_id)
        writer.write(frame)
        idx += 1
        if idx % 25 == 0:
            progress.put(("progress", idx, total or idx))
    cap.release()
    writer.release()


def draw_labeled_box(frame: np.ndarray, box: tuple[int, int, int, int], object_id: int) -> None:
    x1, y1, x2, y2 = box
    color = object_color(object_id)
    label = f"id={object_id}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_x = max(0, min(x1, frame.shape[1] - text_w - 6))
    label_y = max(text_h + 8, y1 - 6)
    bg_tl = (label_x, label_y - text_h - baseline - 6)
    bg_br = (label_x + text_w + 6, label_y + baseline)
    cv2.rectangle(frame, bg_tl, bg_br, color, -1)
    cv2.putText(frame, label, (label_x + 3, label_y - 4), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)


def object_color(object_id: int) -> tuple[int, int, int]:
    """Return a stable, visually distinct BGR color for an object ID."""
    palette = (
        (40, 220, 40),
        (40, 120, 255),
        (230, 80, 80),
        (220, 80, 220),
        (40, 220, 220),
        (200, 160, 40),
    )
    return palette[(object_id - 1) % len(palette)]


def overlay_mask(frame: np.ndarray, mask: np.ndarray, object_bgr: tuple[int, int, int]) -> np.ndarray:
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    color = np.zeros_like(frame)
    color[:, :] = object_bgr
    return np.where(mask[:, :, None].astype(bool), cv2.addWeighted(frame, 0.45, color, 0.55, 0), frame)


class AnnotatorGUI:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.root.title("SAM2 video annotator")
        self.root.minsize(1040, 720)

        self.video_path: Path | None = None
        self.cap: cv2.VideoCapture | None = None
        self.frame_count = 0
        self.fps = 30.0
        self.current_idx = 0
        self.current_frame: np.ndarray | None = None
        self.display_scale = 1.0
        self.display_size = (0, 0)
        self.prompts: list[PromptPoint] = []
        self.result_masks: dict[int, dict[int, np.ndarray]] = {}
        self.result_boxes: dict[int, dict[int, tuple[int, int, int, int]]] = {}
        self.worker_queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None

        self.frame_var = tk.IntVar(value=0)
        self.object_id_var = tk.IntVar(value=args.object_id)
        self.alpha_var = tk.IntVar(value=55)
        self.status_var = tk.StringVar(value="Open a video and click the object.")
        self.progress_var = tk.StringVar(value="")
        self.device = choose_device(args.device)

        self._build_ui()
        if args.video:
            self.open_video(args.video)
        self.root.after(100, self._poll_worker)

    def _build_ui(self) -> None:
        main = tk.Frame(self.root, padx=10, pady=10)
        main.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(left, bg="#111111", cursor="crosshair")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.video_label.bind("<Button-1>", lambda event: self.add_prompt(event, 1))
        self.video_label.bind("<Button-3>", lambda event: self.add_prompt(event, 0))

        slider = tk.Scale(
            left,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            variable=self.frame_var,
            command=self._on_slider,
            showvalue=True,
        )
        slider.pack(fill=tk.X, pady=(8, 0))
        self.frame_slider = slider

        right = tk.Frame(main, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))

        io = tk.LabelFrame(right, text="Video", padx=10, pady=8)
        io.pack(fill=tk.X)
        tk.Button(io, text="Open video", command=self.choose_video).pack(fill=tk.X)
        tk.Button(io, text="Previous frame", command=lambda: self.seek_relative(-1)).pack(fill=tk.X, pady=(6, 0))
        tk.Button(io, text="Next frame", command=lambda: self.seek_relative(1)).pack(fill=tk.X, pady=(6, 0))

        prompts = tk.LabelFrame(right, text="Object prompt", padx=10, pady=8)
        prompts.pack(fill=tk.X, pady=(8, 0))
        tk.Label(
            prompts,
            text="Choose an ID, then mark that object.\nLeft click: object; right click: background",
            justify=tk.LEFT,
        ).pack(fill=tk.X)
        object_row = tk.Frame(prompts)
        object_row.pack(fill=tk.X, pady=(6, 0))
        tk.Label(object_row, text="Active object ID:").pack(side=tk.LEFT)
        object_selector = tk.Spinbox(
            object_row,
            from_=1,
            to=9999,
            width=7,
            textvariable=self.object_id_var,
            command=self.show_current,
        )
        object_selector.pack(side=tk.RIGHT)
        object_selector.bind("<Return>", lambda _event: self.show_current())
        tk.Button(prompts, text="Undo active-object prompt", command=self.undo_prompt).pack(fill=tk.X, pady=(6, 0))
        tk.Button(prompts, text="Clear active-object prompts", command=self.clear_prompts).pack(fill=tk.X, pady=(6, 0))
        tk.Button(prompts, text="Clear all prompts", command=self.clear_all_prompts).pack(fill=tk.X, pady=(6, 0))

        run = tk.LabelFrame(right, text="SAM2", padx=10, pady=8)
        run.pack(fill=tk.X, pady=(8, 0))
        tk.Label(run, text=f"Device: {self.device}", anchor="w").pack(fill=tk.X)
        tk.Label(run, text=f"Checkpoint:\n{self.args.checkpoint}", justify=tk.LEFT, wraplength=300).pack(fill=tk.X)
        tk.Button(run, text="Run annotation", command=self.start_annotation).pack(fill=tk.X, pady=(6, 0))
        tk.Button(run, text="Open output folder", command=self.open_output_dir).pack(fill=tk.X, pady=(6, 0))

        display = tk.LabelFrame(right, text="Display", padx=10, pady=8)
        display.pack(fill=tk.X, pady=(8, 0))
        tk.Label(display, text="Mask opacity").pack(anchor="w")
        tk.Scale(display, from_=0, to=90, orient=tk.HORIZONTAL, variable=self.alpha_var, command=lambda _v: self.show_current()).pack(fill=tk.X)

        tk.Label(right, textvariable=self.status_var, justify=tk.LEFT, anchor="w", wraplength=320).pack(fill=tk.X, pady=(10, 0))
        tk.Label(right, textvariable=self.progress_var, justify=tk.LEFT, anchor="w", wraplength=320).pack(fill=tk.X, pady=(6, 0))

    def choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Open video",
            initialdir=str(ROOT_DIR),
            filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")),
        )
        if path:
            self.open_video(Path(path))

    def open_video(self, path: Path) -> None:
        if self.cap is not None:
            self.cap.release()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            messagebox.showerror("Video error", f"Cannot open video:\n{path}")
            return
        self.video_path = path
        self.cap = cap
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if self.args.max_frames > 0:
            self.frame_count = min(self.frame_count, self.args.max_frames) if self.frame_count > 0 else self.args.max_frames
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.current_idx = 0
        self.prompts.clear()
        self.result_masks.clear()
        self.result_boxes.clear()
        self.frame_slider.configure(to=max(0, self.frame_count - 1))
        self.status_var.set(f"{path.name}\n{self.frame_count} frames, {self.fps:.1f} FPS")
        self.seek(0)

    def seek_relative(self, delta: int) -> None:
        self.seek(self.current_idx + delta)

    def seek(self, idx: int) -> None:
        if self.cap is None:
            return
        idx = max(0, min(max(0, self.frame_count - 1), int(idx)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return
        self.current_idx = idx
        self.current_frame = frame
        self.frame_var.set(idx)
        self.show_current()

    def _on_slider(self, value: str) -> None:
        idx = int(float(value))
        if idx != self.current_idx:
            self.seek(idx)

    def add_prompt(self, event: tk.Event, label: int) -> None:
        if self.current_frame is None:
            return
        display_w, display_h = self.display_size
        label_w = max(1, self.video_label.winfo_width())
        label_h = max(1, self.video_label.winfo_height())
        offset_x = max(0, (label_w - display_w) / 2.0)
        offset_y = max(0, (label_h - display_h) / 2.0)
        image_x = event.x - offset_x
        image_y = event.y - offset_y
        if not (0 <= image_x < display_w and 0 <= image_y < display_h):
            return
        x = image_x / self.display_scale
        y = image_y / self.display_scale
        h, w = self.current_frame.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        object_id = self.active_object_id()
        if object_id is None:
            return
        self.prompts.append(PromptPoint(self.current_idx, x, y, label, object_id))
        self.show_current()

    def undo_prompt(self) -> None:
        object_id = self.active_object_id()
        if object_id is None:
            return
        for index in range(len(self.prompts) - 1, -1, -1):
            if self.prompts[index].object_id == object_id:
                self.prompts.pop(index)
                self.show_current()
                break

    def clear_prompts(self) -> None:
        object_id = self.active_object_id()
        if object_id is None:
            return
        self.prompts = [prompt for prompt in self.prompts if prompt.object_id != object_id]
        self.show_current()

    def clear_all_prompts(self) -> None:
        self.prompts.clear()
        self.show_current()

    def active_object_id(self) -> int | None:
        try:
            object_id = int(self.object_id_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Object ID", "Object ID must be a positive integer.")
            return None
        if object_id < 1:
            messagebox.showerror("Object ID", "Object ID must be at least 1.")
            return None
        return object_id

    def show_current(self) -> None:
        if self.current_frame is None:
            return
        frame = self.current_frame.copy()
        alpha = self.alpha_var.get() / 100.0
        for object_id, masks_by_frame in sorted(self.result_masks.items()):
            mask = masks_by_frame.get(self.current_idx)
            if mask is not None:
                color = np.zeros_like(frame)
                color[:, :] = object_color(object_id)
                frame = np.where(
                    mask[:, :, None].astype(bool),
                    cv2.addWeighted(frame, 1.0 - alpha, color, alpha, 0),
                    frame,
                )
            box = self.result_boxes.get(object_id, {}).get(self.current_idx)
            if box is not None:
                draw_labeled_box(frame, box, object_id)
        for prompt in self.prompts:
            if prompt.frame_idx != self.current_idx:
                continue
            base_color = object_color(prompt.object_id)
            color = base_color if prompt.label == 1 else (0, 0, 255)
            cv2.circle(frame, (int(prompt.x), int(prompt.y)), 7, color, -1)
            cv2.circle(frame, (int(prompt.x), int(prompt.y)), 8, (255, 255, 255), 1)
            cv2.putText(
                frame,
                str(prompt.object_id),
                (int(prompt.x) + 10, int(prompt.y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        self._show_frame(frame)

    def _show_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb, scale = resize_for_display(rgb, self.args.max_display_width)
        self.display_scale = scale
        h, w = rgb.shape[:2]
        self.display_size = (w, h)
        ppm = b"P6\n%d %d\n255\n" % (w, h) + rgb.tobytes()
        image = tk.PhotoImage(data=ppm, format="PPM")
        self.video_label.configure(image=image)
        self.video_label.image = image

    def start_annotation(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showinfo("SAM2", "Annotation is already running.")
            return
        if self.video_path is None:
            messagebox.showerror("SAM2", "Open a video first.")
            return
        positive_object_ids = {prompt.object_id for prompt in self.prompts if prompt.label == 1}
        all_object_ids = {prompt.object_id for prompt in self.prompts}
        if not positive_object_ids:
            messagebox.showerror("SAM2", "Add at least one left-click object point.")
            return
        missing_positive = sorted(all_object_ids - positive_object_ids)
        if missing_positive:
            messagebox.showerror(
                "SAM2",
                f"Add a left-click object point for object ID(s): {', '.join(map(str, missing_positive))}.",
            )
            return
        output_dir = ROOT_DIR / "outputs" / "sam2_annotations"
        worker = Sam2Worker(
            self.args.checkpoint,
            self.args.config,
            self.device,
            max_sam_width=self.args.max_sam_width,
            jpeg_quality=self.args.jpeg_quality,
            vos_optimized=self.args.vos_optimized,
            frame_step=self.args.frame_step,
            max_frames=self.args.max_frames,
        )
        prompts = list(self.prompts)

        def target() -> None:
            try:
                worker.run(self.video_path, prompts, output_dir, self.worker_queue)
            except Exception as exc:
                self.worker_queue.put(("error", str(exc)))

        self.progress_var.set("")
        self.status_var.set("SAM2 annotation started...")
        self.worker_thread = threading.Thread(target=target, daemon=True)
        self.worker_thread.start()

    def _poll_worker(self) -> None:
        try:
            while True:
                item = self.worker_queue.get_nowait()
                kind = item[0]
                if kind == "status":
                    self.status_var.set(str(item[1]))
                elif kind == "progress":
                    done, total = int(item[1]), int(item[2])
                    self.progress_var.set(f"{done}/{total}")
                elif kind == "done":
                    _, annotations_json, masks_npz, preview_mp4, masks, boxes = item
                    self.result_masks = masks
                    self.result_boxes = boxes
                    self.status_var.set(
                        f"Done.\nJSON: {annotations_json}\nMasks: {masks_npz}\nPreview: {preview_mp4}"
                    )
                    self.show_current()
                elif kind == "error":
                    messagebox.showerror("SAM2 error", str(item[1]))
                    self.status_var.set("SAM2 annotation failed.")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_worker)

    def open_output_dir(self) -> None:
        path = ROOT_DIR / "outputs" / "sam2_annotations"
        path.mkdir(parents=True, exist_ok=True)
        messagebox.showinfo("Output folder", str(path))

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


def resize_for_display(frame: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0
    scale = max_width / float(width)
    resized = cv2.resize(frame, (max_width, max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    app = AnnotatorGUI(root, args)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
