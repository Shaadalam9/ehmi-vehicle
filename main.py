import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from tqdm.auto import tqdm

# Qualcomm AI Hub Models (LaMa-Dilated)
from qai_hub_models.utils.asset_loaders import always_answer_prompts
from qai_hub_models.utils.image_processing import torch_tensor_to_PIL_image
from qai_hub_models.models.lama_dilated import Model as LamaModel


# =========================
# CONFIG (edit these)
# =========================
INPUT_ROOT = Path("data")          # contains nested folders with .mp4
OUTPUT_ROOT = Path("_output")      # mirrored structure will be created
KEEP_CLASS = "bus"                 # one of: "car", "bus", "motorcycle", "truck"
YOLO_WEIGHTS = "yolov8x-seg.pt"    # Ultralytics will auto-download if missing
CONF_THRES = 0.25                  # detection confidence
INPAINT_SIZE = 512                 # LaMa typically runs well at 512 (square)
KEEP_AUDIO = True                  # copy audio from original using ffmpeg
EVERY_N = 1                        # process every Nth frame (others keep original)
VIDEO_EXTS = {".mp4"}              # add more if needed

ROAD_USER_CLASSES = {"person", "bicycle", "motorcycle", "car", "bus", "truck"}


@dataclass
class LetterboxMeta:
    orig_w: int
    orig_h: int
    size: int
    new_w: int
    new_h: int
    pad_left: int
    pad_top: int


def letterbox_to_square(img: Image.Image, size: int) -> Tuple[Image.Image, LetterboxMeta]:
    """Resize with aspect ratio preserved + pad to size x size."""
    orig_w, orig_h = img.size
    scale = min(size / orig_w, size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    resized = img.resize((new_w, new_h), Image.BILINEAR)  # type: ignore
    canvas = Image.new("RGB", (size, size), (0, 0, 0))

    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))

    meta = LetterboxMeta(orig_w, orig_h, size, new_w, new_h, pad_left, pad_top)
    return canvas, meta


def unletterbox_from_square(img_sq: Image.Image, meta: LetterboxMeta) -> Image.Image:
    """Crop the padded region and resize back to original resolution."""
    crop = img_sq.crop((meta.pad_left, meta.pad_top, meta.pad_left + meta.new_w, meta.pad_top + meta.new_h))
    return crop.resize((meta.orig_w, meta.orig_h), Image.BILINEAR)  # type: ignore


def pil_rgb_to_torch(pil_img: Image.Image, device: torch.device) -> torch.Tensor:
    """PIL RGB -> torch float tensor [1,3,H,W] in [0,1]."""
    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return t.to(device)


def boolmask_to_torch(mask_bool: np.ndarray, device: torch.device) -> torch.Tensor:
    """bool mask -> torch float tensor [1,1,H,W] with values 0/1."""
    t = torch.from_numpy(mask_bool.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return t.to(device)


def pick_main_instance(candidate_masks: np.ndarray, H: int, W: int) -> Optional[int]:
    """Pick ONE instance to keep (largest area, slightly center-biased)."""
    if candidate_masks.shape[0] == 0:
        return None

    cy, cx = H / 2.0, W / 2.0
    best_i, best_score = None, -1e18

    for i in range(candidate_masks.shape[0]):
        m = candidate_masks[i]
        area = float(m.sum())
        if area <= 0:
            continue

        ys, xs = np.where(m)
        my, mx = float(ys.mean()), float(xs.mean())
        dist = ((my - cy) ** 2 + (mx - cx) ** 2) ** 0.5

        score = area - 0.25 * dist
        if score > best_score:
            best_score = score
            best_i = i

    return best_i


def build_remove_mask_yolo(
    yolo_seg: YOLO,
    frame_rgb: np.ndarray,
    keep_class: str,
    conf: float = 0.25,
    dilate_px: int = 7,
) -> np.ndarray:
    """
    Returns HxW boolean mask where True means "REMOVE/INPAINT".
    Keeps exactly ONE instance of keep_class; removes all other road users.
    """
    H, W, _ = frame_rgb.shape

    # Force YOLO to run at the same resolution as the frame we pass in
    res = yolo_seg.predict(frame_rgb, conf=conf, imgsz=H, verbose=False)[0]

    if res.masks is None or res.boxes is None:
        return np.zeros((H, W), dtype=bool)

    masks_f = res.masks.data.cpu().numpy()  # type: ignore # (N, h, w)
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)  # type: ignore
    names = res.names
    labels = np.array([names[int(c)] for c in cls_ids], dtype=object)

    # Safety: resize masks to (H, W) if needed
    if masks_f.shape[1] != H or masks_f.shape[2] != W:
        resized = np.zeros((masks_f.shape[0], H, W), dtype=np.float32)
        for i in range(masks_f.shape[0]):
            resized[i] = cv2.resize(masks_f[i], (W, H), interpolation=cv2.INTER_NEAREST)
        masks_f = resized

    masks = masks_f > 0.5

    # Choose ONE keep instance
    keep_candidates = np.where(labels == keep_class)[0]
    keep_mask = np.zeros((H, W), dtype=bool)
    if keep_candidates.size > 0:
        local_idx = pick_main_instance(masks[keep_candidates], H, W)
        if local_idx is not None:
            keep_mask = masks[int(keep_candidates[local_idx])]

    # Remove other road users (including other instances of keep_class)
    remove_mask = np.zeros((H, W), dtype=bool)
    for i, lab in enumerate(labels):
        if lab not in ROAD_USER_CLASSES:
            continue

        # Keep only the selected keep instance
        if lab == keep_class and keep_candidates.size > 0:
            overlap = (masks[i] & keep_mask).sum()
            if overlap / max(1, masks[i].sum()) > 0.7:
                continue

        remove_mask |= masks[i]

    remove_mask &= ~keep_mask

    # Dilate to reduce edge halos
    if dilate_px > 0 and remove_mask.any():
        k = np.ones((dilate_px, dilate_px), np.uint8)
        u8 = (remove_mask.astype(np.uint8) * 255)
        u8 = cv2.dilate(u8, k, iterations=1)
        remove_mask = u8 > 0

    return remove_mask


def copy_audio_ffmpeg(video_noaudio: Path, original_video: Path, final_out: Path) -> bool:
    """Copy audio stream from original onto the new video (works even if no audio)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_noaudio),
        "-i", str(original_video),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        str(final_out),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode == 0


def process_one_video(
    video_path: Path,
    out_video_path: Path,
    yolo_seg: YOLO,
    lama_model: torch.nn.Module,
    device: torch.device,
    keep_class: str,
    conf: float,
    inpaint_size: int,
    keep_audio: bool,
    every_n: int,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Try to get total frames for tqdm
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None  # unknown

    out_video_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_noaudio = out_video_path.with_name(out_video_path.stem + "__noaudio.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    writer = cv2.VideoWriter(str(tmp_noaudio), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for: {tmp_noaudio}")

    frame_idx = 0

    # Per-video frame progress bar
    pbar = tqdm(
        total=total_frames,
        desc=f"Frames: {video_path.name}",
        unit="frame",
        leave=False,
    )

    try:
        with torch.inference_mode():
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                # keep length consistent if skipping frames
                if every_n > 1 and (frame_idx % every_n) != 0:
                    writer.write(frame_bgr)
                    frame_idx += 1
                    pbar.update(1)
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(frame_rgb)

                # inpaint on square letterboxed view
                pil_sq, meta = letterbox_to_square(pil, inpaint_size)
                sq_np = np.array(pil_sq)

                remove_mask = build_remove_mask_yolo(
                    yolo_seg=yolo_seg,
                    frame_rgb=sq_np,
                    keep_class=keep_class,
                    conf=conf,
                    dilate_px=7,
                )

                if remove_mask.any():
                    img_t = pil_rgb_to_torch(pil_sq, device)        # [1,3,H,W]
                    mask_t = boolmask_to_torch(remove_mask, device) # [1,1,H,W]
                    out_t = lama_model(img_t, mask_t)               # [1,3,H,W]
                    out_sq = torch_tensor_to_PIL_image(out_t[0].detach().cpu())
                else:
                    out_sq = pil_sq

                out_full = unletterbox_from_square(out_sq, meta)
                out_bgr = cv2.cvtColor(np.array(out_full), cv2.COLOR_RGB2BGR)
                writer.write(out_bgr)

                frame_idx += 1
                pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        writer.release()

    # audio merge
    if keep_audio:
        final_ok = copy_audio_ffmpeg(tmp_noaudio, video_path, out_video_path)
        if final_ok:
            tmp_noaudio.unlink(missing_ok=True)
        else:
            tmp_noaudio.replace(out_video_path)
    else:
        tmp_noaudio.replace(out_video_path)


def process_all_videos():
    input_root = INPUT_ROOT.resolve()
    output_root = OUTPUT_ROOT.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    videos = [p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    if not videos:
        print(f"No videos found under: {input_root}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # YOLO segmentation
    yolo_seg = YOLO(YOLO_WEIGHTS)

    # LaMa-Dilated model (auto-download; may prompt to clone sources -> auto-accept)
    with always_answer_prompts(True):
        lama_model = LamaModel.from_pretrained()
    lama_model = lama_model.to(device).eval()

    # Videos progress bar
    for vp in tqdm(videos, desc="Videos", unit="video"):
        rel = vp.relative_to(input_root)
        out_path = (output_root / rel).with_name(vp.stem + "_cleaned.mp4")

        try:
            process_one_video(
                video_path=vp,
                out_video_path=out_path,
                yolo_seg=yolo_seg,
                lama_model=lama_model,
                device=device,
                keep_class=KEEP_CLASS,
                conf=CONF_THRES,
                inpaint_size=INPAINT_SIZE,
                keep_audio=KEEP_AUDIO,
                every_n=max(1, EVERY_N),
            )
            tqdm.write(f"OK: {rel} -> {out_path.relative_to(output_root)}")
        except Exception as e:
            tqdm.write(f"FAIL: {rel}  ({e})")


if __name__ == "__main__":
    process_all_videos()
