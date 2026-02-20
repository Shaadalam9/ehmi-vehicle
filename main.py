import base64
import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

import common
from openai import OpenAI


# =======================
# EDIT THESE SETTINGS
# =======================
INPUT_ROOT = Path(r"data")
OUTPUT_ROOT = Path(r"_output")

KEEP_VEHICLE = "bus"            # e.g. "bus", "car", "motorcycle"
EXTRACT_FPS = None              # None = all frames, or e.g. 5.0

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

OVERWRITE_FRAMES = False
OVERWRITE_EDITS = False
OVERWRITE_VIDEOS = False

# OpenAI
OPENAI_API_KEY = common.get_secrets("OPENAI_API_KEY")

# Vision model used to output boxes (choose a vision-capable model available to your account)
VISION_MODEL = "gpt-4.1-mini"

# Image edit model
OPENAI_IMAGE_MODEL = "gpt-image-1.5"
OPENAI_QUALITY = "high"            # low|medium|high|auto
OPENAI_INPUT_FIDELITY = "high"     # high|low
OPENAI_OUTPUT_FORMAT = "png"       # png|jpeg|webp

# Output encoding:
ENCODE_MODE = "visually_lossless_mp4"  # or "lossless_mkv"

# Mask tuning (helps reduce edge halos)
REMOVE_BOX_MARGIN_PX = 12
KEEP_BOX_MARGIN_PX = 20
MASK_DILATE_PX = 3  # 0 to disable

# Tiling to preserve original resolution (API supports limited sizes) :contentReference[oaicite:2]{index=2}
TILE_OVERLAP = 128
SQUARE_TILE = (1024, 1024)
LANDSCAPE_TILE = (1536, 1024)
PORTRAIT_TILE = (1024, 1536)

ALLOWED_SIZES = {
    (1024, 1024): "1024x1024",
    (1536, 1024): "1536x1024",
    (1024, 1536): "1024x1536",
}

ALIASES = {"motorbike": "motorcycle", "bike": "bicycle", "people": "person"}


# =======================
# ffmpeg helpers
# =======================
def iter_videos(root: Path) -> List[Path]:
    return sorted(root.rglob("*.mp4"))


def run(cmd: List[str]):
    subprocess.run(cmd, check=True)


def get_video_fps(video_path: Path) -> float:
    cmd = [
        FFPROBE, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate",
        "-of", "json",
        str(video_path)
    ]
    out = subprocess.check_output(cmd).decode("utf-8")
    data = json.loads(out)
    s = data["streams"][0]
    rate = s.get("avg_frame_rate") or s.get("r_frame_rate") or "30/1"
    num, den = rate.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 30.0
    if fps <= 0 or fps > 240:
        fps = 30.0
    return fps


def extract_frames_png(video_path: Path, frames_dir: Path, fps: Optional[float], overwrite: bool):
    frames_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(frames_dir.glob("*.png")):
        return

    if overwrite:
        for f in frames_dir.glob("*.png"):
            f.unlink()

    out_pattern = str(frames_dir / "%06d.png")
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(video_path), "-vsync", "0"]
    if fps is not None and fps > 0:
        cmd += ["-vf", f"fps={fps}"]
    cmd += [out_pattern]  # PNG = lossless frames
    run(cmd)


# =======================
# Vision: ask model for keep_box + remove_boxes
# =======================
def _encode_for_vision(img: Image.Image, max_side: int = 1024, jpeg_q: int = 85):
    """
    Downscale for cheaper vision; return (data_url, scale_x, scale_y, w_small, h_small).
    The Responses API accepts image_url as URL or base64 data URL. :contentReference[oaicite:3]{index=3}
    """
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    ws, hs = int(round(w * scale)), int(round(h * scale))
    small = img.resize((ws, hs), resample=Image.BILINEAR) if (ws, hs) != (w, h) else img

    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=jpeg_q, optimize=True)
    data_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{data_b64}"

    return data_url, (w / ws), (h / hs), ws, hs


def get_boxes_from_model(client: OpenAI, frame_rgb: Image.Image, keep_vehicle: str) -> Tuple[Dict, List[Dict]]:
    """
    Returns (keep_box, remove_boxes) in ORIGINAL frame coords.
    keep_box: {"x1","y1","x2","y2"}
    remove_boxes: [{"x1","y1","x2","y2","label"}...]
    """
    data_url, sx, sy, ws, hs = _encode_for_vision(frame_rgb, max_side=1024)

    instruction = (
        f"Image size is {ws}x{hs}.\n"
        f"Task: Keep ONLY the main {keep_vehicle} (the primary/largest {keep_vehicle}).\n"
        f"1) Return keep_box tightly around that main {keep_vehicle}.\n"
        f"2) Return remove_boxes for ALL other road users: people, bicycles, cars, trucks, motorcycles, other buses.\n"
        f"IMPORTANT: Do NOT include the kept {keep_vehicle} in remove_boxes.\n\n"
        f"Return ONLY JSON in this format:\n"
        f'{{"keep_box":{{"x1":int,"y1":int,"x2":int,"y2":int}},'
        f'"remove_boxes":[{{"x1":int,"y1":int,"x2":int,"y2":int,"label":str}}]}}\n'
        f"All coordinates are pixel coords within {ws}x{hs}."
    )

    resp = client.responses.create(
        model=VISION_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": instruction},
                {"type": "input_image", "image_url": data_url},
            ],
        }],
        text={"format": {"type": "json_object"}},  # structured JSON output :contentReference[oaicite:4]{index=4}
    )

    data = json.loads(resp.output_text)
    keep_box = data["keep_box"]
    remove_boxes = data.get("remove_boxes", [])

    # scale back to original
    def sc(b):
        return {
            "x1": float(b["x1"]) * sx,
            "y1": float(b["y1"]) * sy,
            "x2": float(b["x2"]) * sx,
            "y2": float(b["y2"]) * sy,
        }

    keep_box = sc(keep_box)
    out_remove = []
    for b in remove_boxes:
        bb = sc(b)
        bb["label"] = b.get("label", "")
        out_remove.append(bb)

    return keep_box, out_remove


# =======================
# Mask building
# =======================
def _clip_box(b, w, h):
    x1 = int(round(b["x1"]))
    y1 = int(round(b["y1"]))
    x2 = int(round(b["x2"]))
    y2 = int(round(b["y2"]))
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    return x1, y1, x2, y2


def _expand_box(x1, y1, x2, y2, m, w, h):
    x1 = max(0, x1 - m)
    y1 = max(0, y1 - m)
    x2 = min(w, x2 + m)
    y2 = min(h, y2 + m)
    return x1, y1, x2, y2


def _dilate_edit_region_alpha(alpha: np.ndarray, dilate_px: int) -> np.ndarray:
    if dilate_px <= 0:
        return alpha
    edit = (alpha == 0).astype(np.uint8) * 255
    k = dilate_px * 2 + 1
    if k < 3:
        return alpha
    edit_img = Image.fromarray(edit, mode="L").filter(ImageFilter.MaxFilter(k))
    edit2 = np.array(edit_img) > 0
    out = alpha.copy()
    out[edit2] = 0
    return out


def build_alpha_mask_from_boxes(img_size: Tuple[int, int], keep_box: Dict, remove_boxes: List[Dict]) -> Image.Image:
    """
    RGBA PNG mask where alpha=0 => edit, alpha=255 => keep. Mask rules are defined by the Images API. :contentReference[oaicite:5]{index=5}
    """
    w, h = img_size
    alpha = np.full((h, w), 255, dtype=np.uint8)

    # Remove boxes
    for b in remove_boxes:
        x1, y1, x2, y2 = _clip_box(b, w, h)
        x1, y1, x2, y2 = _expand_box(x1, y1, x2, y2, REMOVE_BOX_MARGIN_PX, w, h)
        if x2 > x1 and y2 > y1:
            alpha[y1:y2, x1:x2] = 0

    # Dilate removal a bit
    alpha = _dilate_edit_region_alpha(alpha, MASK_DILATE_PX)

    # Re-apply keep_box protection (so bus never gets edited)
    kx1, ky1, kx2, ky2 = _clip_box(keep_box, w, h)
    kx1, ky1, kx2, ky2 = _expand_box(kx1, ky1, kx2, ky2, KEEP_BOX_MARGIN_PX, w, h)
    if kx2 > kx1 and ky2 > ky1:
        alpha[ky1:ky2, kx1:kx2] = 255

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


# =======================
# Inpainting with mask
# =======================
def _size_str_for_image(img: Image.Image) -> str:
    # Sizes supported by GPT image edits. :contentReference[oaicite:6]{index=6}
    return ALLOWED_SIZES.get((img.width, img.height), "auto")


def openai_inpaint_pil(client: OpenAI, image_pil: Image.Image, mask_rgba_pil: Image.Image, keep_vehicle: str) -> Image.Image:
    prompt = (
        f"Remove all road users (people and vehicles) EXCEPT the main {keep_vehicle}. "
        f"Do NOT change the main {keep_vehicle} or any unmasked region. "
        f"Inpaint realistically to match the original lighting, road markings, shadows, and textures. "
        f"Do not add new objects."
    )

    img_buf = io.BytesIO()
    mask_buf = io.BytesIO()
    image_pil.save(img_buf, format="PNG")
    mask_rgba_pil.save(mask_buf, format="PNG")
    img_buf.seek(0)
    mask_buf.seek(0)
    img_buf.name = "image.png"   # ensures mime type is image/png
    mask_buf.name = "mask.png"

    # Images.edit expects image inputs as png/webp/jpg and mask as PNG with alpha=0 where to edit. :contentReference[oaicite:7]{index=7}
    result = client.images.edit(
        model=OPENAI_IMAGE_MODEL,
        image=[img_buf],
        mask=mask_buf,
        prompt=prompt,
        quality=OPENAI_QUALITY,
        input_fidelity=OPENAI_INPUT_FIDELITY,
        size=_size_str_for_image(image_pil),
        output_format=OPENAI_OUTPUT_FORMAT,
    )

    out_bytes = base64.b64decode(result.data[0].b64_json)
    out_img = Image.open(io.BytesIO(out_bytes)).convert("RGB")

    if out_img.size != image_pil.size:
        out_img = out_img.resize(image_pil.size, resample=Image.LANCZOS)
    return out_img


def _choose_tile_size(W: int, H: int) -> Tuple[int, int]:
    if W <= 1024 and H <= 1024:
        return SQUARE_TILE
    if W <= 1536 and H <= 1024:
        return LANDSCAPE_TILE
    if W <= 1024 and H <= 1536:
        return PORTRAIT_TILE
    return LANDSCAPE_TILE if W >= H else PORTRAIT_TILE


def _make_weight_map(w: int, h: int, has_left: bool, has_right: bool, has_top: bool, has_bottom: bool, overlap: int) -> np.ndarray:
    overlap = max(1, int(overlap))

    def ramp_1d(n: int, left: bool, right: bool) -> np.ndarray:
        x = np.ones(n, dtype=np.float32)
        if left:
            m = min(overlap, n)
            x[:m] = np.minimum(x[:m], np.linspace(0.0, 1.0, m, dtype=np.float32))
        if right:
            m = min(overlap, n)
            x[-m:] = np.minimum(x[-m:], np.linspace(1.0, 0.0, m, dtype=np.float32))
        return x

    wx = ramp_1d(w, has_left, has_right)
    wy = ramp_1d(h, has_top, has_bottom)
    return np.outer(wy, wx).astype(np.float32)


def inpaint_fullres_with_tiles(client: OpenAI, frame_rgb: Image.Image, full_mask_rgba: Image.Image, keep_vehicle: str, overlap: int) -> Image.Image:
    frame_np = np.array(frame_rgb, dtype=np.float32)
    mask_alpha = np.array(full_mask_rgba.split()[-1], dtype=np.uint8)
    H, W = mask_alpha.shape

    tile_w, tile_h = _choose_tile_size(W, H)
    step_x = max(1, tile_w - overlap)
    step_y = max(1, tile_h - overlap)

    accum = np.zeros((H, W, 3), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)

    xs = list(range(0, W, step_x))
    ys = list(range(0, H, step_y))

    for y0 in ys:
        for x0 in xs:
            x1 = min(W, x0 + tile_w)
            y1 = min(H, y0 + tile_h)

            region_alpha = mask_alpha[y0:y1, x0:x1]
            if region_alpha.min() == 255:
                continue  # nothing to edit in this tile

            region_w = x1 - x0
            region_h = y1 - y0

            tile_img = frame_rgb.crop((x0, y0, x1, y1))
            tile_mask = full_mask_rgba.crop((x0, y0, x1, y1))

            # pad to exact tile size
            if region_w != tile_w or region_h != tile_h:
                pad_img = Image.new("RGB", (tile_w, tile_h))
                pad_img.paste(tile_img, (0, 0))
                pad_mask = Image.new("RGBA", (tile_w, tile_h), (0, 0, 0, 255))
                pad_mask.paste(tile_mask, (0, 0))
                tile_img, tile_mask = pad_img, pad_mask

            edited_tile = openai_inpaint_pil(client, tile_img, tile_mask, keep_vehicle)
            edited_np = np.array(edited_tile, dtype=np.float32)[:region_h, :region_w, :]

            has_left = x0 > 0
            has_right = x1 < W
            has_top = y0 > 0
            has_bottom = y1 < H
            wmap = _make_weight_map(region_w, region_h, has_left, has_right, has_top, has_bottom, overlap)

            # only blend where edits are requested (alpha==0)
            edit_region = (region_alpha == 0).astype(np.float32)
            wmap = wmap * edit_region
            if wmap.max() <= 0:
                continue

            accum[y0:y1, x0:x1, :] += edited_np * wmap[..., None]
            wsum[y0:y1, x0:x1] += wmap

    out = frame_np.copy()
    m = wsum > 1e-6
    out[m] = accum[m] / wsum[m, None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


# =======================
# Encode video
# =======================
def encode_video_from_frames(original_video: Path, cleaned_dir: Path, output_video: Path, fps: float):
    output_video.parent.mkdir(parents=True, exist_ok=True)
    if output_video.exists() and not OVERWRITE_VIDEOS:
        return

    frames_pattern = str(cleaned_dir / "%06d.png")
    vf_even = "scale=trunc(iw/2)*2:trunc(ih/2)*2"

    if ENCODE_MODE == "lossless_mkv":
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", frames_pattern,
            "-i", str(original_video),
            "-map", "0:v:0", "-map", "1:a?",
            "-vf", vf_even,
            "-c:v", "libx264", "-qp", "0",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-shortest",
            str(output_video.with_suffix(".mkv"))
        ]
    else:
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", frames_pattern,
            "-i", str(original_video),
            "-map", "0:v:0", "-map", "1:a?",
            "-vf", vf_even,
            "-c:v", "libx264", "-crf", "16", "-preset", "slow",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            "-shortest",
            str(output_video.with_suffix(".mp4"))
        ]
    run(cmd)


# =======================
# Main
# =======================
def main():
    keep_vehicle = ALIASES.get(KEEP_VEHICLE.strip().lower(), KEEP_VEHICLE.strip().lower())
    videos = iter_videos(INPUT_ROOT)
    if not videos:
        raise SystemExit(f"No mp4 files found under: {INPUT_ROOT}")

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not found (common.get_secrets returned empty).")

    client = OpenAI(api_key=OPENAI_API_KEY)

    for video_path in tqdm(videos, desc="Videos"):
        rel = video_path.relative_to(INPUT_ROOT)

        video_out_dir = OUTPUT_ROOT / rel.parent / video_path.stem
        frames_dir = video_out_dir / "frames_png"
        masks_dir = video_out_dir / "masks"
        cleaned_dir = video_out_dir / "cleaned_png"
        output_dir = video_out_dir / "output_video"

        extract_frames_png(video_path, frames_dir, EXTRACT_FPS, OVERWRITE_FRAMES)
        frame_files = sorted(frames_dir.glob("*.png"))
        if not frame_files:
            continue

        fps = float(EXTRACT_FPS) if (EXTRACT_FPS is not None and EXTRACT_FPS > 0) else get_video_fps(video_path)

        cleaned_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        for frame_path in tqdm(frame_files, desc=f"Frames ({rel})", leave=False):
            out_frame = cleaned_dir / frame_path.name
            mask_path = masks_dir / frame_path.name

            if out_frame.exists() and not OVERWRITE_EDITS:
                continue

            with Image.open(frame_path) as im:
                frame_rgb = im.convert("RGB")
                w, h = frame_rgb.size

            try:
                keep_box, remove_boxes = get_boxes_from_model(client, frame_rgb, keep_vehicle)

                # If nothing to remove -> DO NOT call image edit (prevents hallucinations)
                if not remove_boxes:
                    shutil.copyfile(frame_path, out_frame)
                    continue

                mask_img = build_alpha_mask_from_boxes((w, h), keep_box, remove_boxes)
                mask_img.save(mask_path, format="PNG", optimize=True)

                alpha = np.array(mask_img.split()[-1], dtype=np.uint8)
                if alpha.min() == 255:
                    shutil.copyfile(frame_path, out_frame)
                    continue

                # Full-res safe: tile unless frame is exactly a supported size
                if (w, h) in ALLOWED_SIZES:
                    cleaned = openai_inpaint_pil(client, frame_rgb, mask_img, keep_vehicle)
                else:
                    cleaned = inpaint_fullres_with_tiles(client, frame_rgb, mask_img, keep_vehicle, TILE_OVERLAP)

                cleaned.save(out_frame, format="PNG", optimize=True)

            except Exception as e:
                print(f"[WARN] Failed on {frame_path}: {e}")
                shutil.copyfile(frame_path, out_frame)

        output_path = output_dir / f"{video_path.stem}_cleaned"
        encode_video_from_frames(video_path, cleaned_dir, output_path, fps)

    print("Done.")


if __name__ == "__main__":
    main()
