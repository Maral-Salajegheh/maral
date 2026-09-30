"""AXA connection and image preparation only; no page-classification prompts."""
import base64
import io
import os

from PIL import Image, ImageOps

MODEL_NAME = os.getenv("SECUREGPT_MODEL_NAME", "")
MODEL_VERSION = os.getenv("SECUREGPT_MODEL_VERSION", "")
TEMPERATURE = 0
SEED = 42
MAX_ATTEMPTS = 5
RETRY_DELAYS_SECONDS = [1, 2, 4, 8]
MIN_SHORT_SIDE_PX = 1024
MAX_LONG_SIDE_PX = 2400
JPEG_QUALITY = 92
MAX_PAYLOAD_BYTES = 4_000_000
FALLBACK_JPEG_QUALITIES = [80, 65]
FALLBACK_SCALE_FACTORS = [0.75, 0.5]


def validate_configuration():
    for name, value in (("SECUREGPT_MODEL_NAME", MODEL_NAME),
                        ("SECUREGPT_MODEL_VERSION", MODEL_VERSION)):
        if not value:
            raise ValueError(f"{name} is not set.")


def create_securegpt_client():
    validate_configuration()
    from axallm.securegpt.v2.providers import OpenAIProvider
    from axallm.securegpt.v2.securegpt import SecureGPT

    # Keep the constructor used by the supplied working AXA wrapper.
    # MODEL_VERSION remains an environment setting, not an invented SDK argument.
    return SecureGPT(provider=OpenAIProvider(), model=MODEL_NAME,
                     cache_prompts=True, seed=SEED, temperature=TEMPERATURE, debug=False)


def encode_jpeg(image, quality):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def resize_page(image):
    scale = max(1.0, MIN_SHORT_SIDE_PX / min(image.size))
    scale = min(scale, MAX_LONG_SIDE_PX / max(image.size))
    if scale == 1.0:
        return image
    size = tuple(max(1, round(side * scale)) for side in image.size)
    return image.resize(size, Image.Resampling.LANCZOS)


def fit_payload(image):
    for quality in [JPEG_QUALITY, *FALLBACK_JPEG_QUALITIES]:
        data_url = encode_jpeg(image, quality)
        if len(data_url) <= MAX_PAYLOAD_BYTES:
            return data_url
    for factor in FALLBACK_SCALE_FACTORS:
        size = tuple(max(1, round(side * factor)) for side in image.size)
        reduced = image.resize(size, Image.Resampling.LANCZOS)
        data_url = encode_jpeg(reduced, FALLBACK_JPEG_QUALITIES[-1])
        if len(data_url) <= MAX_PAYLOAD_BYTES:
            return data_url
    raise ValueError(f"Image payload cannot be reduced below {MAX_PAYLOAD_BYTES} bytes.")


def normalize_page_image(image_path):
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    return fit_payload(resize_page(image))


def is_retryable_securegpt_error(error):
    text = str(error).lower()
    markers = ["routing failed", "backend not available", "backend unavailable",
               "esg120", "timeout", "timed out", "connection reset",
               "temporarily unavailable", "too many requests", "rate limit"]
    markers += [f"{prefix}: {code}" for prefix in ("error code", "status code")
                for code in (429, 500, 502, 503, 504)]
    return any(marker in text for marker in markers)
