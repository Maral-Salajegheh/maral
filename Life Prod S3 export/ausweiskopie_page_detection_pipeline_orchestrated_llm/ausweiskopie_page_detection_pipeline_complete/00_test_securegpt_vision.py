from __future__ import annotations

import json
from pathlib import Path

from securegpt_vision import (
    MODEL_NAME,
    MODEL_VERSION,
    create_securegpt_client,
    screen_image,
)


TEST_IMAGE_PATH = Path("test_page.png")


def main() -> None:
    if not TEST_IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Put one non-customer test image at: {TEST_IMAGE_PATH}"
        )

    client = create_securegpt_client()
    result = screen_image(client, TEST_IMAGE_PATH)

    print(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "model_version": MODEL_VERSION,
                "result": result,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
