from pathlib import Path
import argparse

from securegpt_vision_original_plus_confidence_retry import (
    AusweisPageResponse,
    SYSTEM_PROMPT,
    USER_PROMPT,
    create_securegpt_client,
    image_to_data_url,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test one image and inspect the raw SecureGPT response."
    )
    parser.add_argument(
        "image_path",
        type=Path,
        help="Path to one PNG or JPEG image.",
    )
    args = parser.parse_args()

    client = create_securegpt_client()
    image_data_url = image_to_data_url(args.image_path)

    response = client.new_chat(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        user_image=image_data_url,
        response_model=AusweisPageResponse,
    )

    print("\n--- RESPONSE ---")
    print(response)

    print("\n--- RESPONSE ATTRIBUTES ---")
    try:
        print(vars(response))
    except TypeError:
        print("vars(response) is not available for this response type.")

    print("\n--- POSSIBLE USAGE FIELDS ---")
    for field_name in (
        "usage",
        "token_usage",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "metadata",
    ):
        print(f"{field_name}: {getattr(response, field_name, None)}")


if __name__ == "__main__":
    main()
