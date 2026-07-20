from pathlib import Path

from securegpt_vision import (
    AusweisPageResponse,
    SYSTEM_PROMPT,
    USER_PROMPT,
    create_securegpt_client,
    read_image_base64,
)


# مسیر یک عکس واقعی برای تست
IMAGE_PATH = Path("path/to/test_image.png")


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Test image not found: {IMAGE_PATH}"
        )

    client = create_securegpt_client()

    image_base64 = read_image_base64(
        IMAGE_PATH
    )

    response = client.new_chat(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=USER_PROMPT,
        user_image=image_base64,
        image_detail="high",
        response_model=AusweisPageResponse,
    )

    print("\n--- RESPONSE ---")
    print(response)

    print("\n--- RESPONSE TYPE ---")
    print(type(response))

    print("\n--- RESPONSE ATTRIBUTES ---")
    try:
        print(vars(response))
    except TypeError:
        print(
            "vars(response) is not supported "
            "for this response object."
        )

    print("\n--- POSSIBLE TOKEN USAGE ---")

    field_names = [
        "usage",
        "token_usage",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "metadata",
    ]

    for field_name in field_names:
        value = getattr(
            response,
            field_name,
            None,
        )

        print(
            f"{field_name}: {value}"
        )


if __name__ == "__main__":
    main()