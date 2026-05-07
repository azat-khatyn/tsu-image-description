from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from tsu_image_description.data.sample_manifests import write_example_manifests


def create_demo_image(path: str, text: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    img = Image.new("RGB", (512, 512), color=(245, 240, 230))
    draw = ImageDraw.Draw(img)
    draw.rectangle((30, 30, 482, 482), outline=(120, 80, 40), width=4)
    draw.text((60, 220), text, fill=(20, 20, 20))
    img.save(path_obj)


def main() -> None:
    write_example_manifests()

    create_demo_image("data/images/postcard.jpg", "Vintage postcard")
    create_demo_image("data/external/europeana/images/example_europeana.jpg", "Europeana sample")
    create_demo_image("data/external/wikiart/images/example_wikiart.jpg", "WikiArt sample")
    create_demo_image("data/external/artemis/images/example_artemis.jpg", "ArtEmis sample")

    print("Demo data created.")


if __name__ == "__main__":
    main()
