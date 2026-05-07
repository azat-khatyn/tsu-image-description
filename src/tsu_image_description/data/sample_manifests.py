from __future__ import annotations

from tsu_image_description.data.schema import DatasetRecord, StructuredMetadata
from tsu_image_description.data.io import save_dataset_records


def write_example_manifests() -> None:
    local_gold = [
        DatasetRecord(
            sample_id="local_0001",
            image_path="postcard.jpg",
            source_name="local_gold",
            source_type="local_curated",
            caption_ru="Винтажная открытка с праздничным сюжетом и декоративной графикой.",
            caption_en="A vintage postcard with a festive scene and decorative graphics.",
            search_text_ru="открытка праздник винтаж декоративная графика",
            metadata=StructuredMetadata(
                image_type="postcard",
                style="vintage",
                period="1900_1917",
                emotion="nostalgia",
                tags=["postcard", "holiday", "vintage"],
                objects=["illustration", "decorative elements"],
                cultural_context=["printed holiday postcard tradition"],
            ),
            split="train",
        ),
    ]

    europeana = [
        DatasetRecord(
            sample_id="eu_0001",
            image_path="example_europeana.jpg",
            source_name="europeana",
            source_type="cultural_heritage",
            caption_en="An antique postcard showing a city street scene.",
            search_text_en="antique postcard city street historical",
            metadata=StructuredMetadata(
                image_type="postcard",
                style="retro",
                period="1900_1917",
                emotion="neutral",
                tags=["city", "street", "historical", "postcard"],
            ),
            split="train",
        ),
    ]

    wikiart = [
        DatasetRecord(
            sample_id="wiki_0001",
            image_path="example_wikiart.jpg",
            source_name="wikiart",
            source_type="art_style",
            caption_en="A painting with strong decorative vintage aesthetics.",
            metadata=StructuredMetadata(
                style="painting",
                emotion="neutral",
                tags=["painting", "decorative"],
            ),
            split="train",
        ),
    ]

    artemis = [
        DatasetRecord(
            sample_id="artemis_0001",
            image_path="example_artemis.jpg",
            source_name="artemis",
            source_type="emotion_art",
            caption_en="A sentimental illustration with soft tones.",
            metadata=StructuredMetadata(
                style="illustrative",
                emotion="nostalgia",
                tags=["sentimental", "art"],
            ),
            split="train",
        ),
    ]

    save_dataset_records("data/manifests/local_gold.jsonl", local_gold)
    save_dataset_records("data/manifests/europeana.jsonl", europeana)
    save_dataset_records("data/manifests/wikiart.jsonl", wikiart)
    save_dataset_records("data/manifests/artemis.jsonl", artemis)


if __name__ == "__main__":
    write_example_manifests()
    print("Example manifests created.")
