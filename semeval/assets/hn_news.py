import json
import os
from typing import Dict, List  # add imports to the top of `assets.py`

import pandas as pd  # Add new imports to the top of `assets.py`
import requests
import base64
from io import BytesIO

import matplotlib.pyplot as plt

from dagster import AssetExecutionContext, MetadataValue, asset, MaterializeResult


@asset  # add the asset decorator to tell Dagster this is an asset
def topstory_ids() -> List:
    newstories_url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    top_new_story_ids = requests.get(newstories_url).json()[:100]

    return (
        top_new_story_ids  # return top_new_story_ids and the I/O manager will save it
    )


@asset()
def topstories(
    context: AssetExecutionContext,
    topstory_ids: List,  # add topstory_ids as a function argument
) -> pd.DataFrame:  # modify the return type signature

    results = []
    for item_id in topstory_ids:
        item = requests.get(
            f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
        ).json()
        results.append(item)

        if len(results) % 20 == 0:
            context.log.info(f"Got {len(results)} items so far.")

    df = pd.DataFrame(results)

    context.add_output_metadata(
        metadata={
            "num_records": len(df),
            "preview": MetadataValue.md(df.head().to_markdown()),
        }
    )

    return df  # return df and the I/O manager will save it


@asset()
def most_frequent_words(
    context: AssetExecutionContext,
    topstories: pd.DataFrame,  # add topstories as a function argument
) -> Dict:
    stopwords = ["a", "the", "an", "of", "to", "in", "for", "and", "with", "on", "is"]

    # loop through the titles and count the frequency of each word
    word_counts = {}
    for raw_title in topstories["title"]:
        title = raw_title.lower()
        for word in title.split():
            cleaned_word = word.strip(".,-!?:;()[]'\"-")
            if cleaned_word not in stopwords and len(cleaned_word) > 0:
                word_counts[cleaned_word] = word_counts.get(cleaned_word, 0) + 1

    # Get the top 25 most frequent words
    top_words = {
        pair[0]: pair[1]
        for pair in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:25]
    }

    # Make a bar chart of the top 25 words
    plt.figure(figsize=(10, 6))
    plt.bar(list(top_words.keys()), list(top_words.values()))
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 25 Words in Hacker News Titles")
    plt.tight_layout()

    # Convert the image to a saveable format
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    image_data = base64.b64encode(buffer.getvalue())

    # Convert the image to Markdown to preview it within Dagster
    md_content = f"![img](data:image/png;base64,{image_data.decode()})"

    context.add_output_metadata(metadata={"plot": MetadataValue.md(md_content)})

    return top_words  # return top_words and the I/O manager will save it