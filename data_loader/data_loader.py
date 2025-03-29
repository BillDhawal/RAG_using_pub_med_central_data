import os
import pandas as pd
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_dataset


def load_pubmed_data(
    start_idx: int = 500, num_samples: int = 100
) -> List[Dict[str, Any]]:
    try:
        dataset = load_dataset("DhruvDancingBuddha/osho_discourses", split="train", streaming=True)


        # Convert to list for easy access
        subset_list = list(dataset)

        # Format documents with consistent structure
        documents = [
            {
                "id": item["char_url"],
                "title": item["topic_name"],
                "content": item["all_txt"],
                "contents": item["topic_lesson_url"],
                "PMID": item["char_url"],
            }
            for item in subset_list
        ]

        return documents

    except Exception as e:
        print(f"Error loading PubMed data: {str(e)}")
        return []
