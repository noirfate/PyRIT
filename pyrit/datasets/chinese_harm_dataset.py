# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import List, Literal, Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.datasets.dataset_helper import fetch_examples
from pyrit.models import SeedPromptDataset
from pyrit.models.seed_prompt import SeedPrompt


def fetch_chinese_harm_dataset(
    cache: bool = True,
    categories: Optional[
        List[
            Literal[
                "国家形象",
                "历史敏感事件",
                "国际关系",
                "国家分裂势力",
                "政策制度",
            ]
        ]
    ] = None,
) -> SeedPromptDataset:
    """
    chinese harm dataset examples
    """
    dataset = fetch_examples(
        source=str(Path(DATASETS_PATH) / "data" / "chinese_harm_dataset.json"), source_type="file", cache=cache
    )

    filtered = dataset["data"]  # type: ignore

    if categories:
        category_set = set(categories or [])

        # Include an entry if it matches ANY specified main category OR ANY specified subcategory
        filtered = [
            item
            for item in filtered
            if (category_set and any(cat in category_set for cat in item["categories"]))
        ]

    seed_prompts = [
        SeedPrompt(
            value=item["prompt"],
            data_type="text",
            name="Chinese Harm Dataset",
            dataset_name="Chinese Harm Dataset",
            harm_categories=item["categories"],
            description="""Chinese Harm Dataset is a set of 520 harmful behaviors formulated as instructions. This dataset
            has been extended to include harm categories for better filtering and analysis. The adversary's goal
            is instead to find a single attack string that will cause the model to generate any response that
            attempts to comply with the instruction, and to do so over as many harmful behaviors as possible.""",
            source="",
        )
        for item in filtered
    ]

    seed_prompt_dataset = SeedPromptDataset(prompts=seed_prompts)
    return seed_prompt_dataset