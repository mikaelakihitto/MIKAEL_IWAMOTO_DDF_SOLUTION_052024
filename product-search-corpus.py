# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.Wikipedia

# Lint as: python3
"""TREC Product Search dataset."""

import json

import datasets

_CITATION = """

"""

_DESCRIPTION = "dataset load script for TREC Product Search Corpus"

_DATASET_URLS = {
    'train': "https://huggingface.co/datasets/spacemanidol/product-search-corpus/resolve/main/corpus-simple.jsonl",
    #'train': "https://huggingface.co/datasets/spacemanidol/product-search-corpus/resolve/main/corpus.jsonl",
}


class ProductSearchCorpus(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(version=VERSION,
                               description="TREC Product Search Corpus"),
    ]

    def _info(self):
        features = datasets.Features(
            {'docid': datasets.Value('string'), 'title': datasets.Value('string'), 'text': datasets.Value('string')}
        )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="",
            # License for the dataset if available
            license="",
            # Citation for the dataset
            citation=_CITATION,
        )
    
    def _split_generators(self, dl_manager):
        if self.config.data_files:
            downloaded_files = self.config.data_files
        else:
            downloaded_files = dl_manager.download_and_extract(_DATASET_URLS)
        splits = [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "files": [downloaded_files[split]] if isinstance(downloaded_files[split], str) else downloaded_files[split],
                },
            ) for split in downloaded_files
        ]
        return splits

    def _generate_examples(self, files):
        """Yields examples."""
        for filepath in files:
            with open(filepath, encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    yield data['docid'], data


ds_buider = datasets.load_dataset_builder('spacemanidol/product-search-corpus')

print(ds_buider.info.features)