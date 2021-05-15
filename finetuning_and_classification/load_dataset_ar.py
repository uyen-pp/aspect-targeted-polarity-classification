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
# limitations under the License.

"""TODO: Add a description here."""

from __future__ import absolute_import, division, print_function

import csv
import json
import os
import datasets
import ast
import numpy as np

class ARDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for SQUAD."""

    def __init__(self, **kwargs):
        """BuilderConfig for SQUAD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ARDatasetConfig, self).__init__(**kwargs)

_ASPECTS = ['Hạn sử dụng',
        'Kết cấu bột',
        'Hướng dẫn sử dụng',
        'Khuyến mãi & Quà tặng',
        'Nguồn gốc xuất xứ',
        'Tiêu hóa và hấp thụ',
        'Phát triển trí não',
        'Dịch vụ giao hàng',
        'Phát triển thể chất',
        'Chăm sóc khách hàng',
        'Tuổi',
        'Dị ứng',
        'Phân phối',
        'Bao bì đóng gói',
        'Thương mại điện tử',
        'Thành phần dinh dưỡng',
        'Giá',
        'Hệ miễn dịch',
        'Hương vị',
        'Chương trình tiếp thị',
        'Chất lượng chung',
        'Tiện lợi',
        'Hỗ trợ giấc ngủ']

# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class ARDataset(datasets.GeneratorBasedBuilder):
    
    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset

        features = datasets.Features(
            {
                "sentence": datasets.Value("string"),
                "num_tag": datasets.Value("int8"),
                "labels": datasets.Sequence(
                    feature = datasets.ClassLabel(names=_ASPECTS),
                )
                #These are the features of your dataset like images, labels ...
            }
        )

        return datasets.DatasetInfo(
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
        )

    def _split_generators(self, dl_manager):

        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive 
        data_files = dict()

        # Get the train file
        try:
            data_files.update({"train": self.config.data_files['train']})
        except: 
            data_files.update({"train": None})
        

        # Get validate file
        try:
            data_files.update({"validation": self.config.data_files['validation']})
        except: 
            data_files.update({"validation": None})
            
        # Get test file
        try:
            data_files.update({"test": self.config.data_files['test']})
        except: 
            data_files.update({"test": None})
        

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_files["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_files["validation"],
                    "split": "validation"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_files["test"],
                    "split": "test"
                },
            )
        ]

    def _generate_examples(self, filepath, split):
        """ Yields examples. """
        # TODO: This method will receive as arguments the `gen_kwargs` defined in the previous `_split_generators` method.
        # It is in charge of opening the given file and yielding (key, example) tuples from the dataset
        # The key is not important, it's more here for legacy reason (legacy from tfds)
        if filepath is not None:

            with open(filepath, encoding="utf-8") as csv_file:
                csv_reader = csv.reader(
                    csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
                )
                next(csv_reader)  # Skip header row.
                for row in csv_reader:
                    id_, text, label = row
                    aspects = ast.literal_eval(label)
                    yield id_, {
                        "sentence": text.strip('"'), 
                        "num_tag": len(aspects),
                        "labels": aspects
                    }
                