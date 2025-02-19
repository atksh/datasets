# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
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

"""BillSum Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """
@misc{kornilova2019billsum,
    title={BillSum: A Corpus for Automatic Summarization of US Legislation},
    author={Anastassia Kornilova and Vlad Eidelman},
    year={2019},
    eprint={1910.00523},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """
BillSum, summarization of US Congressional and California state bills.

There are several features:
  - text: bill text.
  - summary: summary of the bills.
  - title: title of the bills.
features for us bills. ca bills does not have.
  - text_len: number of chars in text.
  - sum_len: number of chars in summary.
"""

_URL = (
    "https://drive.google.com/uc?export=download&id=1g89WgFHMRbr4QrvA0ngh26PY081Nv3lx"
)

_DOCUMENT = "text"
_SUMMARY = "summary"


class Billsum(tfds.core.GeneratorBasedBuilder):
    """BillSum Dataset."""

    # 2.0.0 data source updated to filter near duplicates.
    # 3.0.0  none of the test examples are 'near duplicates' of an example in the
    #   train set AND they dont have the same title, regardless of similarity.
    VERSION = tfds.core.Version("3.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    _DOCUMENT: tfds.features.Text(),
                    _SUMMARY: tfds.features.Text(),
                    "title": tfds.features.Text(),
                }
            ),
            supervised_keys=(_DOCUMENT, _SUMMARY),
            homepage="https://github.com/FiscalNote/BillSum",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_path = dl_manager.download_and_extract(_URL)
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(dl_path, "us_train_data_final_OFFICIAL.jsonl"),
                    "key": "bill_id",
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "path": os.path.join(dl_path, "us_test_data_final_OFFICIAL.jsonl"),
                    "key": "bill_id",
                },
            ),
            tfds.core.SplitGenerator(
                name="ca_test",
                gen_kwargs={
                    "path": os.path.join(dl_path, "ca_test_data_final_OFFICIAL.jsonl"),
                    "key": "external_id",
                },
            ),
        ]

    def _generate_examples(self, path=None, key=None):
        """Yields examples."""
        with tf.io.gfile.GFile(path) as f:
            for line in f:
                # in us bills, json has fields:
                #   text, summary, title, bill_id, text_len, sum_len
                # in ca bills, json has fields:
                #   text, summary, title, external_id
                d = json.loads(line)
                yield d[key], {k: d[k] for k in [_DOCUMENT, _SUMMARY, "title"]}
