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

"""DAS beamformed phantom images and paired clinical post-processed images test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_datasets import testing
from tensorflow_datasets.image import duke_ultrasound


class DukeUltrasoundTest(testing.DatasetBuilderTestCase):
    DATASET_CLASS = duke_ultrasound.DukeUltrasound
    OVERLAPPING_SPLITS = ["A", "B", "TRAIN"]

    SPLITS = {"train": 1, "test": 1, "validation": 1, "MARK": 1, "A": 1, "B": 1}

    DL_EXTRACT_RESULT = {
        "mark_data": "data",
        "phantom_data": "data",
        "train": "train.csv",
        "test": "test.csv",
        "validation": "validation.csv",
        "A": "train.csv",
        "B": "train.csv",
        "MARK": "mark.csv",
    }


if __name__ == "__main__":
    testing.test_main()
