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

"""Tests for diabetic_retinopathy_detection dataset module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow_datasets import testing
from tensorflow_datasets.image import diabetic_retinopathy_detection


class DiabeticRetinopathyDetectionTest(testing.DatasetBuilderTestCase):
    DATASET_CLASS = diabetic_retinopathy_detection.DiabeticRetinopathyDetection
    SPLITS = {  # Expected number of examples on each split.
        "sample": 4,
        "train": 12,
        "validation": 6,
        "test": 6,
    }
    OVERLAPPING_SPLITS = ["sample"]  # contains examples from other examples


class DiabeticRetinopathyDetectionS3Test(DiabeticRetinopathyDetectionTest):
    VERSION = "experimental_latest"


if __name__ == "__main__":
    testing.test_main()
