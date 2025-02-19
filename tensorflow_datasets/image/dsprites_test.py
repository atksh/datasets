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

"""dSprites dataset test."""
from tensorflow_datasets.image import dsprites
import tensorflow_datasets.testing as tfds_test


class DspritesTest(tfds_test.DatasetBuilderTestCase):
    DATASET_CLASS = dsprites.Dsprites
    SPLITS = {"train": 5}
    DL_EXTRACT_RESULT = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.hdf5"


class DspritesS3Test(DspritesTest):
    VERSION = "experimental_latest"


if __name__ == "__main__":
    tfds_test.test_main()
