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

"""Tests for tensorflow_datasets.core.units."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow_datasets import testing
from tensorflow_datasets.core import units


class UnitsTest(testing.TestCase):
    def test_none(self):
        self.assertEqual("?? GiB", units.size_str(None))

    def test_normal_sizes(self):
        self.assertEqual("1.50 PiB", units.size_str(1.5 * units.PiB))
        self.assertEqual("1.50 TiB", units.size_str(1.5 * units.TiB))
        self.assertEqual("1.50 GiB", units.size_str(1.5 * units.GiB))
        self.assertEqual("1.50 MiB", units.size_str(1.5 * units.MiB))
        self.assertEqual("1.50 KiB", units.size_str(1.5 * units.KiB))

    def test_bytes(self):
        self.assertEqual("150 bytes", units.size_str(150))


if __name__ == "__main__":
    testing.test_main()
