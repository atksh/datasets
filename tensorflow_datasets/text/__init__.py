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

"""Text datasets."""

from tensorflow_datasets.text.c4 import C4
from tensorflow_datasets.text.definite_pronoun_resolution import (
    DefinitePronounResolution,
)
from tensorflow_datasets.text.esnli import Esnli
from tensorflow_datasets.text.gap import Gap
from tensorflow_datasets.text.glue import Glue
from tensorflow_datasets.text.imdb import IMDBReviews
from tensorflow_datasets.text.imdb import IMDBReviewsConfig
from tensorflow_datasets.text.lm1b import Lm1b
from tensorflow_datasets.text.lm1b import Lm1bConfig
from tensorflow_datasets.text.math_dataset import MathDataset
from tensorflow_datasets.text.multi_nli import MultiNLI
from tensorflow_datasets.text.multi_nli_mismatch import MultiNLIMismatch
from tensorflow_datasets.text.scicite import Scicite
from tensorflow_datasets.text.snli import Snli
from tensorflow_datasets.text.squad import Squad
from tensorflow_datasets.text.super_glue import SuperGlue
from tensorflow_datasets.text.trivia_qa import TriviaQA
from tensorflow_datasets.text.wikipedia import Wikipedia
from tensorflow_datasets.text.xnli import Xnli
from tensorflow_datasets.text.ja_cc100 import JaCc100  # TODO(ja_cc100) Sort alphabetically
from tensorflow_datasets.text.oscar_ja import OscarJa  # TODO(oscar_ja) Sort alphabetically
