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

"""CNN/DailyMail Summarization dataset, non-anonymized version."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import hashlib
import os
from absl import logging
import tensorflow as tf
import tensorflow_datasets.public_api as tfds


_DESCRIPTION = """\
CNN/DailyMail non-anonymized summarization dataset.

There are two features:
  - article: text of news article, used as the document to be summarized
  - highlights: joined text of highlights with <s> and </s> around each
    highlight, which is the target summary
"""

# The second citation introduces the source data, while the first
# introduces the specific form (non-anonymized) we use here.
_CITATION = """\
@article{DBLP:journals/corr/SeeLM17,
  author    = {Abigail See and
               Peter J. Liu and
               Christopher D. Manning},
  title     = {Get To The Point: Summarization with Pointer-Generator Networks},
  journal   = {CoRR},
  volume    = {abs/1704.04368},
  year      = {2017},
  url       = {http://arxiv.org/abs/1704.04368},
  archivePrefix = {arXiv},
  eprint    = {1704.04368},
  timestamp = {Mon, 13 Aug 2018 16:46:08 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/SeeLM17},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

@inproceedings{hermann2015teaching,
  title={Teaching machines to read and comprehend},
  author={Hermann, Karl Moritz and Kocisky, Tomas and Grefenstette, Edward and Espeholt, Lasse and Kay, Will and Suleyman, Mustafa and Blunsom, Phil},
  booktitle={Advances in neural information processing systems},
  pages={1693--1701},
  year={2015}
}
"""

_DL_URLS = {
    # pylint: disable=line-too-long
    "cnn_stories": "https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTHk4NFg2SndKcjQ",
    "dm_stories": "https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs",
    "test_urls": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_test.txt",
    "train_urls": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_train.txt",
    "val_urls": "https://raw.githubusercontent.com/abisee/cnn-dailymail/master/url_lists/all_val.txt",
    # pylint: enable=line-too-long
}

_HIGHLIGHTS = "highlights"
_ARTICLE = "article"
_SUPPORTED_VERSIONS = [
    tfds.core.Version("0.0.2", experiments={tfds.core.Experiment.S3: False}),
    # Same data as 0.0.2
    tfds.core.Version(
        "1.0.0", "New split API (https://tensorflow.org/datasets/splits)"
    ),
]

# Having the model predict newline separators makes it easier to evaluate
# using summary-level ROUGE.
_DEFAULT_VERSION = tfds.core.Version("2.0.0", "Separate target sentences with newline.")


class CnnDailymailConfig(tfds.core.BuilderConfig):
    """BuilderConfig for CnnDailymail."""

    @tfds.core.disallow_positional_args
    def __init__(self, text_encoder_config=None, **kwargs):
        """BuilderConfig for CnnDailymail.

    Args:
      text_encoder_config: `tfds.features.text.TextEncoderConfig`, configuration
        for the `tfds.features.text.TextEncoder` used for the CnnDailymail
        (text) features
      **kwargs: keyword arguments forwarded to super.
    """
        # Version history:
        # 1.0.0: S3 (new shuffling, sharding and slicing mechanism).
        # 0.0.2: Initial version.
        super(CnnDailymailConfig, self).__init__(
            version=_DEFAULT_VERSION, supported_versions=_SUPPORTED_VERSIONS, **kwargs
        )
        self.text_encoder_config = (
            text_encoder_config or tfds.features.text.TextEncoderConfig()
        )


def _get_url_hashes(path):
    """Get hashes of urls in file."""
    urls = _read_text_file(path)

    def url_hash(u):
        h = hashlib.sha1()
        try:
            u = u.encode("utf-8")
        except UnicodeDecodeError:
            logging.error("Cannot hash url: %s", u)
        h.update(u)
        return h.hexdigest()

    return {url_hash(u): True for u in urls}


def _find_files(dl_paths, publisher, url_dict):
    """Find files corresponding to urls."""
    if publisher == "cnn":
        top_dir = os.path.join(dl_paths["cnn_stories"], "cnn", "stories")
    elif publisher == "dm":
        top_dir = os.path.join(dl_paths["dm_stories"], "dailymail", "stories")
    else:
        logging.fatal("Unsupported publisher: %s", publisher)
    files = tf.io.gfile.listdir(top_dir)

    ret_files = []
    for p in files:
        basename = os.path.basename(p)
        if basename[0 : basename.find(".story")] in url_dict:
            ret_files.append(os.path.join(top_dir, p))
    return ret_files


def _subset_filenames(dl_paths, split):
    """Get filenames for a particular split."""
    assert isinstance(dl_paths, dict), dl_paths
    # Get filenames for a split.
    if split == tfds.Split.TRAIN:
        urls = _get_url_hashes(dl_paths["train_urls"])
    elif split == tfds.Split.VALIDATION:
        urls = _get_url_hashes(dl_paths["val_urls"])
    elif split == tfds.Split.TEST:
        urls = _get_url_hashes(dl_paths["test_urls"])
    else:
        logging.fatal("Unsupported split: %s", split)
    cnn = _find_files(dl_paths, "cnn", urls)
    dm = _find_files(dl_paths, "dm", urls)
    return cnn + dm


DM_SINGLE_CLOSE_QUOTE = u"\u2019"  # unicode
DM_DOUBLE_CLOSE_QUOTE = u"\u201d"
# acceptable ways to end a sentence
END_TOKENS = [
    ".",
    "!",
    "?",
    "...",
    "'",
    "`",
    '"',
    DM_SINGLE_CLOSE_QUOTE,
    DM_DOUBLE_CLOSE_QUOTE,
    ")",
]


def _read_text_file(text_file):
    lines = []
    with tf.io.gfile.GFile(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def _get_art_abs(story_file, tfds_version):
    """Get abstract (highlights) and article from a story file path."""
    # Based on https://github.com/abisee/cnn-dailymail/blob/master/
    #     make_datafiles.py

    lines = _read_text_file(story_file)

    # Lowercase everything
    lines = [line.lower() for line in lines]

    # Put periods on the ends of lines that are missing them
    # (this is a problem in the dataset because many image captions don't end in
    # periods; consequently they end up in the body of the article as run-on
    # sentences)
    def fix_missing_period(line):
        """Adds a period to a line that is missing a period."""
        if "@highlight" in line:
            return line
        if not line:
            return line
        if line[-1] in END_TOKENS:
            return line
        return line + " ."

    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for line in lines:
        if not line:
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = " ".join(article_lines)

    if tfds_version >= "2.0.0":
        abstract = "\n".join(highlights)
    else:
        abstract = " ".join(highlights)

    return article, abstract


class CnnDailymail(tfds.core.GeneratorBasedBuilder):
    """CNN/DailyMail non-anonymized summarization dataset."""

    BUILDER_CONFIGS = [
        CnnDailymailConfig(name="plain_text", description="Plain text",),
        CnnDailymailConfig(
            name="bytes",
            description=(
                "Uses byte-level text encoding with "
                "`tfds.features.text.ByteTextEncoder`"
            ),
            text_encoder_config=tfds.features.text.TextEncoderConfig(
                encoder=tfds.features.text.ByteTextEncoder()
            ),
        ),
        CnnDailymailConfig(
            name="subwords32k",
            description=(
                "Uses `tfds.features.text.SubwordTextEncoder` with " "32k vocab size"
            ),
            text_encoder_config=tfds.features.text.TextEncoderConfig(
                encoder_cls=tfds.features.text.SubwordTextEncoder, vocab_size=2 ** 15
            ),
        ),
    ]

    def _info(self):
        # Should return a tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    _ARTICLE: tfds.features.Text(
                        encoder_config=self.builder_config.text_encoder_config
                    ),
                    _HIGHLIGHTS: tfds.features.Text(
                        encoder_config=self.builder_config.text_encoder_config
                    ),
                }
            ),
            supervised_keys=(_ARTICLE, _HIGHLIGHTS),
            homepage="https://github.com/abisee/cnn-dailymail",
            citation=_CITATION,
        )

    def _vocab_text_gen(self, paths):
        for _, ex in self._generate_examples(paths):
            yield " ".join([ex[_ARTICLE], ex[_HIGHLIGHTS]])

    def _split_generators(self, dl_manager):
        dl_paths = dl_manager.download_and_extract(_DL_URLS)
        train_files = _subset_filenames(dl_paths, tfds.Split.TRAIN)
        # Generate shared vocabulary
        # maybe_build_from_corpus uses SubwordTextEncoder if that's configured
        self.info.features[_ARTICLE].maybe_build_from_corpus(
            self._vocab_text_gen(train_files)
        )
        encoder = self.info.features[_ARTICLE].encoder
        # Use maybe_set_encoder because the encoder may have been restored from
        # package data.
        self.info.features[_HIGHLIGHTS].maybe_set_encoder(encoder)

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN, num_shards=100, gen_kwargs={"files": train_files}
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                num_shards=10,
                gen_kwargs={
                    "files": _subset_filenames(dl_paths, tfds.Split.VALIDATION)
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                num_shards=10,
                gen_kwargs={"files": _subset_filenames(dl_paths, tfds.Split.TEST)},
            ),
        ]

    def _generate_examples(self, files):
        for p in files:
            article, highlights = _get_art_abs(p, self.version)
            if not article or not highlights:
                continue
            fname = os.path.basename(p)
            yield fname, {_ARTICLE: article, _HIGHLIGHTS: highlights}
