import math
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.core.download import checksums

_URL = "gs://corpus-jp/ja_cc100.txt"


class JaCc100(tfds.core.GeneratorBasedBuilder):
    """A TFDS crude implementation of japanese subset of OSCAR dataset."""

    VERSION = tfds.core.Version("1.0.0", experiments={tfds.core.Experiment.S3: False,})

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({"text": tfds.features.Text(),}),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        # Add checksum for OSCAR datasetx to pre-defined checksums.
        downloaded_files = dl_manager.download_and_extract({"ja": _URL})
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=300,
                gen_kwargs={"filepaths": downloaded_files},
            ),
        ]

    def _generate_examples(self, filepaths):
        with tf.io.gfile.GFile(filepaths["ja"]) as text_file:
            for line in text_file:
                yield {"text": line.strip()}
