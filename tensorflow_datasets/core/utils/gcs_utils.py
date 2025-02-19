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

"""Utilities for accessing TFDS GCS buckets."""

import posixpath
from xml.etree import ElementTree

import concurrent.futures
import requests
import tensorflow as tf

from tensorflow_datasets.core import utils

GCS_URL = "http://storage.googleapis.com"

# for dataset_info/
GCS_BUCKET = posixpath.join(GCS_URL, "tfds-data")
GCS_DATASET_INFO_DIR = "dataset_info"
GCS_DATASETS_DIR = "datasets"


def download_gcs_file(path, out_fname=None, prefix_filter=None):
    """Download a file from GCS, optionally to a file."""
    url = posixpath.join(GCS_BUCKET, path)
    if prefix_filter:
        url += "?prefix=%s" % prefix_filter
    stream = bool(out_fname)
    resp = requests.get(url, stream=stream)
    if not resp.ok:
        raise ValueError("GCS bucket inaccessible")
    if out_fname:
        with tf.io.gfile.GFile(out_fname, "wb") as f:
            for chunk in resp.iter_content(1024):
                f.write(chunk)
    else:
        return resp.content


@utils.memoize()
def gcs_files(prefix_filter=None):
    """List all files in GCS bucket."""
    top_level_xml_str = download_gcs_file("", prefix_filter=prefix_filter)
    xml_root = ElementTree.fromstring(top_level_xml_str)
    filenames = [el[0].text for el in xml_root if el.tag.endswith("Contents")]
    return filenames


def gcs_dataset_info_files(dataset_dir):
    """Return paths to GCS files in the given dataset directory."""
    prefix = posixpath.join(GCS_DATASET_INFO_DIR, dataset_dir, "")
    # Filter for this dataset
    filenames = [
        el
        for el in gcs_files(prefix_filter=prefix)
        if el.startswith(prefix) and len(el) > len(prefix)
    ]
    return filenames


def is_dataset_on_gcs(dataset_name):
    """If the dataset is available on the GCS bucket gs://tfds-data/datasets."""
    dir_name = posixpath.join(GCS_DATASETS_DIR, dataset_name)
    return len(gcs_files(prefix_filter=dir_name)) > 2


def download_gcs_dataset(
    dataset_name, local_dataset_dir, max_simultaneous_downloads=50
):
    """Downloads prepared GCS dataset to local dataset directory."""
    gcs_paths_to_dl = gcs_files(posixpath.join(GCS_DATASETS_DIR, dataset_name))
    with utils.async_tqdm(
        total=len(gcs_paths_to_dl), desc="Dl Completed...", unit=" file"
    ) as pbar:

        def _copy_from_gcs(gcs_path):
            local_path = posixpath.join(local_dataset_dir, posixpath.basename(gcs_path))
            download_gcs_file(gcs_path, local_path)
            pbar.update(1)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_simultaneous_downloads
        ) as executor:
            futures = [
                executor.submit(_copy_from_gcs, path) for path in gcs_paths_to_dl
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
