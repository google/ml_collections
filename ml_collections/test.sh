#!/bin/bash
# Copyright 2020 The ML Collections Authors.
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

set -e
set -x

unset PYTHONPATH
virtualenv -p python3 .
source bin/activate
python --version

# Install/upgrade setuptools and wheel. >=40.1 is required to use
# `find_namespace_packages` in setup.py.
python -m pip install setuptools>=40.1 --upgrade
python -m pip install wheel --upgrade

# Run setup.py and install python dependencies.
python -m pip install .

# Install test requirements.
python -m pip install -r requirements-test.txt

# Install bazel.
sudo apt install curl gnupg
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list

sudo apt update && sudo apt install bazel

# Run bazel test
bazel test --test_output=errors ml_collections/...

echo "PASS"
