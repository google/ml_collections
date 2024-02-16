# Copyright 2024 The ML Collections Authors.
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

"""Config file with field references."""

from ml_collections import config_dict


def get_config():
  cfg = config_dict.ConfigDict()
  cfg.integer = config_dict.placeholder(object)
  cfg.string = config_dict.placeholder(object)
  cfg.nested = config_dict.placeholder(object)
  cfg.other_with_default = config_dict.placeholder(object)
  cfg.other_with_default = 123
  cfg.other_with_default_overitten = config_dict.placeholder(object)
  cfg.other_with_default_overitten = 123
  return cfg
