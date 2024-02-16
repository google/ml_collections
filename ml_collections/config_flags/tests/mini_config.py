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

"""Placeholder Config file."""


class MiniConfig(object):
  """Just a placeholder config."""

  def __init__(self):
    self.dict = {}
    self.field = False

  def __getitem__(self, key):
    return self.dict[key]

  def __contains__(self, key):
    return key in self.dict

  def __setitem__(self, key, value):
    self.dict[key] = value


def get_config():
  cfg = MiniConfig()
  cfg['entry_with_collision'] = False
  cfg.entry_with_collision = True
  return cfg
