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

"""Example of ConfigDict usage of lock.

This example shows the roles and scopes of ConfigDict's lock().
"""

from absl import app
from ml_collections import config_dict


def main(_):
  cfg = config_dict.ConfigDict()
  cfg.integer_field = 123

  # Locking prohibits the addition and deletion of new fields but allows
  # modification of existing values. Locking happens automatically during
  # loading through flags.
  cfg.lock()
  try:
    cfg.intagar_field = 124  # Raises AttributeError and suggests valid field.
  except AttributeError as e:
    print(e)
  cfg.integer_field = -123  # Works fine.

  with cfg.unlocked():
    cfg.intagar_field = 1555  # Works fine too.

  print(cfg)


if __name__ == '__main__':
  app.run()
