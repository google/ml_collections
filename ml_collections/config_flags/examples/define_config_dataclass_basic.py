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

r"""Example of basic DEFINE_config_dataclass usage.

To run this example:
python define_config_dataclass_basic.py -- --my_config.field1=8 \
  --my_config.nested.field=2.1 --my_config.tuple='(1, 2, (1, 2))'
"""

import dataclasses
from typing import Any, Mapping, Sequence

from absl import app
from ml_collections import config_flags


@dataclasses.dataclass
class MyConfig:
  field1: int
  field2: str
  nested: Mapping[str, Any]
  tuple: Sequence[int]


config = MyConfig(
    field1=1,
    field2='tom',
    nested={'field': 2.23},
    tuple=(1, 2, 3),
)

_CONFIG = config_flags.DEFINE_config_dataclass('my_config', config)


def main(_):
  print(_CONFIG.value)


if __name__ == '__main__':
  app.run(main)
