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

"""A enum to be used in a mock_config.

The enum can't be defined directly in the config because a config imported
through a flag is a separate instance of the config module. Therefore it would
define an own instance of the enum class which won't be equal to the same enum
from the config imported directly.
"""

import enum


class SporkType(enum.Enum):
  SPOON = 1
  SPORK = 2
  FORK = 3
