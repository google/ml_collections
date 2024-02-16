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

"""Config flags module."""

from .config_flags import DEFINE_config_dataclass
from .config_flags import DEFINE_config_dict
from .config_flags import DEFINE_config_file
from .config_flags import get_config_filename
from .config_flags import get_override_values
from .config_flags import register_flag_parser
from .config_flags import register_flag_parser_for_type

__all__ = (
    "DEFINE_config_dataclass",
    "DEFINE_config_dict",
    "DEFINE_config_file",
    "get_config_filename",
    "get_override_values",
    "register_flag_parser",
    "register_flag_parser_for_type",
)
