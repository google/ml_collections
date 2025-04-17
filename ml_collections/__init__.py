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

"""ML Collections is a library of Python collections designed for ML usecases."""

from ml_collections.config_dict import ConfigDict
from ml_collections.config_dict import FieldReference
from ml_collections.config_dict import FrozenConfigDict

__all__ = ("ConfigDict", "FieldReference", "FrozenConfigDict")

# A new PyPI release will be pushed every time `__version__` is increased.
__version__ = "1.1.0"
