# Copyright 2026 The ML Collections Authors.
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

"""Dataclass configs whose annotations are postponed (stored as strings)."""

from __future__ import annotations

import dataclasses
from typing import Literal, Optional, Tuple


@dataclasses.dataclass
class ModelConfig:
  foo: int = 0
  rate: float = 1.0
  name: str = 'model'
  mode: Literal['a', 'b'] = 'a'
  sizes: Tuple[int, ...] = ()
  qux: Optional[int] = None


@dataclasses.dataclass
class SubConfig:
  model: ModelConfig = dataclasses.field(default_factory=ModelConfig)


@dataclasses.dataclass
class Config:
  value: int = 1
  sub: Optional[SubConfig] = None
  model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
