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

"""Config file that raises ValueError on import.

When trying loading the configuration file as a flag, the flags library catches
ValueError exceptions then recasts them as a IllegalFlagValueError and rethrows
(b/63877430). The rethrow does not include the stacktrace from the original
exception, so we manually add the stracktrace in configflags.parse(). This is
tested in `ConfigFlagTest.testValueError` in `config_overriding_test.py`.
"""


def value_error_function():
  raise ValueError('This is a ValueError.')


def get_config():
  return {'item': value_error_function()}
