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

"""Config file that raises TypeError on import.

When trying loading the configuration file as a flag, the flags library catches
TypeError exceptions then recasts them as a IllegalFlagTypeError and rethrows
(b/63877430). The rethrow does not include the stacktrace from the original
exception, so we manually add the stracktrace in configflags.parse(). This is
tested in `ConfigFlagTest.testTypeError` in `config_overriding_test.py`.
"""


def type_error_function():
  raise TypeError('This is a TypeError.')


def get_config():
  return {'item': type_error_function()}
