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

"""Config file that raises ImportError on import.

This simulates a config file that imports a module not available in the current
binary's deps (e.g. using the wrong binary target). We need to ensure that the
original ImportError traceback is preserved rather than being swallowed by the
config loading machinery. This is tested in `ConfigFlagTest.testImportError` in
`config_overriding_test.py`.
"""

raise ImportError('This is an ImportError.')
