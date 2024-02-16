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

"""Config file that raises IOError on import.

The flags library tries to load configuration files in a few different ways.
For this it relies on catching IOError exceptions of the type "File not
found" and ignoring them to continue trying with a different loading method.
But we need to ensure that other types of IOError exceptions are propagated
correctly (b/63165566). This is tested in `ConfigFlagTest.testIOError` in
`config_overriding_test.py`.
"""

raise IOError('This is an IOError.')
