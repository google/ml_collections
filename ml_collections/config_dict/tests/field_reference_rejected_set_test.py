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

"""Tests that rejected overrides preserve lazy configuration values."""

from absl.testing import absltest
from absl.testing import parameterized
from ml_collections import config_dict


class RejectedFieldReferenceSetTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('value', '1000'),
      ('reference', config_dict.FieldReference('1000')),
      ('placeholder', config_dict.FieldReference(None, field_type=str)),
  )
  def testRejectedTypePreservesComputation(self, invalid_value):
    steps_per_epoch = config_dict.FieldReference(100)
    epochs = config_dict.FieldReference(10)
    total_steps = steps_per_epoch * epochs
    warmup_steps = total_steps // 10

    with self.assertRaises(TypeError):
      total_steps.set(invalid_value)

    self.assertEqual(total_steps.get(), 1000)
    self.assertEqual(warmup_steps.get(), 100)
    epochs.set(20)
    self.assertEqual(total_steps.get(), 2000)
    self.assertEqual(warmup_steps.get(), 200)

  def testRejectedCyclePreservesComputation(self):
    steps_per_epoch = config_dict.FieldReference(100)
    epochs = config_dict.FieldReference(10)
    total_steps = steps_per_epoch * epochs
    warmup_steps = total_steps // 10

    with self.assertRaises(config_dict.MutabilityError):
      total_steps.set(warmup_steps)

    self.assertEqual(total_steps.get(), 1000)
    self.assertEqual(warmup_steps.get(), 100)
    epochs.set(20)
    self.assertEqual(total_steps.get(), 2000)
    self.assertEqual(warmup_steps.get(), 200)

  def testRejectedConfigUpdatePreservesTrainingSchedule(self):
    cfg = config_dict.ConfigDict({'steps_per_epoch': 100, 'epochs': 10})
    cfg.total_steps = cfg.get_ref('steps_per_epoch') * cfg.get_ref('epochs')
    cfg.warmup_steps = cfg.get_ref('total_steps') // 10

    with self.assertRaises(TypeError):
      cfg.update({'total_steps': '1000'})

    self.assertEqual(cfg.total_steps, 1000)
    self.assertEqual(cfg.warmup_steps, 100)
    cfg.update({'epochs': 20})
    self.assertEqual(cfg.total_steps, 2000)
    self.assertEqual(cfg.warmup_steps, 200)


if __name__ == '__main__':
  absltest.main()
