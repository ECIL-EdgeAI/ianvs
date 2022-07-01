# Copyright 2022 The KubeEdge Authors.
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

from core.common import utils
from core.common.log import LOGGER
from core.testcasecontroller.testcase import TestCase


class TestCaseController:
    def __init__(self):
        self.test_cases = []

    def build_testcases(self, test_env, algorithms):
        for algorithm in algorithms:
            testcase = TestCase(test_env, algorithm)
            self.test_cases.append(testcase)

    def run_testcases(self, metrics, workspace):
        succeed_results = {}
        succeed_testcases = []
        for testcase in self.test_cases:
            testcase.prepare(metrics, workspace)
            try:
                res, time = (testcase.run(), utils.get_local_time())
            except Exception as err:
                LOGGER.error(f"testcase(id={testcase.id}) runs failed, error: {err}")
                continue

            succeed_results[testcase.id] = (res, time)
            succeed_testcases.append(testcase)

        return succeed_testcases, succeed_results
