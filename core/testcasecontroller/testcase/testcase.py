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

import os
import uuid

from core.common.constant import SystemMetricKind
from core.testcasecontroller.metrics import get_metric_func


class TestCase:
    def __init__(self, test_env, algorithm):
        """
        Distributed collaborative AI algorithm in certain test environment
        Parameters
        ----------
        test_env : instance
            The test environment of  distributed collaborative AI benchmark
            including samples, dataset setting, metrics
        algorithm : instance
            Distributed collaborative AI algorithm
        """
        self.test_env = test_env
        self.algorithm = algorithm

    def prepare(self, metrics, workspace):
        self.id = self._get_id()
        self.output_dir = self._get_output_dir(workspace)
        self.metrics = metrics

    def _get_output_dir(self, workspace):
        output_dir = os.path.join(workspace, self.algorithm.name)
        flag = True
        while flag:
            output_dir = os.path.join(workspace, self.algorithm.name, str(self.id))
            if not os.path.exists(output_dir):
                flag = False
        return output_dir

    def _get_id(self):
        return uuid.uuid1()

    def run(self):
        try:
            dataset = self.test_env.dateaset
            test_env_config = {}
            for k, v in self.test_env.__dict__.items():
                test_env_config[k] = v

            paradigm = self.algorithm.paradigm(dataset, self.output_dir, **test_env_config)
            res, system_metric_info = paradigm.run()
            test_result, system_metric_info = self.compute_metrics(res, dataset, **system_metric_info)

        except Exception as err:
            raise Exception(f"(paradigm={self.algorithm.paradigm}) pipeline runs failed, error: {err}")
        return test_result

    def compute_metrics(self, paradigm_result, dataset, **kwargs):
        metric_funcs = {}
        for metric_dict in self.test_env.metrics:
            metric_name, metric_func = get_metric_func(metric_dict=metric_dict)
            if callable(metric_func):
                metric_funcs.update({metric_name: metric_func})

        test_dataset_file = dataset.test_url
        test_dataset = self.dataset.load_data(test_dataset_file, data_type="eval overall", label=self.dataset.label)

        metric_res = {}
        for metric_name, metric_func in metric_funcs:
            if metric_name in SystemMetricKind.__members__.values():
                metric_res[metric_name] = metric_func(kwargs)
            else:
                metric_res[metric_name] = metric_func(test_dataset.y, paradigm_result)

        return metric_res
