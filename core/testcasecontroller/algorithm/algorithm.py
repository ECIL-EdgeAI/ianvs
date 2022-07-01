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

from core.common.constant import ParadigmKind
from core.testcasecontroller.algorithm.paradigm import SingleTaskLearning, IncrementalLearning


class Algorithm:
    def __init__(self):
        self.name: str = ""
        self.paradigm_kind: str = ""
        self.incremental_learning_data_setting: dict = {
            "train_ratio": 0.8,
            "splitting_method": "default"
        }
        self.initial_model_url: str = ""
        self.modules: dict = {}

    def paradigm(self, dataset, workspace, **kwargs):
        config = kwargs
        for k, v in self.__dict__.items():
            config[k] = v

        if self.paradigm_kind == ParadigmKind.SINGLE_TASK_LEARNING.value:
            return SingleTaskLearning(self.modules, dataset, workspace, **config)
        elif self.paradigm_kind == ParadigmKind.INCREMENTAL_LEARNING.value:
            return IncrementalLearning(self.modules, dataset, workspace, **config)

    def check_fields(self):
        if not self.name and not isinstance(self.name, str):
            raise ValueError(f"algorithm name({self.name}) must be provided and be string type.")
        if not self.paradigm_kind and not isinstance(self.paradigm_kind, str):
            raise ValueError(f"algorithm paradigm({self.paradigm_kind}) must be provided and be string type.")
        paradigm_kinds = [e.value for e in ParadigmKind.__members__.values()]
        if self.paradigm_kind not in paradigm_kinds:
            raise ValueError(f"not support paradigm({self.paradigm_kind})."
                             f"the following paradigms can be selected: {paradigm_kinds}")
        if not isinstance(self.incremental_learning_data_setting, dict):
            raise ValueError(
                f"algorithm incremental_learning_data_setting({self.incremental_learning_data_setting} "
                f"must be dictionary type.")
        if not isinstance(self.initial_model_url, str):
            raise ValueError(f"algorithm initial_model_url({self.initial_model_url}) must be string type.")
        for m in self.modules:
            m.check_fields()
