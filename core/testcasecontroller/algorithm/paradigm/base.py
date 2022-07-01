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

from sedna.core.incremental_learning import IncrementalLearning

from core.common.constant import ModuleKind, ParadigmKind


class ParadigmBase:
    def __init__(self, modules, dataset, workspace, **kwargs):
        self.modules = modules
        self.dataset = dataset
        self.workspace = workspace
        os.environ["LOCAL_TEST"] = "TRUE"

    def dataset_output_dir(self):
        output_dir = os.path.join(self.workspace, "dataset")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    def build_paradigm_job(self, paradigm_kind):
        for module in self.modules:
            if module.kind == ModuleKind.BASEMODEL.value:
                basemodel = module.get_basemodel()
            elif module.kind == ModuleKind.HARD_EXAMPLE_MINING.value:
                hard_example_mining = module.get_hard_example_mining_func()

        job = self.module_funcs[ModuleKind.BASEMODEL.value]
        if paradigm_kind == ParadigmKind.INCREMENTAL_LEARNING.value:
            job = IncrementalLearning(estimator=basemodel, hard_example_mining=hard_example_mining)

        return job
