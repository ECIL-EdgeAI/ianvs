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

from core.common.constant import ParadigmKind
from core.testcasecontroller.paradigm.base import ParadigmBase
from core.common.log import LOGGER


class SingleTaskLearning(ParadigmBase):
    """ SingleTaskLearning pipeline """

    def __init__(self, modules, dataset, workspace, **kwargs):
        super(SingleTaskLearning, self).__init__(modules, dataset, workspace)
        self.initial_model = kwargs.get("initial_model_url")

    def run(self):
        current_model_url = self.initial_model
        train_output_dir = os.path.join(self.workspace, f"output/train/")
        os.environ["BASE_MODEL_URL"] = current_model_url

        job = self.build_paradigm_job(ParadigmKind.SINGLE_TASK_LEARNING.value)
        train_dataset = self.dataset.load_data(self.dataset.train_url, "train")
        job.train(train_dataset)
        trained_model_path = job.save(train_output_dir)

        self._print_status_log("inference", "starting")
        inference_dataset = self.dataset.load_data(self.dataset.test_url, "inference")
        inference_output_dir = os.path.join(self.workspace, f"output/inference/")
        os.environ["RESULT_SAVED_URL"] = inference_output_dir
        job.load(trained_model_path)
        infer_res = job.predict(inference_dataset.x)

        return infer_res