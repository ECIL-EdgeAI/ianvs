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

from sedna.common.class_factory import ClassFactory, ClassType

from core.common import utils
from core.common.constant import ModuleKind


class Module:
    def __init__(self):
        self.kind: str = ""
        self.name: str = ""
        self.url: str = ""
        self.hyperparameters: dict = {}

    def check_fields(self):
        if not self.kind and not isinstance(self.kind, str):
            raise ValueError(f"module kind({self.kind}) must be provided and be string type.")
        kinds = [e.value for e in ModuleKind.__members__.values()]
        if self.kind not in kinds:
            raise ValueError(f"not support module kind({self.kind}."
                             f"the following paradigms can be selected: {kinds}")
        if not self.name and not isinstance(self.name, str):
            raise ValueError(f"module name({self.name}) must be provided and be string type.")
        if not isinstance(self.url, str):
            raise ValueError(f"module url({self.url}) must be string type.")
        if not isinstance(self.hyperparameters, dict):
            raise ValueError(f"module hyperparameters({self.hyperparameters}) must be dictionary type.")

    def get_basemodel(self):
        if not self.url:
            raise ValueError(f"url({self.url}) of basemodel module must be provided.")

        utils.load_module(self.url)
        try:
            basemodel = ClassFactory.get_cls(type_name=ClassType.GENERAL, t_cls_name=self.name)(
                **self.hyperparameters)
        except Exception as err:
            raise Exception(f"basemodel module loads class(name={self.name}) failed, error: {err}.")

        return basemodel

    def get_hard_example_mining_func(self):
        if self.url:
            utils.load_module(self.url)
            try:
                hard_example_mining = ClassFactory.get_cls(type_name=ClassType.HEM, t_cls_name=self.name)(
                    **self.hyperparameters)
            except Exception as err:
                raise Exception(f"hard_example_mining module loads class(name={self.name}) failed, error: {err}.")
        else:
            hard_example_mining = {"method": self.name}
            if self.hyperparameters:
                hard_example_mining["param"] = self.hyperparameters
        return hard_example_mining
