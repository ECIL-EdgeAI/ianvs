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

from enum import Enum


# 常量是什么意思，写清楚
# 方便用户加常量

class DatasetFormat(Enum):
    """
    dataset format
    """
    CSV = "csv"
    TXT = "txt"


class ParadigmKind(Enum):
    """
    paradigm kind
    """
    SINGLE_TASK_LEARNING = "singletasklearning"
    INCREMENTAL_LEARNING = "incrementallearning"


class ModuleKind(Enum):
    """
    module kind
    """
    BASEMODEL = "basemodel"
    HARD_EXAMPLE_MINING = "hard_example_mining"


class SystemMetricKind(Enum):
    """
    system metric kind
    """
    DATA_TRANSFER_COUNT_RATIO = "data_transfer_count_ratio"
