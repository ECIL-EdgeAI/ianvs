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

import argparse

from core.common.log import LOGGER
from core.common import utils
from core.cmd.obj.benchmarkingjob import BenchmarkingJob


def main():
    args = parse_args()
    if args.benchmarking_config_file:
        try:
            config = utils.yaml2dict(args.benchmarking_config_file)
            job = BenchmarkingJob(config[str.lower(BenchmarkingJob.__name__)])
            job.run()
        except Exception as err:
            LOGGER.exception(f"benchmarking job runs failed, error: {err}.")
            return

        LOGGER.info(f"benchmarking job runs successfully!")


def parse_args():
    parser = argparse.ArgumentParser(description='AI Benchmarking Tool')
    parser.add_argument("-f", "--benchmarking_config_file",
                        nargs="?", default="~/benchmarking_config_file.yaml",
                        type=str,
                        help="the benchmarking config file must be yaml/yml file")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
