# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import json
import yaml
import six
import logging

class ArgumentGroup(object):
    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, **kwargs):
        type = str2bool if type == bool else type
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)

class PDConfig(object):
    """
    A high-level API for managing configuration files in PaddlePaddle.
    Can jointly work with command-line-arugment, json files and yaml files.
    """

    def __init__(self, json_file="", yaml_file="", fuse_args=True):
        """
            Init funciton for PDConfig.
            json_file: the path to the json configure file.
            yaml_file: the path to the yaml configure file.
            fuse_args: if fuse the json/yaml configs with argparse.
        """
        assert isinstance(json_file, str)
        assert isinstance(yaml_file, str)

        if json_file != "" and yaml_file != "":
            raise Warning(
                "json_file and yaml_file can not co-exist for now. please only use one configure file type."
            )
            return

        self.args = None
        self.arg_config = {}
        self.json_config = {}
        self.yaml_config = {}

        parser = argparse.ArgumentParser()

        self.default_g = ArgumentGroup(parser, "default", "default options.")
        self.yaml_g = ArgumentGroup(parser, "yaml", "options from yaml.")
        self.json_g = ArgumentGroup(parser, "json", "options from json.")
        self.com_g = ArgumentGroup(parser, "custom", "customized options.")

        # self.default_g.add_arg("do_train", bool, False,
        #                        "Whether to perform training.")
        # self.default_g.add_arg("do_predict", bool, False,
        #                        "Whether to perform predicting.")
        # self.default_g.add_arg("do_eval", bool, False,
        #                        "Whether to perform evaluating.")
        # self.default_g.add_arg("do_save_inference_model", bool, False,
        #                        "Whether to perform model saving for inference.")

        # # NOTE: args for profiler
        # self.default_g.add_arg(
        #     "is_profiler", int, 0,
        #     "the switch of profiler tools. (used for benchmark)")
        # self.default_g.add_arg(
        #     "profiler_path", str, './',
        #     "the profiler output file path. (used for benchmark)")
        # self.default_g.add_arg("max_iter", int, 0,
        #                        "the max train batch num.(used for benchmark)")

        self.parser = parser

        if json_file != "":
            self.load_json(json_file, fuse_args=fuse_args)

        if yaml_file:
            self.load_yaml(yaml_file, fuse_args=fuse_args)

    def load_json(self, file_path, fuse_args=True):

        if not os.path.exists(file_path):
            raise Warning("the json file %s does not exist." % file_path)
            return

        with open(file_path, "r") as fin:
            self.json_config = json.loads(fin.read())
            fin.close()

        if fuse_args:
            for name in self.json_config:
                if isinstance(self.json_config[name], list):
                    self.json_g.add_arg(
                        name,
                        type(self.json_config[name][0]),
                        self.json_config[name],
                        "This is from %s" % file_path,
                        nargs=len(self.json_config[name]))
                    continue
                if not isinstance(self.json_config[name], int) \
                    and not isinstance(self.json_config[name], float) \
                    and not isinstance(self.json_config[name], str) \
                    and not isinstance(self.json_config[name], bool):

                    continue

                self.json_g.add_arg(name,
                                    type(self.json_config[name]),
                                    self.json_config[name],
                                    "This is from %s" % file_path)

    def load_yaml(self, file_path, fuse_args=True):

        if not os.path.exists(file_path):
            raise Warning("the yaml file %s does not exist." % file_path)
            return

        with open(file_path, "r") as fin:
            self.yaml_config = yaml.load(fin, Loader=yaml.SafeLoader)
            fin.close()

        if fuse_args:
            for name in self.yaml_config:
                if isinstance(self.yaml_config[name], list):
                    self.yaml_g.add_arg(
                        name,
                        type(self.yaml_config[name][0]),
                        self.yaml_config[name],
                        "This is from %s" % file_path,
                        nargs=len(self.yaml_config[name]))
                    continue

                if not isinstance(self.yaml_config[name], int) \
                    and not isinstance(self.yaml_config[name], float) \
                    and not isinstance(self.yaml_config[name], str) \
                    and not isinstance(self.yaml_config[name], bool):

                    continue

                self.yaml_g.add_arg(name,
                                    type(self.yaml_config[name]),
                                    self.yaml_config[name],
                                    "This is from %s" % file_path)

    def build(self):
        self.args = self.parser.parse_args()
        self.arg_config = vars(self.args)

    def __add__(self, new_arg):
        assert isinstance(new_arg, list) or isinstance(new_arg, tuple)
        assert len(new_arg) >= 3
        assert self.args is None

        name = new_arg[0]
        dtype = new_arg[1]
        dvalue = new_arg[2]
        desc = new_arg[3] if len(
            new_arg) == 4 else "Description is not provided."

        self.com_g.add_arg(name, dtype, dvalue, desc)

        return self

    def __getattr__(self, name):
        if name in self.arg_config:
            return self.arg_config[name]

        if name in self.json_config:
            return self.json_config[name]

        if name in self.yaml_config:
            return self.yaml_config[name]

        raise Warning("The argument %s is not defined." % name)

    def Print(self):

        print("-" * 70)
        for name in self.arg_config:
            print("%s:\t\t\t\t%s" % (str(name), str(self.arg_config[name])))

        for name in self.json_config:
            if name not in self.arg_config:
                print("%s:\t\t\t\t%s" %
                      (str(name), str(self.json_config[name])))

        for name in self.yaml_config:
            if name not in self.arg_config:
                print("%s:\t\t\t\t%s" %
                      (str(name), str(self.yaml_config[name])))

        print("-" * 70)

if __name__ == "__main__":

    pd_config = PDConfig(yaml_file="transformer_in_all/myTENER/TENER.yaml")
    pd_config.build()

    print(pd_config.cnn_dim)
    print(pd_config.n_kernals)
    