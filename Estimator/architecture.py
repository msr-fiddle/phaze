# internal imports
from .utils import get_core_area, get_core_energy

from collections import namedtuple
from math import log2

# python imports
import os
import sys
import json

# Global Variables for Architecture
bandwidth = 4 * 1024 * 1024 * 1024  # 32 Gbs or 4 GBs, PCIE 3.0
num_accelerators = 1024  # accelerators
bytes_per_element = 2  # 16 bits
frequency = 1.05 * (10**6)  # 1.05 GHz

# types of cores in the architectures
cores = ["TC", "VC"]

# config per core, for VC depth is always 1
per_core_config = namedtuple(
    "per_core_config", ["num", "width", "depth", "GLB_Buffer"])

# accelerator config tuple
acc_config = namedtuple(
    "acc_config", ["num_tc", "num_vc", "width", "depth", "width_vc", "GLB_Buffer", "area"])

# The maximum accelerator config might not be able to fit the max of each of the above aspects of the core
max_acc_config_per_dim = acc_config(
    4096, 4096, 256, 256, 256, 128*1024*1024, -1)

# maximum accelerator config for area constraint
max_acc_config = acc_config(8, 2, 128, 128, 128, 128*1024*1024,  -1)
area_constraint = -1

# potential accelerator configs to explore
all_possible_acc_configs = []
acc_configs_to_explore = []
tc_configs = []
vc_configs = []

# only largest area
only_explore_largest_area = False

# only specific configs
only_explore_specific_configs = False


def create_core_config(cc, core_type="TC"):
    if core_type not in cores:
        raise TypeError("Core type not in cores supported.", core_type)

    if core_type == "TC":
        return per_core_config(cc.num_tc, cc.width, cc.depth, cc.GLB_Buffer)
    elif core_type == "VC":
        return per_core_config(cc.num_vc, cc.width_vc, 1, cc.GLB_Buffer)


def generate_area_of_acc(acc_config):
    tc_coreconfig = create_core_config(acc_config, "TC")
    area, glb_area = get_core_area(tc_coreconfig, "TC")

    vc_coreconfig = create_core_config(acc_config, "VC")
    core_area_vc, glb_area_vc = get_core_area(vc_coreconfig, "VC")

    glb_area = max(glb_area, glb_area_vc)
    area += core_area_vc + glb_area
    return area


def get_area_constraint():
    global area_constraint
    global max_acc_config

    if area_constraint == -1:
        area_constraint = generate_area_of_acc(max_acc_config)

    max_acc_config = max_acc_config._replace(area=area_constraint)
    return area_constraint


def generate_all_cores_to_explore():

    global all_possible_acc_configs

    def check_if_acc_to_explore(config):
        area_factor = 0.0 if config.num_tc == 1 or config.num_vc == 1 else 0.3
        config = config._replace(area=generate_area_of_acc(config))

        max_area = get_area_constraint()

        if (area_factor * max_area) <= config.area <= max_area:
            all_possible_acc_configs.append(config)
            return True
        return False

    max_log_depth = int(log2(max_acc_config_per_dim.depth))
    max_log_width = int(log2(max_acc_config_per_dim.width))
    max_log_cores = int(log2(max_acc_config_per_dim.num_tc))
    max_log_GLB_bufferMB = max_acc_config_per_dim.GLB_Buffer / (1024*1024)
    max_log_GLB_buffer = int(log2(max_log_GLB_bufferMB))

    # restricting the total number of possibilites by making the TC and VC core width the same

    for log_x in range(max_log_cores, -1, -1):
        for log_y in range(max_log_cores, -1, -1):
            for log_w in range(max_log_width, 0, -1):
                for log_d in range(max_log_depth, 0, -1):
                    for log_glb in range(max_log_GLB_buffer, 0, -1):
                        config = acc_config(2**log_x, 2**log_y,
                                            2**log_w, 2**log_d, 2**log_w, (2**log_glb)*1024*1024, -1)
                        check_if_acc_to_explore(config)


def generate_unique_core_configs():
    global tc_configs
    global vc_configs

    if (tc_configs and vc_configs):
        return

    for config in acc_configs_to_explore:
        tc_config = create_core_config(config, "TC")
        vc_config = create_core_config(config, "VC")

        if tc_config not in tc_configs:
            tc_configs.append(tc_config)
        if vc_config not in vc_configs:
            vc_configs.append(vc_config)


def get_configs_to_explore():
    if acc_configs_to_explore and vc_configs and tc_configs:
        return acc_configs_to_explore
    else:
        raise ValueError("No accelerator configs or core configs set")


def set_configs_to_explore():
    curr_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    config_dir = os.path.join(curr_dir, "arch_configs")

    config_filename = "core_largest.json" if only_explore_specific_configs else "cores.json"
    config_file = os.path.join(config_dir, config_filename)

    global acc_configs_to_explore

    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            configs = json.load(f)
            acc_configs_to_explore = [acc_config(**cc) for cc in configs]
        f.close()
    else:
        if only_explore_specific_configs:
            raise ValueError("No specific config file found.")

        generate_all_cores_to_explore()
        acc_configs_to_explore = all_possible_acc_configs
        with open(config_file, "w") as f:
            f.write("[")
            for idx, config in enumerate(acc_configs_to_explore):
                json.dump(config._asdict(), f,
                          ensure_ascii=False, indent=None)
                if (idx != len(acc_configs_to_explore) - 1):
                    f.write(",\n")
            f.write("]")
        f.close()

    if only_explore_largest_area:
        max_area = max([x.area for x in acc_configs_to_explore])
        max_area_acc_configs_to_explore = []
        for cc in acc_configs_to_explore:
            if cc.area == max_area:
                max_area_acc_configs_to_explore.append(cc)
            elif cc.num_tc == 1 or cc.num_vc == 1:
                max_area_acc_configs_to_explore.append(cc)

        acc_configs_to_explore = max_area_acc_configs_to_explore

    generate_unique_core_configs()
    return acc_configs_to_explore
