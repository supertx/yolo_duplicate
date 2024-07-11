"""
@author supermantx
@date 2024/7/3 16:23
"""
import math


def make_divisible(x, divisor) -> int:
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor


def get_efficientRep_build_lst(depth_scale=0.33, width_scale=0.25):
    num_repeats = [6, 12, 18, 6]
    out_channels = [128, 256, 512, 1024]
    num_repeat = [(max(round(i * depth_scale), 1) if i > 1 else i) for i in (num_repeats)]
    out_channels = [make_divisible(i * width_scale, 8) for i in (out_channels)]
    in_channels = [16] + out_channels[:-1]
    return [[x, y, z] for x, y, z in zip(in_channels, out_channels, num_repeat)]


def get_decouple_head_build_lst(width_scale=0.25):
    out_channels = [512, 256, 128]
    out_channels = [make_divisible(i * width_scale, 8) for i in (out_channels)]
    return out_channels


def get_reoBiPan_build_lst(depth_scale=0.33, width_scale=0.25):
    num_repeats = [12, 12, 12, 12]
    out_channels = [1024, 256, 512, 128, 256, 512]
    num_repeats = [(max(round(i * depth_scale), 1) if i > 1 else i) for i in (num_repeats)]
    out_channels = [make_divisible(i * width_scale, 8) for i in (out_channels)]
    return num_repeats, out_channels
