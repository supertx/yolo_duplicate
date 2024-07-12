"""
@author supermantx
@date 2024/7/12 17:11
加载yaml配置文件
"""

import yaml


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
