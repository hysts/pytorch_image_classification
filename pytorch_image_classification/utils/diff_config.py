from typing import Optional

import yacs.config

from pytorch_image_classification import get_default_config
from pytorch_image_classification.config.config_node import ConfigNode


def find_config_diff(
        config: yacs.config.CfgNode) -> Optional[yacs.config.CfgNode]:
    def _find_diff(node: yacs.config.CfgNode,
                   default_node: yacs.config.CfgNode):
        root_node = ConfigNode()
        for key in node:
            val = node[key]
            if isinstance(val, yacs.config.CfgNode):
                new_node = _find_diff(node[key], default_node[key])
                if new_node is not None:
                    root_node[key] = new_node
            else:
                if node[key] != default_node[key]:
                    root_node[key] = node[key]
        return root_node if len(root_node) > 0 else None

    default_config = get_default_config()
    new_config = _find_diff(config, default_config)
    return new_config
