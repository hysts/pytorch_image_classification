import yacs.config


class ConfigNode(yacs.config.CfgNode):
    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        super().__init__(init_dict, key_list, new_allowed)

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        r = ''
        s = []
        for k, v in self.items():
            separator = '\n' if isinstance(v, ConfigNode) else ' '
            if isinstance(v, str) and not v:
                v = '\'\''
            attr_str = f'{str(k)}:{separator}{str(v)}'
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += '\n'.join(s)
        return r

    def as_dict(self):
        def convert_to_dict(node):
            if not isinstance(node, ConfigNode):
                return node
            else:
                dic = dict()
                for k, v in node.items():
                    dic[k] = convert_to_dict(v)
                return dic

        return convert_to_dict(self)
