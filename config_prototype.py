import yaml
from types import SimpleNamespace

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

def yaml_to_object(filename = "config.yaml"):
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    return dict_to_namespace(data)

