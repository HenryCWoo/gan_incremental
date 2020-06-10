import os
import yaml
from pathlib import Path
import pandas as pd


all_results = []

for yaml_path in Path('./experiments').glob('**/*.yml'):
    with open(yaml_path, 'r') as yamlfile:
        cur_yaml = yaml.safe_load(yamlfile)

    data = {}

    for k, v in cur_yaml.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                data[k2] = v2
        else:
            data[k] = v
    all_results.append(data)

all_results = pd.DataFrame(all_results)
all_results.to_csv('./aggregate_experiments.csv')
