import json
import pathlib
import numpy as np

base_dir = pathlib.Path("exp_evaluate_sm/results/evaluate_dis_sm/syr2k/DatasetIdentity:syr2k")
for j in base_dir.iterdir():
    if j.suffix != '.json':
        continue
    with open(j,'r') as f:
        print(j)
        jdata = json.load(f)
    for technique, details in jdata.items():
        print(technique)
        tech_pred = np.asarray(details['y_pred']).ravel()
        tech_true = np.asarray(details['y_true']).ravel()
        print("\tTRUE:",tech_true)
        print("\tPRED:",tech_pred)

        truesort = np.argsort(tech_true)
        predsort = np.argsort(tech_pred)
        print("\tTSORT:",truesort)
        print("\tPSORT:",predsort)

        print("\tAVG_DISPLACEMENT:",np.asarray([predsort.tolist().index(_) for _ in truesort[:len(truesort)//2]]).mean())

