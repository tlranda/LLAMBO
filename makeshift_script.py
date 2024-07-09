import re
import numpy as np
with open('27.log','r') as f:
    first_run = f.readlines()

with open('27_2.log','r') as f:
    second_run = f.readlines()

s = re.compile(r"## (-?[\d.]+) ##")
firsts = np.hstack([[float(x) for x in re.findall(s,_)] for _ in first_run if len(re.findall(s,_)) > 0])
secondss = np.hstack([[float(x) for x in re.findall(s,_)] for _ in second_run if len(re.findall(s,_)) > 0])
uniq = set(firsts).union(set(secondss))
print("Unique values:",len(uniq))
print("Total values:",len(firsts)+len(secondss))
print(dict((_,np.hstack((firsts,secondss)).tolist().count(_)) for _ in set(firsts).union(set(secondss))))
