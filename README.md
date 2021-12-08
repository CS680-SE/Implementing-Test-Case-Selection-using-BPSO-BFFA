  
# BPSO Vs. BFFA 
## In test case selection
We introduce a comparison study with aims to reduce test case selection using the Binary Particle Swarm Optimization (BPSO) and Binary FireFly algorithms (BFFA).<br />
<br />
## Datasets <br />
It is a dataset that contains historical information about test case execution and their results, 
we use are open source and named<br /> **ABB Paint Control** 
We can introduce it as follows: <br /><br />

| Test Cases | CI-cycles | Verdicts | Failed |
| :---         |     :---:      |          ---: |     :---:      |
| 89   | 352     | 25,594    |     19.36%      |


<br /> <br /> you can **download** it from:<br /> 
**https://github.com/sqaunderhood/atcs-data#adaptive-test-case-selection---data-sets**

# Libraries 
 Need to import some libraries such as :<br />
 ```
import numpy as np
import csv
import random
from itertools import combinations as cb
import math
from copy import deepcopy as dc
from tqdm import tqdm
```





