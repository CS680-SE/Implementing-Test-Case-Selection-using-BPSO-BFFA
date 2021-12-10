  
# BPSO Vs. BFFA 
## Comparison between BPSO and BFFA in Test Case Selection
We introduce a comparison study with aims to reduce test case selection using the Binary Particle Swarm Optimization (BPSO) and Binary FireFly algorithms (BFFA).
BPSO is an optimistic approach that provides optimum best results in minimum time compared to BFFA that covers all possible faults.
<br /><br />
# Datasets <br />
## Adaptive Test Case Selection - Data Sets
This dataset contains information on test case executions and results from the past. It can be used to assess test case priority and selection methods, as well as identify test cases that are most likely to fail on their next run.The duration of a test case's execution, its most recent execution time, and the outcomes of its most recent executions are used to define it. 
<br /><br />It's open source dataset named **ABB Paint Control** 
 , We can introduce it as follows: <br /><br />




| Data Set          | Test Cases    | CI cycles | Verdicts | Failed |
| :---------------- |:-------------:| :--------:| :------: |-------:|
| ABB Paint Control | 89            |    352    | 25,594   | 19.36% | 


***

### File Format
It is provided in CSV format (with a ';' delimiter). Since we're looking for failures, test results show True for failed runs and False for successful runs to make our product easier to use.


| Column Name       | Content                                             | 
| :---------------- |:----------------------------------------------------| 
|    Cycle          | The CI cycle to which this test execution belongs.                                      |   
|     Id            | Each test execution has a Unique numeric identifier                                     |      
|     Duration      | The test case's estimated runtime                                                       |    
|     LastRun       | as a date-time-string the latest execution of the test case (Format: YYYY-MM-DD HH:ii)  |   
|     LastResults   | Previous test results, arranged by ascending age (Failed: True, Passed: False). [ ] is used to separate lists. |        
|     Verdict       | Test verdict of this test execution (Failed: True, Passed: False)                       |  

<br /> <br /> You can **download** it from:<br /> 
**https://github.com/sqaunderhood/atcs-data#adaptive-test-case-selection---data-sets**
<br />
<br />
***
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





