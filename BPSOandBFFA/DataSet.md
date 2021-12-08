# Adaptive Test Case Selection - Data Sets

These dataset contain information on test case executions and results from the past. It can be used to assess test case priority and selection methods, as well as identify test cases that are most likely to fail on their next run. The duration of a test case's execution, its most recent execution time, and the outcomes of its most recent executions are used to define it.

ABB Robotics Norway provided of these dataset.



| Data Set          | Test Cases    | CI cycles | Verdicts | Failed |
| :---------------- |:-------------:| :--------:| :------: |-------:|
| ABB Paint Control | 89            |    352    | 25,594   | 19.36% | 


***

### File Format
The data sets are provided in CSV format (with a ';' delimiter). Because we're looking for failures, test results show True for failed runs and False for successful runs to make our product easier to use.


| Column Name       | Content                                             | 
| :---------------- |:----------------------------------------------------| 
|    Cycle          | The CI cycle to which this test execution belongs.                                      |   
|     Id            | Each test execution has a Unique numeric identifier                                     |      
|     Duration      | The test case's estimated runtime                                                       |    
|     LastRun       | as a date-time-string the latest execution of the test case (Format: YYYY-MM-DD HH:ii)  |   
|     LastResults   | Previous test results, arranged by ascending age (Failed: True, Passed: False). [ ] is used to separate lists. |        
|     Verdict       | Test verdict of this test execution (Failed: True, Passed: False)                       |  
