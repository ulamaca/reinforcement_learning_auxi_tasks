# Usage:

## prerequisites:
1. install libraries: azure, opencv
2. set ./baselines into PYTHONPATH 
3. the program is tested under Python3 environment

## Running the program
1. Change the current directory i.e. .../custom 
2. python -m train_aux_joint.py <parameters (for setting them, please refer to train_aux_joint.py)>
3. An example: python -m train_aux_joint.py --env Alien --regularization-cnst 0.1 --spred-cnst 1 --rpred-cnst 0
