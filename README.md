#  Atari State Representation Learning

## Usage

First see the [baselines documentation](https://github.com/openai/baselines) to learn how to download pretrained models etc.

__IMPORTANT__: In order to be able to execute modules from the base folder of the repository you have to add the baseline folder to your python search path

```
export PYTHONPATH=./baselines:$PYTHONPATH; python -m <name-of-module>
```

## Custom Module

Scripts in the custom directory are modified version of scripts provided by the baselines project or new scripts

* model.py / build_graph.py: Modified such that hidden activities can be extracted
* enjoy.py: Modified so that hidden activities can be stored to files for later use
* utils: Just some helper functions
* visualize: Visualize game states and hidden activities