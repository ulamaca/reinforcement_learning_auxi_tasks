#  Atari State Representation Learning

## Usage

First see the [baselines documentation](https://github.com/openai/baselines) to learn how to download pretrained models etc.

__IMPORTANT__: In order to be able to execute modules from the base folder of the repository you have to add the baseline folder to your python search path

```
export PYTHONPATH=./baselines:$PYTHONPATH; python -m <name-of-module>
```

## Quick start guide

1. Go to the base directory of the git repo

2. Add baselines to the python search path
```
export PYTHONPATH=./baselines:$PYTHONPATH
```

3. List available models
```
python -m baselines.deepq.experiments.atari.download_model
```

4. Download pre-trained model
```
mkdir pretrained_models
python -m baselines.deepq.experiments.atari.download_model --blob <name-of-model> --model-dir pretrained_models
```

5. Run model and extract game states as well as hidden activities
```
python -m custom.enjoy --model-dir pretrained_models/<name-of-model> --env <name-of-environment> --saveStateDir output [--maxNumEpisodes <N>] [--dueling]
```
Available environments for atari games can be found [here](https://gym.openai.com/envs#atari).
States and activities are stored in the `output/<name-of-model>` folder. There is one folder for each episode (an episode is defined on a per game basis). Each state (current reward, action, input etc.) and layer activity is stored in a separate file that contains all frames of the current episode. 

6. To visualize hidden activities use
```
python -m custom.visualize --buffersDir output/<name-of-model> --episode <N> --frames <K>
```
K is a comma separated list of ranges (e.g. `1,5-6,10`) of frames that will be plotted.


## Custom Module

Scripts in the custom directory are modified version of scripts provided by the baselines project or new scripts

* model.py / build_graph.py: Modified such that hidden activities can be extracted
* enjoy.py: Modified so that hidden activities can be stored to files for later use
* utils: Just some helper functions
* visualize: Visualize game states and hidden activities