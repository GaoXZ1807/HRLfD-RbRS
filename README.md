# HRLfD-RbRS
This is a PyTorch implementation for our paper "Hierarchical Reinforcement Learning from Demonstration via Reachability
-Based Reward Shaping" 

## Dependencies
- Python 3.7
- PyTorch 1.7
- OpenAI Gym
- MuJoCo

Now the MuJoCo license is no longer needed, but if you need (see [here](https://www.roboti.us/license.html)).

## Usage

**Update:** implementation for discrete control tasks is in `discrete/` folder; please refer to the usage therein.

### Training
- Ant Gather
```
python main.py --env_name AntGather
```
- Ant Maze
```
python main.py --env_name AntMaze
```
### Evaluation
- Ant Gather
```
python eval.py --env_name AntGather --model_dir [MODEL_DIR]
```
- Ant Maze
```
python eval.py --env_name AntMaze --model_dir [MODEL_DIR]
```

Default `model_dir` is `pretrained_models/`.


