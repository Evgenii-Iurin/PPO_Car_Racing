# PPO with Clip function / CarRacing-v2 Environment
This repository features an implementation of the Proximal Policy Optimization (PPO) algorithm, crafted from scratch in Python, tailored for the CarRacing-v2 environment. I developed this implementation primarily for educational purposes. It's important to note that after 1,000,000 steps, the algorithm still hasn't trained sufficiently to successfully complete the race. Currently, I am investigating this issue. Below, you can find the results of the training.

![rollout_exp_7_1](https://github.com/Evgenii-Iurin/PPO_Car_Racing/blob/10ab2be8ddad4aed0c467496f4e6bf95a73e2948/eval_replays/exp_7/rollout_exp_7_1.gif)
![rollout_exp_7_2](https://github.com/Evgenii-Iurin/PPO_Car_Racing/blob/10ab2be8ddad4aed0c467496f4e6bf95a73e2948/eval_replays/exp_7/rollout_exp_7_2.gif)
![rollout_exp_7_3](https://github.com/Evgenii-Iurin/PPO_Car_Racing/blob/10ab2be8ddad4aed0c467496f4e6bf95a73e2948/eval_replays/exp_7/rollout_exp_7_3.gif)

### How to run
1. Create a virtual environment using Python 3.10 or higher. <br/>
While I haven't tested it with other versions, I recommend using at least Python 3.8.
```bash
pyenv virtualenv 3.10.13 ppo-car-racing
pyenv activate ppo-car-racing
```
2. Install [PyTorch](https://pytorch.org/) for your device
3. Install gymnasium manually
```bash
pip install swig
pip install "gymnasium[box2d]"
pip install gymnasium
```
5. Install all requirements.
```bash
pip install -r requirements.txt
```
4. Export Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:path/to/ppo_folder"
```

## Run the training
I didn't use Parser or Hydra for parameterization, so all parameters are located in the `scripts/train.py` file. There are many parameters, but you might not need to adjust all of them. Here are some parameters you can change: `num_steps`, `max_steps`, `num_epochs`, `clip_coef`, `lr`, and `num_minibatches`.
```bash
python scripts/train.py
```
## Results
The script creates `logs` and `models` folders:
1. The `logs` folder contains the `loss.csv` and `returns.csv` files. The actor and critic losses are saved in `loss.csv`. The average return from each evaluation is saved in `returns.csv`.
2. The `models` folder contains the best weights of the model.
### Visualize the results
Change the paths in `visualize.py` and run:
```bash
python utils/visualize.py
```

# Discuss  the results
The results of the experiment are presented below. As you can see, one of the main issues is that the loss is unstable and fluctuates constantly. After 1,000,000 steps, the agent still hasn’t learned to control the car. It's also important to note the entropy, which decreases to zero towards the end of training. This indicates that the agent is very confident in choosing a specific action. It can be inferred that in the visualization (mentioned at the beginning of this description), the agent brakes at turns and seems unsure of what to do next because it is confident that choosing to brake is a good decision. One reason might be that the agent hasn't explored the environment enough and is stuck. As a result, the reward doesn't rise above zero throughout the journey. One possible solution could be to introduce a terminal state if the agent hasn't received a reward for crossing tiles in the last N actions. It makes sense that if the car hasn't crossed the tiles, it could have gone off-track, be moving in the wrong direction, etc., so it would be logical to "terminate" the agent.

<div style="display: flex; justify-content: space-around;">
    <img src="https://github.com/Evgenii-Iurin/PPO_Car_Racing/blob/6fca9f7887c2d58a156b93fbec7946f3d4da15e1/result_exp/exp_7/actor_loss.jpg" style="width:49%;">
    <img src="https://github.com/Evgenii-Iurin/PPO_Car_Racing/blob/6fca9f7887c2d58a156b93fbec7946f3d4da15e1/result_exp/exp_7/critic_loss.jpg" style="width:49%;">
</div>

<div style="display: flex; justify-content: space-around;">
    <img src="https://github.com/Evgenii-Iurin/PPO_Car_Racing/blob/6fca9f7887c2d58a156b93fbec7946f3d4da15e1/result_exp/exp_7/entropy.jpg" style="width:49%;">
    <img src="https://github.com/Evgenii-Iurin/PPO_Car_Racing/blob/6fca9f7887c2d58a156b93fbec7946f3d4da15e1/result_exp/exp_7/return_value.jpg" style="width:49%;">
</div>

