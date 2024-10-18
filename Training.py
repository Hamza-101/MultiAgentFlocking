import numpy as np
import torch as th
from Parameters import *
from stable_baselines3 import PPO
from main import FlockingEnv, CustomMultiAgentPolicy
from Callbacks import TQDMProgressCallback, LossCallback
import os
from stable_baselines3.common.vec_env import DummyVecEnv

device = th.device("cuda" if th.cuda.is_available() else "cpu")
# print("device", device)

if os.path.exists(Results["Rewards"]):
    os.remove(Results["Rewards"])
    print(f"File {Results['Rewards']} has been deleted.")

if os.path.exists("training_rewards.json"):
    os.remove("training_rewards.json")
    print(f"File training_rewards has been deleted.")    

def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    env.seed(seed)
    env.action_space.seed(seed)


loss_callback = LossCallback()
env = DummyVecEnv([lambda: FlockingEnv()])

seed_everything(SimulationVariables["Seed"])

# # Model Training
model = PPO(CustomMultiAgentPolicy, env, tensorboard_log="./ppo_Agents_tensorboard/", verbose=1, device=device)
model.set_random_seed(SimulationVariables["ModelSeed"])
progress_callback = TQDMProgressCallback(total_timesteps=SimulationVariables["LearningTimeSteps"])
# Train the model
model.learn(total_timesteps=SimulationVariables["LearningTimeSteps"], callback=[progress_callback, loss_callback])
# model.learn(total_timesteps=SimulationVariables["LearningTimeSteps"], callback=loss_callback)
model.save(rf"{Files['Flocking']}\\Models\\FlockingCombinedNew")
