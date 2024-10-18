import os
import json
from tqdm import tqdm
from stable_baselines3 import PPO
from main import FlockingEnv
from Parameters import *

# Model Testing
env = FlockingEnv()
model = PPO.load(rf'{Files["Flocking"]}\Models\FlockingCombinedNew')

positions_directory = rf"{Files['Flocking']}/Testing/Episodes/"
os.makedirs(positions_directory, exist_ok=True)

env.counter = 389
episode_rewards_dict = {}

for episode in tqdm(range(SimulationVariables['Episodes'])):
    env.episode = episode
    print("Episode:", episode)
    env.CTDE = True
    obs, _ = env.reset()  # Reset environment and get initial observation
    done = False
    timestep = 0
    reward_episode = []

    # Initialize dictionaries to store data
    positions_dict = {i: [] for i in range(len(env.agents))}
    velocities_dict = {i: [] for i in range(len(env.agents))}
    accelerations_dict = {i: [] for i in range(len(env.agents))}
    trajectory_dict = {i: [] for i in range(len(env.agents))}

    while timestep < min(SimulationVariables["EvalTimeSteps"], 3000):
        actions, _ = model.predict(obs)  # Predict actions
        obs, reward, done, _, _ = env.step(actions)  # Step in the environment
        reward_episode.append(reward)
        
        for i, agent in enumerate(env.agents):
            positions_dict[i].append(agent.position.tolist())
            velocities_dict[i].append(agent.velocity.tolist())
            accelerations_dict[i].append(agent.acceleration.tolist())
            trajectory_dict[i].append(agent.position.tolist())

        timestep += 1
        episode_rewards_dict[str(episode)] = reward_episode

    # Save episode data
    with open(os.path.join(positions_directory, f"Episode{episode}_positions.json"), 'w') as f:
        json.dump(positions_dict, f, indent=4)
    with open(os.path.join(positions_directory, f"Episode{episode}_velocities.json"), 'w') as f:
        json.dump(velocities_dict, f, indent=4)
    with open(os.path.join(positions_directory, f"Episode{episode}_accelerations.json"), 'w') as f:
        json.dump(accelerations_dict, f, indent=4)
    with open(os.path.join(positions_directory, f"Episode{episode}_trajectory.json"), 'w') as f:
        json.dump(trajectory_dict, f, indent=4)

    env.counter += 1
    print(f"Total Reward for Episode {episode}: {sum(reward_episode)}")

# Save episodic rewards
with open(rf"{Results['EpisodalRewards']}.json", 'w') as f:
    json.dump(episode_rewards_dict, f, indent=4)

env.close()
print("Testing completed")
