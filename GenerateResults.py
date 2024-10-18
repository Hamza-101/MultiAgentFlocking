import os
import json
import numpy as np
from Parameters import *
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from main import FlockingEnv as env

def delete_files(): 
    Paths = ["Results\Flocking\Testing\Dynamics\Accelerations", "Results\Flocking\Testing\Dynamics\Velocities", 
            "Results\Flocking\Testing\Rewards\Other"]

    Logs = ["AlignmentReward_log.json", "CohesionReward_log.json",
            "SeparationReward_log.json", "CollisionReward_log.json",
            "Reward_Total_log.json"]

    for Path in Paths:
        for episode in range(0, 10):
            file_path = os.path.join(Files['Flocking'], Path, f"Episode{episode}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")

    for log_file in Logs:
        for episode in range(0, 10):
            file_path = os.path.join(Files['Flocking'], "Testing", "Rewards", "Components", f"Episode{episode}", log_file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")       

def generateCombined():
    with open(rf"{Results['EpisodalRewards']}.json", "r") as f:
        episode_rewards_dict = json.load(f)

    keys_above_threshold = []
    keys_below_threshold = []

    for episode, rewards in episode_rewards_dict.items():
        total_sum = sum(rewards)
        if total_sum > 1000000:
            keys_above_threshold.append(episode)
        else:
            keys_below_threshold.append(episode)

    plt.figure(figsize=(10, 6))
    plt.clf()

    #Fix this
    for episode in keys_above_threshold:
        rewards = episode_rewards_dict[episode]
        plt.plot(range(1, len(rewards) + 1), rewards, label=f"Episode {episode}", alpha=0.7)

    for episode in keys_below_threshold:
        rewards = episode_rewards_dict[episode]
        plt.plot(range(1, len(rewards) + 1), rewards, label=f"Episode {episode}", alpha=0.7)

    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title(f"Rewards for Episodes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Output.png", dpi=300)

def generateVelocity():
    # Loop through episodes
    for episode in range(0, SimulationVariables["Episodes"]):
        velocities_dict = {}

        with open(os.path.join(positions_directory, f"Episode{episode}_velocities.json"), 'r') as f:
            episode_velocities = json.load(f)

        for agent_id in range(len(env.agents)):
            velocities_dict.setdefault(agent_id, []).extend(episode_velocities.get(str(agent_id), []))

        plt.figure(figsize=(10, 5))
        plt.clf()  
        for agent_id in range(len(env.agents)):
            agent_velocities = np.array(velocities_dict[agent_id])
            # agent_velocities = savgol_filter(agent_velocities, window_length=5, polyorder=3, axis=0)
            velocities_magnitude = np.sqrt(agent_velocities[:, 0]**2 + agent_velocities[:, 1]**2)  
      
            plt.plot(velocities_magnitude, label=f"Agent {agent_id+1}")

        plt.title(f"Velocity - Episode {episode}")
        plt.xlabel("Time Step")
        plt.ylabel("Velocity Magnitude")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Episode{episode}_SmoothedVelocity.png")

def generateAcceleration():
    for episode in range(0, SimulationVariables["Episodes"]):
        with open(os.path.join(positions_directory, f"Episode{episode}_accelerations.json"), 'r') as f:
            episode_accelerations = json.load(f)

        plt.figure(figsize=(10, 5))
        plt.clf()

        for agent_id in range(len(env.agents)):
            agent_accelerations = np.array(episode_accelerations[str(agent_id)])
            smoothed_accelerations=np.sqrt(agent_accelerations[:, 0]**2 + agent_accelerations[:, 1]**2)
            # print(smoothed_accelerations)
            smoothed_accelerations = savgol_filter(smoothed_accelerations, window_length=3, polyorder=2, axis=0)
            accelerations_magnitude = np.clip(smoothed_accelerations, 0, 5) 

            plt.plot(accelerations_magnitude, label=f"Agent {agent_id+1}")

        plt.title(f"Acceleration - Episode {episode}")
        plt.xlabel("Time Step")
        plt.ylabel("Acceleration Magnitude")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Episode{episode}_SmoothedAcceleration.png")

# Analytics
print("Generating Results")
generateCombined()
print("Generating Velocity")
generateVelocity()
print("Generating Acceleration")
generateAcceleration()