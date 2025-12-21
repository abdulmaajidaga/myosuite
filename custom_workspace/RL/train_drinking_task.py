import gymnasium as gym
import myosuite
import numpy as np
import pandas as pd
import os
from stable_baselines3 import PPO
from gymnasium import Wrapper
from myosuite.envs.myo.myochallenge.relocate_v0 import RelocateEnvV0

# --- 1. Function to read the .mot file ---
def read_mot_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at {filepath}")
    
    with open(filepath, "r") as file:
        for i, line in enumerate(file):
            if "endheader" in line:
                skiprows = i + 1
                break
    
    df = pd.read_csv(filepath, sep=r'\\s+', skiprows=skiprows, engine='python')
    df = df.drop(columns=df.columns[0])
    return df

# --- 2. The Imitation Learning Wrapper ---
class ImitationWrapper(Wrapper):
    def __init__(self, env, ref_motion, imitation_weight=10.0):
        super().__init__(env)
        self.ref_motion = ref_motion
        self.imitation_weight = imitation_weight
        self.frame_idx = 0
        
        self.joint_indices = []
        self.ref_motion_joint_names = ref_motion.columns.tolist()
        sim_joint_names = [self.env.sim.model.joint(i).name for i in range(self.env.sim.model.njnt)]
        
        for joint_name in self.ref_motion_joint_names:
            if joint_name in sim_joint_names:
                qpos_addr = self.env.sim.model.joint(joint_name).qposadr[0]
                self.joint_indices.append({'ref_name': joint_name, 'qpos_addr': qpos_addr})

    def reset(self, **kwargs):
        self.frame_idx = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        imitation_err = 0.0
        if self.frame_idx < len(self.ref_motion):
            ref_qpos_step = self.ref_motion.iloc[self.frame_idx]
            
            for joint_map in self.joint_indices:
                sim_qpos_val = self.env.sim.data.qpos[joint_map['qpos_addr']]
                ref_qpos_val = ref_qpos_step[joint_map['ref_name']]
                imitation_err += (sim_qpos_val - ref_qpos_val) ** 2
            
            imitation_reward = -np.sqrt(imitation_err) * self.imitation_weight
            reward += imitation_reward

        self.frame_idx += 1
        
        # FIX: The episode is "truncated" (ends due to a time limit) if the motion is over.
        # We combine this with any truncation signal from the base environment.
        is_motion_over = self.frame_idx >= len(self.ref_motion)
        truncated = truncated or is_motion_over
        
        # The final return MUST be a 5-tuple for gymnasium compliance with SB3
        return obs, reward, terminated, truncated, info

# --- 3. Main Training Script ---
if __name__ == "__main__":
    mot_path = "/home/abdul/Desktop/myosuite/custom_workspace/IK/output/S5_12_1.mot"
    motion_df = read_mot_file(mot_path)

    model_path = os.path.join(os.path.dirname(myosuite.__file__), 'envs/myo/assets/arm/myoarm_relocate.xml')
    
    base_env = RelocateEnvV0(
        model_path=model_path,
        obs_keys=['hand_qpos', 'obj_pos', 'goal_pos', 'pos_err'],
        weighted_reward_keys={
            "pos_dist": 1.0,
            "act_reg": 0.1,
        },
        target_xyz_range={'high': (0.2, 0.2, 0.2), 'low': (-0.2, -0.2, -0.2)},
        target_rxryrz_range={'high': (0.0, 0.0, 0.0), 'low': (0.0, 0.0, 0.0)}
    )

    env = ImitationWrapper(base_env, motion_df, imitation_weight=10.0)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200000)
    model.save("ppo_drinking_model_wrapped")

    print("Training complete. Model saved to ppo_drinking_model_wrapped.zip")
    env.close()
