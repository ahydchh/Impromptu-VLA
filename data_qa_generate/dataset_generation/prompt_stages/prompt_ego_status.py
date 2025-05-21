import os

import numpy as np
from dataset_generation.prompt_stages.prompt_planning import PromptNuscenesPlanning

class PromptNuscenesEgoStatus(PromptNuscenesPlanning):
    def __init__(self, nuscenes):
        super().__init__(nuscenes)
    
    def format_data(self, data, precision=3) -> str:
        # data : a list of floats
        new_strings = []
        for d in data:
            new_strings.append(f"{d:.{precision}f}")
        return '[' + ', '.join(new_strings) + ']'

    
    def __call__(self, container_out, container_in):
        this_id = container_in['img_metas'].data['token']
        
        ego_poses, this_sample_idx = self.extract_scene_trajs(container_out, container_in)
        ego_dthetas = self.compute_traj_to_thetas(ego_poses)
        ego_velocity = self.compute_traj_to_velocity(ego_poses)
        ego_dthetas, ego_velocity = np.round(ego_dthetas, 0), np.round(ego_velocity, 2)
        previous_ego_dthetas = ego_dthetas[max(0, this_sample_idx-5): this_sample_idx+1]
        previous_ego_velocity = ego_velocity[max(0, this_sample_idx-5): this_sample_idx+1]
        if len(previous_ego_dthetas) < 6:
            previous_ego_dthetas = np.pad(previous_ego_dthetas, ((6-len(previous_ego_dthetas), 0),), mode='edge')
            previous_ego_velocity = np.pad(previous_ego_velocity, ((6-len(previous_ego_velocity), 0),), mode='edge')

        # ego_status = container_in['ego_status'].data
        # ego_status.extend(pose_data["accel"]) # acceleration in ego vehicle frame, m/s/s, 3 elements
        # ego_status.extend(pose_data["rotation_rate"]) # angular velocity in ego vehicle frame, rad/s, 3 elements
        # ego_status.extend(pose_data["vel"]) # velocity in ego vehicle frame, m/s, 3 elements
        # ego_status.append(steer_data["value"]) # steering angle, positive: left turn, negative: right turn, 1 element
        
        # ego_status_string += "You are given with ego vehicle status:\n"
        # ego_status_string += f"Acceleration: {self.format_data(ego_status[:3])} m/s^2\n"
        # ego_status_string += f"Angular velocity: {self.format_data(ego_status[3:6])} rad/s\n"
        # ego_status_string += f"Velocity: {self.format_data(ego_status[6:9])} m/s\n"
        # ego_status_string += f"Steering angle: {ego_status[9]} (positive: left turn, negative: right turn)\n"
        
        ego_status_string = ""
        ego_status_string += "Provided are the previous ego vehicle statuses recorded over the last 3 seconds (at 0.5-second intervals). This includes the distance traveled during each 0.5-second interval (in meters) and the change in direction (in degrees). A direction value of 0 indicates moving straight, while positive values represent left turns (with the angle increasing during the initial phase of the turn and decreasing back to 0 as the turn concludes). The data is presented for the last 6 timesteps in the format [distance, direction]:\n"
        ego_status_string += f"(t-3s) [{previous_ego_velocity[0]}, {previous_ego_dthetas[0]}], (t-2.5s) [{previous_ego_velocity[1]}, {previous_ego_dthetas[1]}], (t-2s) [{previous_ego_velocity[2]}, {previous_ego_dthetas[2]}], (t-1.5s) [{previous_ego_velocity[3]}, {previous_ego_dthetas[3]}], (t-1s) [{previous_ego_velocity[4]}, {previous_ego_dthetas[4]}], (t-0.5s) [{previous_ego_velocity[5]}, {previous_ego_dthetas[5]}]\n"
        
        
        container_out['messages'][0]['content'] += ego_status_string

        return container_out
