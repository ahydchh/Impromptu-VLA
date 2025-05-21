import re
import torch
import numpy as np
from dataset_generation.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import PlanningMetric

class PromptNuscenesPlanning:
    def __init__(self, nuscenes):
        self.planning_metrics = PlanningMetric()
        self.nuscenes = nuscenes
    
    def compute_traj_to_velocity(self, traj):
        # traj: [N, 2]
        # unit: m/0.5s
        velocity = np.zeros(len(traj))
        # Compute velocity at each point (excluding the first)
        for i in range(1, len(traj)):
            x1, y1 = traj[i - 1][0], traj[i - 1][1]
            x2, y2 = traj[i][0], traj[i][1]

            # Compute velocity
            velocity[i] = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # Set the first velocity to the second
        velocity[0] = velocity[1]
        return velocity
    
    def compute_traj_to_thetas(self, traj):
        # traj: [N, 2]
        # displacement: [N]
        # Return: [N]
        thetas = np.zeros(len(traj))
        for i in range(1, len(traj)):
            x1, y1 = traj[i - 1][0], traj[i - 1][1]
            x2, y2 = traj[i][0], traj[i][1]

            # Compute theta
            if np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) < 0.2:
                thetas[i] = 0
            else:
                thetas[i] = np.arctan2(y2 - y1, x2 - x1) / np.pi * 180 - 90

        thetas[0] = 0
        
        dthetas = np.zeros(len(traj))
        for i in range(1, len(traj)):
            dthetas[i] = thetas[i] - thetas[i-1]

        for i in range(1, len(traj)):
            if dthetas[i] > 180:
                dthetas[i] -= 360
            if dthetas[i] < -180:
                dthetas[i] += 360
        
        return dthetas
    
    def compute_traj_to_curvature(self, traj):
        curvature = np.zeros(len(traj))
        
        x = traj[:, 0]
        y = traj[:, 1]
        
        dx = np.gradient(x)
        dy = np.gradient(y)
        
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # 计算曲率
        for i in range(len(traj)):
            denominator = (dx[i]**2 + dy[i]**2)**1.5
            if denominator != 0:
                curvature[i] = (dx[i] * ddy[i] - dy[i] * ddx[i]) / denominator
            else:
                curvature[i] = 0
        
        return curvature

    def compute_traj_to_curvature_linear(self, traj):
        # traj: [N, 2]
        curvature = np.zeros(len(traj))
        # Compute curvature at each point (excluding the first and last)
        for i in range(1, len(traj) - 1):
            x1, y1 = traj[i - 1][0], traj[i - 1][1]
            x2, y2 = traj[i][0], traj[i][1]
            x3, y3 = traj[i + 1][0], traj[i + 1][1]

            # Compute side lengths
            L1 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            L2 = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            L3 = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

            # Compute triangle area
            area = 0.5 * np.abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

            # Compute curvature
            if L1 > 0 and L2 > 0 and L3 > 0:  # Avoid division by zero
                curvature[i] = 4 * area / (L1 * L2 * L3)

        curvature[0] = curvature[1]  # Set the first curvature to the second
        curvature[-1] = curvature[-2]  # Set the last curvature to the second-to-last
        return curvature
    
    def compute_traj_to_curvature_spline(self, traj):
        from scipy.interpolate import CubicSpline
        t = np.arange(len(traj))
        x = traj[:, 0]
        y = traj[:, 1]
        
        # Fit cubic splines
        spline_x = CubicSpline(t, x)
        spline_y = CubicSpline(t, y)
        
        # Compute derivatives
        dx = spline_x.derivative(1)(t)
        dy = spline_y.derivative(1)(t)
        ddx = spline_x.derivative(2)(t)
        ddy = spline_y.derivative(2)(t)
        
        # Compute curvature
        curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-5)**1.5
        return curvature
    
    def compute_vel_curvature_to_traj(self, velocity, curvature):
        # velocity: [N] displacements in every 0.5s
        # curvature: [N]
        # Return: [N, 2]
        from scipy.integrate import cumulative_trapezoid
        traj = np.zeros((len(velocity), 2))
        
        t = np.arange(len(velocity))
        thetas = cumulative_trapezoid(curvature*velocity, t, initial=0)
        
        v_x = velocity * np.cos(thetas+np.pi/2)
        v_y = velocity * np.sin(thetas+np.pi/2)
        
        x = cumulative_trapezoid(v_x, t, initial=0)
        y = cumulative_trapezoid(v_y, t, initial=0)
        
        traj[:, 0] = x
        traj[:, 1] = y
        return traj

    def compute_vel_theta_to_traj(self, velocity, dthetas):
        # velocity: [N] displacements in every 0.5s
        # thetas: [N]
        # Return: [N-1, 2]
        dthetas = np.array(dthetas) / 180 * np.pi
        thetas = np.cumsum(dthetas)
        
        from scipy.integrate import cumulative_trapezoid
        traj = np.zeros((len(velocity), 2))
        
        for i in range(1, len(velocity)):
            traj[i][0] = traj[i-1][0] + velocity[i] * np.cos(thetas[i]+np.pi/2)
            traj[i][1] = traj[i-1][1] + velocity[i] * np.sin(thetas[i]+np.pi/2)
        
        return traj[1:]

    def extract_scene_trajs(self, container_out, container_in):
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion

        # 1. this_sample
        this_sample = self.nuscenes.get('sample', container_in['img_metas'].data['token'])
        lidar_token = this_sample['data']['LIDAR_TOP']
        sd_rec = self.nuscenes.get('sample_data', lidar_token)
        cs_record = self.nuscenes.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = self.nuscenes.get('ego_pose', sd_rec['ego_pose_token'])

        # 2. scene samples
        this_sample = self.nuscenes.get('sample', container_in['img_metas'].data['token'])
        this_sample_idx = 0  # store the index of this_sample in the scene_samples
        scene_samples = []
        scene_samples.append(this_sample)
        while this_sample['next'] != '':
            this_sample = self.nuscenes.get('sample', this_sample['next'])
            scene_samples.append(this_sample)
        this_sample = self.nuscenes.get('sample', container_in['img_metas'].data['token'])
        while this_sample['prev'] != '':
            this_sample = self.nuscenes.get('sample', this_sample['prev'])
            scene_samples.insert(0, this_sample)
            this_sample_idx += 1

        # 3. normalize
        ego_all_trajs = np.zeros((len(scene_samples), 3))
        for i in range(len(scene_samples)):
            lidar_sample_data_i = self.nuscenes.get('sample_data', scene_samples[i]['data']['LIDAR_TOP'])
            pose_record_i = self.nuscenes.get('ego_pose', lidar_sample_data_i['ego_pose_token'])
            cs_record_i = self.nuscenes.get('calibrated_sensor', lidar_sample_data_i['calibrated_sensor_token'])
            ego2global_i = transform_matrix(pose_record_i["translation"], Quaternion(pose_record_i["rotation"]), inverse=False)
            sensor2ego_i = transform_matrix(cs_record_i["translation"], Quaternion(cs_record_i["rotation"]), inverse=False)
            pose_mat_i = ego2global_i.dot(sensor2ego_i)
            ego_all_trajs[i] = pose_mat_i[:3, 3]
        ego_all_trajs = ego_all_trajs - np.array(pose_record['translation'])
        rot_mat = Quaternion(pose_record['rotation']).inverse.rotation_matrix
        ego_all_trajs = np.dot(rot_mat, ego_all_trajs.T).T
        ego_all_trajs = ego_all_trajs - np.array(cs_record['translation'])
        rot_mat = Quaternion(cs_record['rotation']).inverse.rotation_matrix
        ego_all_trajs = np.dot(rot_mat, ego_all_trajs.T).T
        ego_all_trajs = ego_all_trajs[:, :2]
        ego_all_trajs = np.round(ego_all_trajs, 3) # round to 3 decimal places
        return ego_all_trajs, this_sample_idx
    
    def extract_future_traj(self, container_out, container_in):
        out_string = ""
        out_string += "<PLANNING>"
        
        # 4. compute curvature and velocity
        ego_poses, this_sample_idx = self.extract_scene_trajs(container_out, container_in)
        ego_poses_future = ego_poses[this_sample_idx:this_sample_idx+7]
        if len(ego_poses_future) < 7:
            ego_poses_future = np.pad(ego_poses_future, ((0, 7-len(ego_poses_future)), (0, 0)), 'edge')

        ego_dthetas_future = self.compute_traj_to_thetas(ego_poses_future)
        ego_velocity_future = self.compute_traj_to_velocity(ego_poses_future)
        ego_dthetas_future, ego_velocity_future = np.round(ego_dthetas_future, 0), np.round(ego_velocity_future, 2)
        
        ego_future_traj_reconstructed = self.compute_vel_theta_to_traj(ego_velocity_future, ego_dthetas_future)
        
            
        # diff = (torch.tensor(ego_future_traj_reconstructed) - container_in['gt_ego_fut_trajs'].data.cumsum(axis=0)).norm(dim=1)
        # # print(ego_velocity_future, ego_dthetas_future)
        # print(diff)

        out_string += "Predicted future movement details for the next 3 seconds (sampled at 0.5-second intervals), including distance traveled (in meters) and change in direction (in degrees). A value of 0 for direction indicates moving straight, while positive values represent left turns (with the angle increasing during the initial phase and decreasing back to 0 as the turn completes). The output is formatted as [displacement, theta]: "
        for i in range(1, 7):
            displacement = ego_velocity_future[i]
            theta = ego_dthetas_future[i]
            if displacement == -0.0: displacement = 0.0
            if theta == -0.0: theta = 0.0
            out_string += f"[{displacement}, {theta}]"
            if i < 6:
                out_string += ", "
        out_string += "</PLANNING>"
        return out_string
        
    
    def __call__(self, container_out, container_in):
        # write results to container_out["buffer_container"]["planning"]
        container_out["buffer_container"]["planning"] = self.extract_future_traj(container_out, container_in)
        return container_out

    def evaluation_reset(self):
        self.planning_metrics.reset()
        
    def evaluation_update(self, pred, container_in, use_gt_str=False):
        # call self.planning_metrics.update(trajs, gt_trajs, gt_trajs_mask, fut_boxes)
        
        # read from pred['predict], and parse the trajectories.
        # 'predict': '<PLANNING>Trajectory points for the next 3 seconds: [-0.02, 5.22], [-0.09, 5.14], [-0.19, 5.04], [-0.31, 4.94], [-0.44, 4.76], [-0.64, 4.61]</PLANNING>'
        
        trajs_str = pred['predict']
        gt_str = pred['label']
        
        # parse the matches into a list of tuples
        def trajs_str_to_trajs(trajs_str):
            try:
                trajs_str = trajs_str[trajs_str.find("<PLANNING>") + len("<PLANNING>"):trajs_str.find("</PLANNING>")]
                
                # use regex to extract the trajectory points
                pattern = r'\[([^\]]+)\]'
                matches = re.findall(pattern, trajs_str)
                
                # if there are 'displacement and theta in matches, drop them
                if 'displacement, theta' in matches:
                    matches.remove('displacement, theta')
                
                trajs = []
                for match in matches:
                    point = tuple(map(float, match.split(',')))
                    trajs.append(point)
                trajs = torch.tensor(np.array(trajs))[:6, :2]
                
                # if < 6 points, pad last point
                if trajs.shape[0] < 6:
                    pad = trajs[-1:, :].repeat(6 - trajs.shape[0], 1)
                    trajs = torch.cat([trajs, pad], dim=0)
                
                vel, theta = trajs[:, 0], trajs[:, 1]
                vel = np.concatenate([np.zeros(1), vel.numpy()])  # left pad zero
                theta = np.concatenate([np.zeros(1), theta.numpy()])
                trajs = self.compute_vel_theta_to_traj(vel, theta,)[None, ...]  # after cumsum
                trajs = torch.tensor(trajs)
                
            except:
                trajs = torch.zeros(1, 6, 2)
                print(f"Error parsing trajectory: {trajs_str}")
            return trajs
        trajs = trajs_str_to_trajs(trajs_str)
        # get the ground truth trajectories
        gt_trajs = container_in['gt_ego_fut_trajs'].data[None, ...]  # [1, 6, 2]
        sdc_planning = gt_trajs.cumsum(dim=-2).unsqueeze(1)  # [1, 1, 6, 2]
        
        gt_trajs_mask = container_in['gt_ego_fut_masks'].data[None, ...]  # [1, 6]
        sdc_planning_mask = gt_trajs_mask.unsqueeze(-1).repeat(1, 1, 2).unsqueeze(1)  # [1, 1, 6, 2]
        
        
        if use_gt_str:
            print("GT string is used, this will lead to a false collision rate.")
            gt_trajs = trajs_str_to_trajs(gt_str)  # [1, 6, 2]
            sdc_planning = gt_trajs.cumsum(dim=-2).unsqueeze(1)  # [1, 1, 6, 2]
            
            gt_trajs_mask = torch.ones(1, 6)  # [1, 6]
            sdc_planning_mask = gt_trajs_mask.unsqueeze(-1).repeat(1, 1, 2).unsqueeze(1)  # [1, 1, 6, 2]
        
        
        if not sdc_planning_mask.all(): ## for incomplete gt, we do not count this sample
            return

        fut_boxes = container_in['fut_boxes']
        for i in range(len(fut_boxes)):
            fut_boxes[i] = torch.tensor(fut_boxes[i][None, ...])
        
        if use_gt_str:
            fut_boxes = [torch.zeros(1, 0, 7)] * 6
        
        self.planning_metrics.update(trajs, sdc_planning[0, :, :6, :2], sdc_planning_mask[0,:, :6, :2], fut_boxes)

    
    def evaluation_compute(self):
        final_dict = self.planning_metrics.compute()
        final_dict = {
            k: v.tolist() if isinstance(v, torch.Tensor) else v
            for k, v in final_dict.items()
        }
        self.evaluation_reset()
        
        from prettytable import PrettyTable
        planning_tab = PrettyTable()
        metric_dict = {}
        planning_tab.field_names = [
        "metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s", "avg"]
        for key in final_dict.keys():
            value = final_dict[key]
            new_values = []
            for i in range(len(value)):
                new_values.append(np.array(value[:i+1]).mean())
            value = new_values
            avg = [value[1], value[3], value[5]]
            avg = sum(avg) / len(avg)
            value.append(avg)
            metric_dict[key] = avg
            row_value = []
            row_value.append(key)
            for i in range(len(value)):
                if 'col' in key:
                    row_value.append('%.3f' % float(value[i]*100) + '%')
                else:
                    row_value.append('%.4f' % float(value[i]))
            planning_tab.add_row(row_value)
        
        final_dict['table'] = str(planning_tab)
        
        return final_dict



if __name__ == "__main__":
    prompt_stage = PromptNuscenesPlanning(None)
    string = "<PLANNING>Predicted future movement details for the next 3 seconds (sampled at 0.5-second intervals), including distance traveled (in meters) and change in direction (in degrees). A value of 0 for direction indicates moving straight, while positive values represent left turns (with the angle increasing during the initial phase and decreasing back to 0 as the turn completes). The output is formatted as [displacement, theta]: [2.44, -1.0], [2.26, 0.0], [1.97, 0.0], [1.81, 0.0], [1.78, -1.0], [1.79, 0.0]</PLANNING>"
    prompt_stage.evaluation_update({"predict": string}, {"gt_ego_fut_trajs": torch.zeros(1, 6, 2), "gt_ego_fut_masks": torch.ones(1, 6), "fut_boxes": []})


    