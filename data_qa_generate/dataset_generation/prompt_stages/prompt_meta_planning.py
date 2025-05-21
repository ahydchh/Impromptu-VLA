import os
import re
import json
import torch
import numpy as np
from dataset_generation.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import PlanningMetric
from dataset_generation.prompt_stages.utils.external_query import construct_external_query
from nuscenes import NuScenes

class PromptNuscenesMetaPlanning:
    def __init__(self, nuscenes, previous_stage_cache_path):
        self.start_cache = False
        self.cache_filename = "dataset_generation/raw_data/nuscenes_infos/meta_planning.json"
        if os.path.exists(self.cache_filename):
            with open(self.cache_filename, "r") as f:
                self.cache = json.load(f)
        else:
            # start the caching process
            self.start_cache = True
            self.cache = {}  # a dictionary to store the scene descriptions
        
        self.external_helper = None  # construct_external_query()
        
        # self.nuscenes_3d = nuscenes
        self.nuscenes = nuscenes
        
        # load ALL previous caches for the following stages
        self.previous_stage_cache_path = previous_stage_cache_path
        self.previous_stage_cache = None
        if os.path.exists(self.previous_stage_cache_path):
            with open(self.previous_stage_cache_path, "r") as f:
                self.previous_stage_cache = json.load(f)
            self.previous_stage_cache = {
                x['id']: x for x in self.previous_stage_cache
            }
                
        
        # Speed-control actions. Discerned from acceleration and braking signals within the ego state data.
        # These actions include speed up, slow down, slow down rapidly, go straight slowly, go straight at a constant speed, stop, wait, reverse, and maintain current speed.
        self.speed_meta_actions = ['speed up', 'slow down', 'slow down rapidly', 'go straight slowly', 'go straight at a constant speed', 'stop', 'wait', 'reverse', 'maintain current speed']

        # Turning actions. Deduced from steering wheel signals.
        # These actions consist of turn left, turn right, turn around, and maintain current direction.
        self.turning_meta_actions = ['turn left', 'turn right', 'turn around', 'maintain current direction']

        # Lane-control actions. Encompassing lane selection decisions derived from a combination of steering wheel signals and either map or perception data.
        # These actions involve change lane to the left, change lane to the right, shift slightly to the left, shift slightly to the right, and maintain current lane.
        self.lane_control_meta_actions = ['change lane to the left', 'change lane to the right', 'shift slightly to the left', 'shift slightly to the right', 'maintain current lane']


        self.query_prompt = """## Problem Statement\nYou are an expert in autonomous driving planning analysis. You will be shown front view pictures of a driving scene: one representing the current time <image>, one representing the ego view 1.5 seconds later <image> and one representing the scene 3 seconds later <image>.  You need to select the most appropriate action from each of the following categories:

1. **Speed-control actions** (discerned from acceleration and braking signals within the ego state data):
    - speed up rapidly
    - speed up
    - slow down
    - slow down rapidly
    - go straight slowly
    - go straight at a constant speed
    - stop
    - wait
    - reverse
    - maintain current speed

2. **Turning actions** (deduced from steering wheel signals):
    - turn left - starting phase
    - turn left - ending phase
    - turn right - starting phase
    - turn right - ending phase
    - turn around - starting phase
    - turn around - ending phase
    - maintain current direction

3. **Lane-control actions** (encompassing lane selection decisions derived from a combination of steering wheel signals and either map or perception data):
    - change lane to the left - starting phase
    - change lane to the left - ending phase
    - change lane to the right - starting phase
    - change lane to the right - ending phase
    - shift slightly to the left
    - shift slightly to the right
    - maintain current lane

Please provide your reasoning and the selected actions in the following JSON format:

{
    "M_speed": "Because ..., I should xxxxx",
    "M_direction": "Because ..., I should xxxxx",
    "M_lane": "Because ..., I should xxxxx"
}

In the Context section below, you will be provided with the ground truth velocities and change of directions of ego vehicle. For the reasoning part, or the "Because" part, you need to find the reason from the environment in **current frames**, i.e., the image of now, not from future frames. For the action planning, you need to deduce the meta action from the ego status and the ground truth velocities and change of directions. Do NOT use the ego status and ground truth velocities and change of directions as your reasons, as they are ground truth information that won't be available in real-world scenarios. You need to think why such ego status and ground truth velocities and change of directions happen, i.e., what action leads to such status and why such action is taken. Example: Because there is a white car cutting in, I should ... / Because I need to yield the way for the pedestrain, I need to shift slightly to the right. / Because I am nearly done the turn and already parallel to the main lane, I should turn right - ending phase. Write more with your creativity.\n## Context\n"""

    def cleanup(self, *args):
        if self.start_cache:
            with open(self.cache_filename, "w") as f:
                json.dump(self.cache, f)
            print("Cleanup done.")
    
    
    def _get_final_query(self, container_out, container_in):
        # query the external model
        # 0. basic check
        assert self.previous_stage_cache is not None, "Previous stage cache is not loaded."
        
        # 1. get current_sample, future_sample
        this_sample = self.nuscenes.get('sample', container_in['img_metas'].data['token'])
        
        sample_t_tp6 = []  # [t, t+3s]
        sample_t_tp6.append(this_sample)
        while len(sample_t_tp6) < 7 and this_sample['next'] != '':
            this_sample = self.nuscenes.get('sample', this_sample['next'])
            sample_t_tp6.append(this_sample)
        
        if len(sample_t_tp6) < 7:
            # not enough samples. pad from the left side with previous samples
            this_sample = self.nuscenes.get('sample', container_in['img_metas'].data['token'])
            while len(sample_t_tp6) < 7 and this_sample['prev'] != '':
                this_sample = self.nuscenes.get('sample', this_sample['prev'])
                sample_t_tp6.insert(0, this_sample)
        current_image_token, after_image_token, future_image_token = sample_t_tp6[0]['data']['CAM_FRONT'], sample_t_tp6[3]['data']['CAM_FRONT'], sample_t_tp6[6]['data']['CAM_FRONT']
        current_image = self.nuscenes.get('sample_data', current_image_token)['filename']
        after_image = self.nuscenes.get('sample_data', after_image_token)['filename']
        future_image = self.nuscenes.get('sample_data', future_image_token)['filename']
        current_image = os.path.join(self.nuscenes.dataroot, current_image)
        after_image = os.path.join(self.nuscenes.dataroot, after_image)
        future_image = os.path.join(self.nuscenes.dataroot, future_image)
        
        prompt_t_tp6 = []
        for sample in sample_t_tp6:
            prompt_t_tp6.append(self.previous_stage_cache[sample['token']])
        
        # 2. construct more information for query
        # 2.1 ego status
        query_prev_ego_status = "### 1. Ego status for the previous 3 seconds (can be used in the reasoning part)\n"
        
        start_idx = container_out['messages'][0]['content'].find("Provided are the previous ego vehicle statuses recorded over")
        query_prev_ego_status += container_out['messages'][0]['content'][start_idx:]
        
        # query_ego_status = "### 2. Ego status for the upcoming 3 seconds (for deducing plans only)\n"
        # query_ego_status_list = []
        # # match this
        # # Acceleration: [0.000, 0.000, 0.000] m/s^2\nAngular velocity: [0.000, 0.000, 0.000] rad/s\nVelocity: [0.000, 0.000, 0.000] m/s\nSteering angle: 0.0 (positive: left turn, negative: right turn)
        # for i, prompt in enumerate(prompt_t_tp6):
        #     orig_input = prompt['messages'][0]['content']
        #     orig_input = orig_input[orig_input.find("Acceleration"):]
        #     orig_input = orig_input.strip().replace("\n", ", ")
        #     query_ego_status_list.append(f"t+{i*0.5}s: {orig_input}")
        # query_ego_status += "\n".join(query_ego_status_list)
        # query_ego_status += "\n\n"
        
        # # 2.2 perception objects and predictions
        # query_agents = "### 1. Environment Caption, Perceived objects in the scene and predictions for their movement in the current frame (for reasoning and planning)\n"
        # query_agents_candidate = prompt_t_tp6[0]['messages'][1]['content'][:prompt_t_tp6[0]['messages'][1]['content'].find("<PLANNING>")]
        # # query_agents_candidate = query_agents_candidate[query_agents_candidate.find("All locations are"):]
        # query_agents += query_agents_candidate
        
        
        # 2.3. gt trajectory
        query_gt_trajectory = "### 2. Ground truth movement per 0.5 second for the ego vehicle in the upcoming 3 seconds (for deducing plans only, NOT for reasoning, please look at this with most care)\n"
        query_gt_trajectory += prompt_t_tp6[0]['messages'][1]['content'][prompt_t_tp6[0]['messages'][1]['content'].find("<PLANNING>"):].replace("Predicted", "")
        query_gt_trajectory += "Note that positive y indicates forward movement, and positive x indicates rightward movement."
        
        
        query_final = f"{self.query_prompt}\n\n{query_prev_ego_status}\n{query_gt_trajectory}\n\n## Answer\nPlease start by thinking step by step and finally formulate your ideas into the JSON format. Compared with previous ego status, are the velocity vectors in the future speeding up, keeping a constant speed, or slowing down? (Extract these numbers and comparing them. Note that the velocity may change through time. e.g. At the turning the speed maybe slow, but after turning the speed is increased. You may answer, keep constant speed at the turn and speed up after turning.) Is the direction changing? If it is changing, is it turning left, turning right, or turning around? Then look at the images, are they consistent with your judgement on the velocity and direction? Use this to deduce the meta planning actions first. Then, look at the picture of **only current frame** again, what is the reason for such velocity and direction in the future? After analyzing, write down your reasoning and the selected actions in the above JSON format."
        return query_final, [current_image, after_image, future_image]
    
    def _get_meta_planning(self, container_out, container_in):
        token = container_in['img_metas'].data['token']
        if token in self.cache:
            return self.cache[token]
        
        self.start_cache = True
        if self.external_helper is None:
            self.external_helper = construct_external_query("Qwen/Qwen2.5-VL-72B-Instruct")
        
        attempts = 0
        max_attempts = 3
        helper_ret = None
        
        while attempts < max_attempts:
            try:
                query_final, [current_image, after_image, future_image] = self._get_final_query(container_out, container_in)
                helper_ret = self.external_helper.query_with_context(query_final, img=[current_image, after_image, future_image])
                # helper_ret = '```json\n{\n....}\n```' now remove the ```json and ```, and parse the json
                pattern = re.compile(r"```json(.*?)```", re.DOTALL)
                helper_ret = pattern.search(helper_ret).group(1)   
                helper_ret = json.loads(helper_ret)
                self.cache[token] = helper_ret
                self.cleanup()  # save the cache
                break  # Break the loop if successful
            except Exception as e:
                print(f"Error parsing the external model response on attempt {attempts + 1}:")
                print(e)
                attempts += 1

        if helper_ret is None:
            helper_ret = {}
            
        print("Caching the meta planning for image", container_out['images'][0])
        print(helper_ret)
        return self.cache[token]
    
    def _cache_construct_query(self, container_out, container_in):
        query_final, this_images = self._get_final_query(container_out, container_in)
        this_id = container_in['img_metas'].data['token']
        messages = [
            {"role": "user", "content": query_final},
            {"role": "assistant", "content": "[Please fill the meta planning based on the images and the context.]"}
        ]
        query = {"id": this_id, "images": this_images, "messages": messages}
        return query
    
    def _cache_construct_response(self, container_out, container_in, response):
        this_id = container_in['img_metas'].data['token']
        self.start_cache = True
        
        try:
            helper_ret = response["predict"]
            pattern = re.compile(r"```json(.*?)```", re.DOTALL)
            helper_ret = pattern.search(helper_ret).group(1)
            ret = json.loads(helper_ret)
            self.cache[this_id] = ret
        except Exception as e:
            print(this_id)
    
    def __call__(self, container_out, container_in):
        # write results to container_out["buffer_container"]["meta_planning"]
        meta_planning_dict = self._get_meta_planning(container_out, container_in)
        meta_planning_str = "<Meta Planning>\n"
        if "M_speed" in meta_planning_dict:
            meta_planning_str += f"Speed: {meta_planning_dict['M_speed']}\n"
        if "M_direction" in meta_planning_dict:
            meta_planning_str += f"Direction: {meta_planning_dict['M_direction']}\n"
        if "M_lane" in meta_planning_dict:
            meta_planning_str += f"Lane: {meta_planning_dict['M_lane']}\n"
        meta_planning_str += "</Meta Planning>\n"
         
        container_out["buffer_container"]["meta_planning"] = meta_planning_str
        return container_out
