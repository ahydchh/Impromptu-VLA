import re
import os
import json
import torch
import numpy as np
from dataset_generation.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import PlanningMetric

class PromptNuscenesRoadAgentAnalysis:
    def __init__(self,):
        self.start_cache = False
        self.cache_filename = "dataset_generation/raw_data/nuscenes_infos/road_agent_analysis.json"
        if os.path.exists(self.cache_filename):
            with open(self.cache_filename, "r") as f:
                self.cache = json.load(f)
        else:
            # start the caching process
            self.start_cache = True
            self.cache = {}  # a dictionary to store the scene descriptions
        
        # external helper
        self.tod3cap_path = "dataset_generation/raw_data/final_caption_bbox_token.json"
        if os.path.exists(self.tod3cap_path):
            self.tod3cap = self.load_tod3cap()
        else:
            self.tod3cap = None

    
    def load_tod3cap(self):
        with open(self.tod3cap_path, "r") as f:
            return json.load(f)
    
    def cleanup(self, *args):
        if self.start_cache:
            with open(self.cache_filename, "w") as f:
                json.dump(self.cache, f)
            print("Cleanup done.")
    
    def _get_analysis_obj(self, container_out, container_in):
        token = container_in['img_metas'].data['token']
        # if token in self.cache:
        #     return self.cache[token]
        
        self.start_cache = True  # start caching!
        assert self.tod3cap is not None, "tod3cap is not loaded."
        
        
        # extract captions accoring to the bounding box.
        gt_captions_3d = container_in['gt_captions_3d'].tolist()
        gt_captions_3d = [x for x in gt_captions_3d if x in self.tod3cap]
        gt_captions_3d = [self.tod3cap[x] for x in gt_captions_3d]
        
        agg_by_camera = {
            "CAM_FRONT": [],
            "CAM_FRONT_RIGHT": [],
            "CAM_FRONT_LEFT": [],
            "CAM_BACK": [],
            "CAM_BACK_LEFT": [],
            "CAM_BACK_RIGHT": []
        }
        
        perception_obj_str = ""
        for i, obj_attr in enumerate(gt_captions_3d):
            obj_camera = obj_attr["cam_file"].split("/")[1]
            if obj_attr["depth_caption"]["depth"] < 55:  # we only care about objects within 55 meters
                agg_by_camera[obj_camera].append(obj_attr)
        
        # keep AT MOST 20 objects in the front view
        # rules: categorize into moving_near, moving_far, static_near, static_far
        # moving_near: 0-15m, moving_far: >15m, static_near: 0-15m, static_far: >15m
        # keep order: 1. moving_near, 2. static_near, 3. moving_far, 4. static_far
        # if there are more than 20 objects, keep the first 20 objects
        orders = ["moving_near", "static_near", "moving_far", "static_far"]
        
        # sort from near to far
        for camera, obj_attrs in agg_by_camera.items():
            obj_attrs = sorted(obj_attrs, key=lambda x: x["depth_caption"]["depth"])
            agg_by_camera[camera] = obj_attrs

            # tag the objects
            for obj_attr in obj_attrs:
                vx = obj_attr["motion_caption"]["x_vel"]
                vy = obj_attr["motion_caption"]["y_vel"]
                v = np.sqrt(vx ** 2 + vy ** 2)
                is_move = "moving" if v > 0.5 else "static"
                is_near = "near" if obj_attr["depth_caption"]["depth"] < 15 else "far"
                obj_attr["tag"] = f"{is_move}_{is_near}"
            # sort by tag
            obj_attrs = sorted(obj_attrs, key=lambda x: orders.index(x["tag"]))
            
            if len(obj_attrs) > 20:
                obj_attrs = obj_attrs[:20]  # keep the first 20 objects
        
        
        for camera, obj_attrs in agg_by_camera.items():
            if camera != "CAM_FRONT":
                continue
            if len(obj_attrs) == 0:
                perception_obj_str += "I have seen no notable objects in this view.\n"
            else:
                for obj_attr in obj_attrs:
                    # import random
                    # if random.random() < 0.04:
                    #     print(dx, dy,obj_attr["localization_caption"]["localization_theta"])
                    #     from PIL import Image, ImageDraw
                    #     img = Image.open(container_out['images'][0])
                    #     draw = ImageDraw.Draw(img)
                    #     draw.rectangle(obj_attr['2d_bbox'], outline="red")
                    #     img.save("trash.jpg")
                    #     import IPython; IPython.embed()
                    # fv_cam_intrinsic = container_in['cam_intrinsic'][0]  # [3, 3]
                    # bbox_center = np.array([(obj_attr['2d_bbox'][0] + obj_attr['2d_bbox'][2]) / 2, (obj_attr['2d_bbox'][1] + obj_attr['2d_bbox'][3]) / 2])
                    # # transform to 3d coordinate
                    # bbox_3d_center = np.dot(np.linalg.inv(fv_cam_intrinsic), np.array([bbox_center[0], bbox_center[1], 1]))
                    

                    # bbox_3d_center_selected = bbox_3d_center[[0, 2]]  # 选取索引 0 和 2 的值
                    # norm = np.linalg.norm(bbox_3d_center_selected)  # 使用 numpy 的 linalg.norm 函数
                    # result = (bbox_3d_center_selected / norm) * obj_attr['depth_caption']['depth']
                    
                    # location
                    dx = obj_attr["localization_caption"]["x_offset"]  # positive means forward
                    dy = obj_attr["localization_caption"]["y_offset"]  # positie means leftward
                                        
                    # dx, dy = -dy, dx  # convert to BEV coordinate
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    # theta = np.arctan2(dy, dx).item() * 180 / np.pi - 90  # in degree, positive means leftward
                    theta = -obj_attr["localization_caption"]["localization_theta"]  # in degree, positive means leftward
                    
                    # theta = obj_attr["localization_caption"]["localization_theta"]
                    # lwh = obj_attr["3d_size"]
                    
                    # dx, dy, lwh round to 2 decimal places, theta quantize to 1
                    distance, theta = round(distance, 2), round(theta, 0)
                    # theta = round(theta / 15) * 15
                    
                    # moving
                    vx = obj_attr["motion_caption"]["x_vel"] / 2
                    vy = obj_attr["motion_caption"]["y_vel"] / 2
                    vx, vy = -vy, vx  # convert to BEV coordinate
                    v = np.sqrt(vx ** 2 + vy ** 2)
                    vtheta = np.arctan2(vy, vx).item() * 180 / np.pi - 90 if v > 0.1 else 0  # in degree, positive means leftward
                    v, vtheta = round(v, 2), round(vtheta, 0)
                    
                    if v > 0.5:
                        print(v, vtheta, obj_attr["motion_caption"]["x_vel"], obj_attr["motion_caption"]["y_vel"])
                        from PIL import Image, ImageDraw
                        img = Image.open(container_out['images'][0])
                        draw = ImageDraw.Draw(img)
                        draw.rectangle(obj_attr['2d_bbox'], outline="red")
                        img.save("trash.jpg")
                        import IPython; IPython.embed()
                    
                    if 'moving' in obj_attr['tag']:
                        perception_obj_str += f"A {obj_attr['attribute_caption']['attribute_caption']} is seen moving at [{distance}, {theta}], with a velocity of [{v}, {vtheta}].\n"
                    else:
                        perception_obj_str += f"A {obj_attr['attribute_caption']['attribute_caption']} is seen stationary at [{distance}, {theta}].\n"
                    
                    # perception_obj_str += f"A {obj_attr['attribute_caption']['attribute_caption']} ([{dx}, {dy}, {lwh[0]}, {lwh[1]}, {lwh[2]}, {theta}]) is seen {obj_attr['localization_caption']['localization_caption']} {obj_attr['depth_caption']['depth_caption']} (around {round(obj_attr['depth_caption']['depth'], 2)} meters away).\n"
        
        self.cache[token] = perception_obj_str
        return perception_obj_str
    
    
    def __call__(self, container_out, container_in):
        perception_obj_str = ""
        perception_obj_str += "<Road Agent Analysis>\n"
        perception_obj_str += "In this stage, my goal is to maintain a high level of situational awareness, ensuring that all agents on the road, whether common or rare, are perceived and predicted with accurate motion of future to ensure safe and reliable operation of the vehicle."
        perception_obj_str += "I'll perceive all of the objects in (1) front view image. All locations are provided with [distance, direction], where the direction value indicates the relative direction to ego car in degrees (0 for straight, positive for left turn, negative for right turn). If the object is moving, the velocity is also provided in the form of [displacement_of_next_timestamp, direction].\n"
        # perception_obj_str += "I'll perceive all of the objects in (1) front view image. All objects are presented with a form of [x_BEV, y_BEV, l, w, h, theta_in_degree].\n"
        perception_obj_str += "I shall take care of the following objects: car, truck, construction vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, traffic cone, traffic light, stop sign, yield sign, speed bump, pothole, manhole, pedestrian_crossing, ped_crossing, divider, and boundary. Besides, I shall take care of unfrequent objects, such as animals, fallen trees, debris, and other unusual obstacles that may appear on the road.\n"

        perception_obj_str += self._get_analysis_obj(container_out, container_in)
        perception_obj_str += "</Road Agent Analysis>\n"
        container_out["buffer_container"]["road_agent_analysis"] = perception_obj_str
        
        return container_out
