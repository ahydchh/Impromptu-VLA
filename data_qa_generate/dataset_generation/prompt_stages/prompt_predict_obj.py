import re
import os
import json
import torch
import numpy as np
from dataset_generation.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import PlanningMetric

# TODO: better make this pipeline to be conditioned on time series data.

class PromptNuscenesPredictObj:
    def __init__(self,):
        self.start_cache = False
        self.cache_filename = "dataset_generation/raw_data/nuscenes_infos/predict_obj.json"
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
    
    def _get_predict_obj(self, container_out, container_in):
        token = container_in['img_metas'].data['token']
        if token in self.cache:
            return self.cache[token]
        
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
        
        predict_obj_str = ""
        for i, obj_attr in enumerate(gt_captions_3d):
            obj_camera = obj_attr["cam_file"].split("/")[1]
            if obj_attr["depth_caption"]["depth"] < 55:  # we only care about objects within 55 meters
                agg_by_camera[obj_camera].append(obj_attr)
        
        # sort from near to far
        for camera, obj_attrs in agg_by_camera.items():
            obj_attrs = sorted(obj_attrs, key=lambda x: x["depth_caption"]["depth"])
            agg_by_camera[camera] = obj_attrs

        for camera, obj_attrs in agg_by_camera.items():
            # predict_obj_str += f"[Camera: {camera}]\n"
            if camera != "CAM_FRONT":
                continue
            
            if len(obj_attrs) == 0:
                predict_obj_str += "I have seen no salient objects in this view.\n"
            else:
                for obj_attr in obj_attrs:
                    dx = obj_attr["localization_caption"]["x_offset"]
                    dy = obj_attr["localization_caption"]["y_offset"]
                    theta = obj_attr["localization_caption"]["localization_theta"]
                    lwh = obj_attr["3d_size"]
                    
                    vx = obj_attr["motion_caption"]["x_vel"]
                    vy = obj_attr["motion_caption"]["y_vel"]
                    
                    v = np.sqrt(vx ** 2 + vy ** 2)
                    theta = np.arctan2(vy, vx) * 180 / np.pi - 90 if v > 0.1 else 0
                    if theta < -180:
                        theta += 360
                    if theta > 180:
                        theta -= 360

                    if v > 0:
                        import IPython; IPython.embed()

                    # dx, dy, lwh round to 2 decimal places, theta quantize to 15
                    dx, dy, lwh = round(dx, 2), round(dy, 2), [round(x, 2) for x in lwh]
                    theta = round(theta / 15) * 15
                    vx, vy = round(vx, 2), round(vy, 2)
                    
                    predict_obj_str += f"The {obj_attr['attribute_caption']['attribute_caption']} ([{dx}, {dy}, {lwh[0]}, {lwh[1]}, {lwh[2]}, {theta}]) is {obj_attr['motion_caption']['motion_caption']} ([{vx}, {vy}] m/s).\n"
        
        self.cache[token] = predict_obj_str
        return predict_obj_str
    
    
    def __call__(self, container_out, container_in):
        predict_obj_str = ""
        predict_obj_str += "<Motion Predict>\n"
        predict_obj_str += "In this stage, I will predict the speed and direction of all detected objects to ensure proper motion planning and collision avoidance. All objects are presented with a form of [x_BEV, y_BEV, l, w, h, theta_in_degree], with predicted speed of form [vx_BEV, vy_BEV].\n"
        predict_obj_str += "I'll finish the task in the same order with the perception stage.\n"
        # predict_obj_str += "I shall take care of the following objects: car, truck, construction vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, traffic cone, traffic light, stop sign, yield sign, speed bump, pothole, manhole, pedestrian_crossing, ped_crossing, divider, and boundary. Besides, I shall take care of unfrequent objects, such as animals, fallen trees, debris, and other unusual obstacles that may appear on the road.\n"

        predict_obj_str += self._get_predict_obj(container_out, container_in)
        predict_obj_str += "</Motion Predict>\n"
        container_out["buffer_container"]["predict_obj"] = predict_obj_str
        
        return container_out
