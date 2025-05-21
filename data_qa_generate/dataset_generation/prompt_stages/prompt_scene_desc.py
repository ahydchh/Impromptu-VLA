SCENE_DESC_PROMPT = """You are an expert in Driving Scene analysing.

You have access to multi-view camera images of a vehicle: (1) front view (which you should focus on with the most attention) <image>, (2) front right view <image>, (3) front left view <image>, (4) back view <image>, (5) back left view <image>, and (6) back right view <image>.

Please describe the scene content and provide a formatted output for the environment description :

### Environment Description:
- **E_weather**: Describe the weather condition (e.g., sunny, rainy, snowy, etc.), and explain its impact on visibility and vehicle traction.
- **E_time**: Describe the time of day (e.g., daytime or nighttime), and explain how it affects driving strategies due to visibility changes.
- **E_road**: Describe the type of road (e.g., urban road or highway), and explain the challenges associated with it for driving.
- **E_lane**: Describe the current lane positioning and possible maneuvers, particularly focusing on lane selection and control decisions.

### Scenario Categories:
- Based on the scene information, tag the category that matches the scene from the following classifications (you can choose multiple categories if applicable):
categories = [
    "Road Construction",
    "Close-range Cut-ins",
    "Roundabout",
    "Animals Crossing Road",
    "Traffic Police Officers",
    "Blocking Traffic Lights",
    "Cutting into Other Vehicle",
    "Ramp",
    "Debris on the Road",
    "Narrow Roads",
    "Pedestrians Popping Out",
    "People on Bus Posters",
    "Complex Intersections",
    "Near Multiple Vehicles",
    "On Pickup Dropoff",
    "Turning at Intersection",
    "Waiting for Traffic Lights",
    "Emergency Vehicles",
    "Parking Lot",
]

Your output format should be a json object with the following structure:
{{
    "E_weather": "$$weather condition$$",
    "E_time": "$$judged by the brightness of the image$$",
    "E_road": "$$road type, can be a list of multiple tags$$",
    "E_lane": "$$lane positioning and possible maneuvers, in less than 3 words$$",
    "Scenario Category": $$category most matching the scene based on the pic you see$$",
    "More description"：$$brief additional details about the scene$$"
}}

Be focus on providing accurate and relevant information. Think step by step, first extract the caption for these images, then analyze the scene content, and finally provide a detailed description of the scene. In the end, summarize your ideas into the above json format.
"""

import os
import re
import json
import signal
from dataset_generation.prompt_stages.utils.external_query import construct_external_query

class PromptNuscenesSceneDesc:
    def __init__(self):
        self.start_cache = False
        self.cache_filename = "dataset_generation/raw_data/nuscenes_infos/scene_desc.json"
        if os.path.exists(self.cache_filename):
            with open(self.cache_filename, "r") as f:
                self.cache = json.load(f)
        else:
            # start the caching process
            self.start_cache = True
            self.cache = {}  # a dictionary to store the scene descriptions
        
        self.external_helper = None  # construct_external_query()


    def cleanup(self, *args):
        if self.start_cache:  # save the cache
            with open(self.cache_filename, "w") as f:
                json.dump(self.cache, f)
            print("Cleanup done.")

    def _get_scene_desc(self, container_out, container_in):
        token = container_in['img_metas'].data['token']
        if token in self.cache:
            return self.cache[token]
        
        self.start_cache = True  # start caching!
        
        if self.external_helper is None:
            self.external_helper = construct_external_query("Qwen/Qwen2.5-VL-72B-Instruct")
        
        attempts = 0
        max_attempts = 3
        helper_ret = None

        while attempts < max_attempts:
            try:
                # query the external model
                helper_ret = self.external_helper.query_with_context(SCENE_DESC_PROMPT, img=container_out['images'])
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
            

        print("Caching the scene description for image", container_out['images'][0])
        print(helper_ret)
        return self.cache[token]

    def _cache_construct_query(self, container_out, container_in):
        this_id = container_in['img_metas'].data['token']
        this_images = container_out['images']
        messages = [
            {"role": "user", "content": SCENE_DESC_PROMPT},
            {"role": "assistant", "content": "[Please fill the scene description based on the image.]"}
        ]
        query = {"id": this_id, "images": this_images, "messages": messages}
        return query

    def _cache_construct_response(self, container_out, container_in, response):
        this_id = container_in['img_metas'].data['token']
        self.start_cache = True  # start caching!
        
        try:
            helper_ret = response["predict"]
            # find ```json ``` and extract what is inside
            pattern = re.compile(r"```json(.*?)```", re.DOTALL)
            helper_ret = pattern.search(helper_ret).group(1)            
            # helper_ret = helper_ret.replace("```json", "").replace("```", "")
            ret = json.loads(helper_ret)
            self.cache[this_id] = ret
        except Exception as e:
            print(this_id)
        

    def __call__(self, container_out, container_in):
        
        scene_desc = self._get_scene_desc(container_out, container_in)
        # {'E_weather': 'Sunny, clear skies. The bright sunlight enhances visibility, reducing the likelihood of glare and ensuring good vehicle traction on the dry road surface.',
        #  'E_time': 'Daytime, as indicated by the bright sunlight and shadows cast by objects, which suggests drivers need to be cautious of potential glare and adjust their speed and attention accordingly.',
        #  'E_road': 'Urban road with multiple lanes, surrounded by buildings and commercial establishments. The road appears to be in a city center with moderate traffic, presenting challenges such as navigating through other vehicles and potential pedestrian crossings.',
        #  'E_lane': 'Middle lane, maintain position.',
        #  'Scenario Category': 'Near Multiple Vehicles',
        #  'More description': 'The scene shows a busy urban street with a variety of vehicles including cars and trucks. The road is flanked by tall buildings, including a prominent casino, and there are palm trees lining the sidewalk. The traffic is moderate, with vehicles in both lanes moving in the same direction. There are no immediate signs of road construction, animals, or emergency vehicles, and the traffic lights are not visible in the image. The overall environment suggests a typical city driving scenario with the need for attentiveness to other vehicles and potential pedestrians.'}
        
        scene_desc_string = ""
        scene_desc_string += "<SCENE_DESC>"
        scene_desc_string += f"Let's start with the environment description: \n"
        if "E_weather" in scene_desc:
            scene_desc_string += f"[Weather]: {scene_desc['E_weather']}\n"
        if "E_time" in scene_desc:
            scene_desc_string += f"[Time]: {scene_desc['E_time']}\n"
        if "E_road" in scene_desc:
            scene_desc_string += f"[Road]: {scene_desc['E_road']}\n"
        if "E_lane" in scene_desc:
            scene_desc_string += f"[Lane]: {scene_desc['E_lane']}\n"
        if "Scenario Category" in scene_desc:
            scene_desc_string += f"[Category]: {scene_desc['Scenario Category']}\n"
        if "More description" in scene_desc:
            scene_desc_string += f"[Summary]: {scene_desc['More description']}\n"
        scene_desc_string += "</SCENE_DESC>"
        
        container_out["buffer_container"]["scene_desc"] = scene_desc_string
        return container_out

