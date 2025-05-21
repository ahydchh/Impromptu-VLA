import re
import os
import json
import torch
import numpy as np
from dataset_generation.mmdet3d_plugin.datasets.evaluation.planning.planning_eval import PlanningMetric
from dataset_generation.prompt_stages.utils.external_query import construct_external_query


class PromptNuscenesRoadAgentAnalysisText:
    def __init__(self,):
        self.start_cache = False
        self.cache_filename = "dataset_generation/raw_data/nuscenes_infos/road_agent_analysis_text.json"
        if os.path.exists(self.cache_filename):
            with open(self.cache_filename, "r") as f:
                self.cache = json.load(f)
        else:
            # start the caching process
            self.start_cache = True
            self.cache = {}  # a dictionary to store the scene descriptions
    
        self.external_helper = None
    
    def cleanup(self, *args):
        if self.start_cache:
            with open(self.cache_filename, "w") as f:
                json.dump(self.cache, f)
            print("Cleanup done.")
    
    def _get_final_query(self, container_out, container_in):
        prompt = f"""You are an expert in driving scene analysis. Your task is to analyze a driving scene using information provided from six camera views:  
1. **Front view** (focus on this view with the most attention) <image>  
2. **Front-right view** <image>  
3. **Front-left view** <image>  
4. **Back view** <image>  
5. **Back-left view** <image>  
6. **Back-right view** <image>  

In this stage, your task is to identify **critical objects** in the scene and analyze them based on the following three perspectives:  

1. **Static Attributes**: Describe the inherent, unchanging properties of the object. These include visual characteristics (e.g., a roadside billboard, a stationary parked car, or oversized cargo on a truck).  
2. **Motion States**: Describe the object's dynamics, including its position, direction of movement, speed, and current actions (e.g., a pedestrian crossing the street, a vehicle merging into the lane, or a cyclist stopped at a red light).  
3. **Particular Behaviors**: Highlight any special or unusual actions or gestures that could influence the ego vehicle’s next decision (e.g., a pedestrian waving to signal crossing intentions, a car suddenly braking, or a cyclist swerving into the lane).  

After analyzing these traits, you must assess the **potential influence** of these objects on the ego vehicle, explaining how they might affect its driving decisions, such as slowing down, changing lanes, or stopping.  

Your output should be in JSON format with the following structure:  

```json
{{
    "Critical_Objects": [
        {{
            "class": "$category of the object, using nouns$",
            "characteristics": "$description of the object from Static Attributes, Motion States, and Particular Behaviors, written in a paragraph$",
            "influence": "$potential influence of the object on the ego vehicle, written in a paragraph$"
        }},
        {{
            "class": "$category of another object$",
            "characteristics": "$description of this object from Static Attributes, Motion States, and Particular Behaviors, written in a paragraph$",
            "influence": "$potential influence of this object on the ego vehicle, written in a paragraph$"
        }}
        // Add more objects as needed
    ]
}}
```

**Examples for Better Understanding:**

1. **Example 1**: A pedestrian crossing the road.  
   ```json
   {{
       "Critical_Objects": [
           {{
               "class": "pedestrian",
               "characteristics": "A pedestrian is wearing a red jacket and carrying a backpack. They are currently walking across a zebra crossing from left to right. The pedestrian has turned their head toward the ego vehicle, possibly checking for traffic.",
               "influence": "The ego vehicle should slow down and prepare to stop to allow the pedestrian to cross safely, as they are directly in the vehicle's path."
           }}
       ]
   }}
   ```

2. **Example 2**: A stopped vehicle with hazard lights on.  
   ```json
   {{
       "Critical_Objects": [
           {{
               "class": "stopped vehicle",
               "characteristics": "A white sedan with visible hazard lights on the right side of the road. The vehicle is stationary with no visible driver inside. The hazard lights suggest it may be broken down or parked temporarily.",
               "influence": "The ego vehicle should maintain a safe distance and consider changing lanes to avoid potential obstruction caused by the stopped vehicle."
           }}
       ]
   }}
   ```

3. **Example 3**: A cyclist swerving into the lane.  
   ```json
   {{
       "Critical_Objects": [
           {{
               "class": "cyclist",
               "characteristics": "A cyclist wearing a yellow helmet and blue jacket. The cyclist is moving at a moderate speed but is swerving slightly into the ego vehicle's lane. The swerving behavior suggests they may be avoiding an obstacle or losing control.",
               "influence": "The ego vehicle should reduce speed and maintain a larger buffer zone to avoid a potential collision if the cyclist continues swerving into the lane."
           }}
       ]
   }}
   ```

4. **Example 4**: A truck with oversized cargo.  
   ```json
   {{
       "Critical_Objects": [
           {{
               "class": "truck",
               "characteristics": "A large truck with oversized cargo extending beyond its rear. The truck is moving slowly in the right lane. The oversized cargo is poorly secured and appears to sway slightly with the vehicle's motion.",
               "influence": "The ego vehicle should avoid driving too close to the truck and consider overtaking it safely to minimize the risk of collision with the swaying cargo."
           }}
       ]
   }}
   ```

5. **Example 5**: A dog running across the road.  
   ```json
   {{
       "Critical_Objects": [
           {{
               "class": "dog",
               "characteristics": "A small brown dog is running diagonally across the road from the right sidewalk to the left. The dog is moving unpredictably and may stop or change direction suddenly.",
               "influence": "The ego vehicle should slow down immediately and prepare to stop to avoid hitting the dog, as its movements are erratic and it is crossing the vehicle's path."
           }}
       ]
   }}
   ```

By following this structure and using the examples as a guide, you can provide detailed and actionable analyses of driving scenes. Output JSON in at most 800 tokens. Now give your output in the above JSON format:\n"""
        return prompt, container_out["images"]

    
    def _cache_construct_query(self, container_out, container_in):
        query_final, this_images = self._get_final_query(container_out, container_in)
        this_id = container_in["img_metas"].data["token"]
        messages = [
            {"role": "user", "content": query_final},
            {"role": "assistant", "content": "[Please fill the road agent analysis based on the images and the context.]"}
        ]
        query = {"id": this_id, "images": this_images, "messages": messages}
        return query



    def fix_helper_ret(self, helper_ret: str):
        helper_ret_lines = helper_ret.split('\n')  # Split by newline
        processed_lines = []

        for line in helper_ret_lines:
            quote_indices = [i for i, char in enumerate(line) if char == '"']  # Find all quote positions
            
            # A piece of UGLY code that maps
            #  "characteristics": "A white van with "FACTORY OUTLET" writt...
            # to
            #  "characteristics": "A white van with FACTORY OUTLET writt...
            if len(quote_indices) > 4:
                # Keep the first three quotes and the last one
                keep_indices = quote_indices[:3] + [quote_indices[-1]]
                kill_indices = []
                for idx in quote_indices:
                    if idx not in keep_indices:
                        kill_indices.append(idx)
                # REMOVE all quotes that are in kill_indices, from right to left
                new_line = line
                for i in kill_indices[::-1]:
                    new_line = new_line[:i] + new_line[i + 1:]
                    
                processed_lines.append(new_line)
            else:
                # Keep the line as is if it has 4 or fewer quotes
                processed_lines.append(line)

        helper_ret_fixed = '\n'.join(processed_lines)
        return helper_ret_fixed


    def _cache_construct_response(self, container_out, container_in, response):
        this_id = container_in['img_metas'].data['token']
        self.start_cache = True  # start caching!
        
        try:
            helper_ret = response["predict"]
            pattern = re.compile(r"```json(.*?)```", re.DOTALL)
            helper_ret = pattern.search(helper_ret).group(1)
            helper_ret_fixed = self.fix_helper_ret(helper_ret)
            ret = json.loads(helper_ret_fixed)
            self.cache[this_id] = ret
        except Exception as e:
            print(this_id)
        


    def _get_analysis_obj(self, container_out, container_in):
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
                query_final, this_images = self._get_final_query(container_out, container_in)
                helper_ret = self.external_helper.query_with_context(query_final, img=this_images)
                # helper_ret = '```json\n{\n....}\n```' now remove the ```json and ```, and parse the json
                pattern = re.compile(r"```json(.*?)```", re.DOTALL)
                helper_ret = pattern.search(helper_ret).group(1)
                helper_ret_fixed = self.fix_helper_ret(helper_ret)
                helper_ret = json.loads(helper_ret_fixed)
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
    
    def __call__(self, container_out, container_in):
        perception_obj_dict = self._get_analysis_obj(container_out, container_in)
        perception_obj_str = "<Road Agent Analysis>\n"
        perception_obj_str = "Then I'll analyze all critical agents in the scene based on their static attributes, motion states, and particular behaviors. I'll also assess their potential influence on the ego vehicle.\n\n"
        
        for obj in perception_obj_dict["Critical_Objects"]:
            perception_obj_str += f"[{obj['class']}] "
            perception_obj_str += obj['characteristics'] + " "
            perception_obj_str += obj['influence'] + "\n"
        
        perception_obj_str += "</Road Agent Analysis>\n"
        container_out["buffer_container"]["road_agent_analysis"] = perception_obj_str
        
        return container_out
