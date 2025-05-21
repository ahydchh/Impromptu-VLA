import os

class PromptNuscenesMetadata:
    def __init__(self):
        self.user_prompt = "You are an autonomous driving agent. You have access to multi-view camera images of a vehicle: (1) front view (which you should focus on with the most attention) <image>, (2) front right view <image>, (3) front left view <image>, (4) back view <image>, (5) back left view <image>, and (6) back right view <image>. Your task is to do your best to predict future waypoints for the vehicle over the next 3 timesteps, given the vehicle's intent inferred from the images. Please format your output as a list of [distance traveled, change of direction] for each of the last 6 timesteps, where the direction value indicates the change in direction in degrees (0 for straight, positive for left turn, negative for right turn)."

    def __call__(self, container_out, container_in):
        container_out = {
            "id": container_in['img_metas'].data['token'],
            "images": [ os.path.join("dataset_generation/raw_data/nuscenes",  x[x.find("samples"):]) for x in container_in['img_metas'].data['img_filename']],
            "messages": [
                {"role": "user", "content": self.user_prompt}, 
                {"role": "assistant", "content": ""}
            ],
            "buffer_container": {}
        }
        
        return container_out
