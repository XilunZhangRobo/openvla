from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import imageio
import torch
import numpy as np
import cv2
from robosuite.controllers import load_controller_config
from robosuite.environments.manipulation.pick_place import PickPlace
import warnings
import os

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

def load_model():
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", 
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    )
    vla.to("cuda:0")
    return processor, vla

def get_action_from_image(processor, vla, image_array: np.ndarray, instruction: str) -> np.ndarray:
    inputs = processor(instruction, Image.fromarray(image_array).convert("RGB")).to("cuda:0", dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key='jaco_play', do_sample=False)
    return action

instruction = "In: What action should the robot take to {Grab the red cube}?\nOut:"
## Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand').
camera_view = 'agentview'
config = load_controller_config(default_controller="OSC_POSE")

env = PickPlace(
    robots="Jaco",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    controller_configs=config,
    camera_names=camera_view,
    control_freq=20,
    camera_heights=224,
    camera_widths=224,
)
obs = env.reset()

writer = imageio.get_writer(
    f"dummyDemo_video.mp4", fps=env.control_freq
)
image_path = 'real_image.png'
## load image 
image_real = cv2.imread(image_path)
## process the image to size 224 by 224

processor, vla = load_model()
for i in range(300):
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    image_input = cv2.flip(obs[f"{camera_view}_image"], 0)
    action = get_action_from_image(processor, vla, image_real, instruction)
    print (action)
    # print (action.shape)
    obs, reward, done, info = env.step(action)
    if writer is not None:
        # writer.append_data(
        #     cv2.rotate(obs[f"{camera_view}_image"], cv2.ROTATE_180))
        
        writer.append_data(image_input)

writer.close()
env.close()
