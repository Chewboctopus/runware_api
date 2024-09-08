import uuid
import time
from dotenv import load_dotenv
import os
import asyncio
import requests
from PIL import Image
import io
import numpy as np
import torch
from runware import Runware, IImageInference

# Load environment variables from .env file
load_dotenv()

class RunwareTextToImageNode:
    def __init__(self):
        # Retrieve API Key
        self.api_key = os.getenv("RUNWARE_API_KEY")
        self.runware = None
        self.logs = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True}),
                "aspect_ratio": (
                    "STRING", {
                        "default": "16:9",
                        "choices": ["16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"]
                    }
                ),
                "scaling_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 30.0}),
                "steps": ("INT", {"default": 20, "min": 10, "max": 100}),
                "model": ("STRING", {"default": "runware:100@1"}),
                "number_results": ("INT", {"default": 1, "min": 1, "max": 10}),
                "output_type": ("STRING", {"default": "URL"}),
                "output_format": ("STRING", {"default": "PNG"}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE", "TEXT")
    FUNCTION = "generate_image"

    def generate_image(self, prompt: str, negative_prompt: str, aspect_ratio: str, scaling_factor: float, model: str, steps: int, cfg_scale: float, number_results: int, output_type: str, output_format: str):
        self.logs = []  # Reset logs for each run
        unique_run_id = str(time.time())  # Create a unique identifier based on the current time
        self.logs.append(f"Run ID: {unique_run_id}")  # Log the unique run ID
        self.logs.append(f"Generating image with prompt: '{prompt}' and model: '{model}'")

        # Calculate width and height based on aspect ratio and scaling factor
        width, height = self.calculate_dimensions(aspect_ratio, scaling_factor)

        try:
            images = asyncio.run(
                self._generate_image_async(
                    positivePrompt=prompt,
                    negativePrompt=negative_prompt,
                    model=model,
                    width=width,
                    height=height,
                    steps=steps,
                    CFGScale=cfg_scale,
                    outputType=output_type,
                    outputFormat=output_format,
                    numberResults=number_results
                )
            )

            if images is None or len(images) == 0:
                self.logs.append("No images generated.")
                return None, "\n".join(self.logs)

            img_tensors = []
            for img in images:
                image_url = img.imageURL
                self.logs.append(f"Received image URL: {image_url}")

                # Download and convert each image
                img_tensor = self.download_and_convert_image(image_url)
                if img_tensor is not None:
                    img_tensors.append(img_tensor)
                else:
                    self.logs.append(f"Failed to download or process image: {image_url}")

            if img_tensors:
                return img_tensors, "\n".join(self.logs)
            else:
                self.logs.append("No valid images to return.")
                return None, "\n".join(self.logs)

        except Exception as e:
            self.logs.append(f"Error occurred: {str(e)}")
            return None, "\n".join(self.logs)

    async def _generate_image_async(self, positivePrompt: str, negativePrompt: str, model: str, width: int, height: int, steps: int, CFGScale: float, outputType: str, outputFormat: str, numberResults: int):
        try:
            if not self.runware:
                # Establish WebSocket connection if not already done
                self.runware = Runware(api_key=self.api_key)
                await self.runware.connect()

            # Generate unique taskUUID
            taskUUID = str(uuid.uuid4())

            # Create image inference request object
            image_request = IImageInference(
                positivePrompt=positivePrompt,
                negativePrompt=negativePrompt,
                model=model,
                width=width,
                height=height,
                steps=steps,
                CFGScale=CFGScale,
                outputType=outputType,
                outputFormat=outputFormat,
                numberResults=numberResults,
                taskUUID=taskUUID  # Ensure unique UUID
            )
            self.logs.append(f"Sending request to Runware API with params: {image_request}")

            # Send request to Runware API
            images = await self.runware.imageInference(requestImage=image_request)

            if images:
                self.logs.append(f"Received {len(images)} images from Runware")
                return images
            else:
                self.logs.append("Image generation returned no results.")
                return None
        except requests.exceptions.HTTPError as http_err:
            self.logs.append(f"HTTP Error: {http_err}")
            return None
        except Exception as e:
            self.logs.append(f"Unexpected Error: {e}")
            return None

    def download_and_convert_image(self, url):
        """
        Download the image from the given URL and convert it to a PyTorch tensor in RGBA format.
        """
        try:
            # Send a GET request to fetch the image
            response = requests.get(url)
            if response.status_code == 200:
                # Open the image using PIL
                img = Image.open(io.BytesIO(response.content))

                # Ensure the image is in RGB format (convert to RGB if necessary)
                img = img.convert('RGB')

                # Convert the image to a NumPy array
                img_array = np.array(img).astype(np.float32) / 255.0

                # Add an alpha channel (fully opaque)
                alpha_channel = np.ones((img_array.shape[0], img_array.shape[1], 1), dtype=np.float32)

                # Concatenate the alpha channel to the image array (RGBA)
                img_array = np.concatenate([img_array, alpha_channel], axis=-1)

                # Convert from (H, W, C) to (1, H, W, C) for batch size 1, compatible with ComfyUI
                img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Shape: (1, H, W, 4)

                # Squeeze any extra dimension if it's present (e.g., (1, 1, H, W, C))
                img_tensor = img_tensor.squeeze()

                return img_tensor
            else:
                self.logs.append(f"Failed to download image. HTTP Status Code: {response.status_code}")
                return None
        except Exception as e:
            self.logs.append(f"Error downloading the image: {str(e)}")
            return None

    def calculate_dimensions(self, aspect_ratio, scaling_factor):
        """
        Calculate the width and height based on the aspect ratio and scaling factor, rounding to the nearest multiple of 64.
        """
        # Maximum dimensions
        max_width = 2048
        max_height = 2048

        # Aspect ratios
        aspect_ratios = {
            "16:9": (16, 9),
            "1:1": (1, 1),
            "21:9": (21, 9),
            "2:3": (2, 3),
            "3:2": (3, 2),
            "4:5": (4, 5),
            "5:4": (5, 4),
            "9:16": (9, 16),
            "9:21": (9, 21),
        }

        if aspect_ratio in aspect_ratios:
            width_ratio, height_ratio = aspect_ratios[aspect_ratio]
        else:
            width_ratio, height_ratio = 16, 9  # Default to 16:9

        # Calculate width and height with scaling
        width = int(max_width * scaling_factor)
        height = int((width / width_ratio) * height_ratio)

        # Round to the nearest multiple of 64
        width = round(width / 64) * 64
        height = round(height / 64) * 64

        # Ensure the dimensions are within valid bounds
        width = min(max_width, width)
        height = min(max_height, height)

        return width, height
