! pip install accelerate diffusers controlnet_aux   #Install the required pip




import torch
from controlnet_aux import CannyDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image,make_image_grid



controlnet = ControlNetModel.from_pretrained( "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, varient="fp16")
pipe = StableDiffusionControlNetPipeline.from_pretrained( "Yntec/AbsoluteReality", controlnet=controlnet, torch_dtype=torch.float16)



pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe.enable_model_cpu_offload()



from google.colab import files
files.upload()
print("upload is done")



img = load_image("unititled.jpg")
img



pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe.enable_model_cpu_offload()



from google.colab import files
files.upload()
print("upload is done")


img = load_image("unititled.jpg")
img



ip_adap_img = load_image("ghy2.jpg")
ip_adap_img



canny = CannyDetector()
canny_img = canny(img,detect_resolution=512,image_resolution=768)
canny_img


prompt = """ (photorealistic:0.5), raw, clean shaved rusty,rounded hairtip, 2d cartoon style, keep the dress color exact, 512px,chrome yellow palette, smooth palette, nose will be small,starting from the eyebrow,,rounded faces,ghibli studio, pastel images, muted color,round eyes much closer to eyebrows, pronounced nosebridge, longer nose, eyes bigger, less round, maintain expression"""
pipe.set_ip_adapter_scale(0.5)
images = pipe(prompt = prompt,
              negative_prompt = "detailed face, high quality,chiseled, bright colors", height = 768, width = 768, ip_adapter_image = ip_adap_img, image = canny_img, guidance_scale = 9.5, controlnet_conditioning_scale = 0.6, num_inference_steps = 20, num_images_per_prompt = 3).images



make_image_grid(images,cols=3,rows=1)
