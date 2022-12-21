from transformers import AutoProcessor, Swinv2Model
import torch
from datasets import load_dataset
from model.detr import SwinDetr
dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

feature_extractor = AutoProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
#model = Swinv2Model.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
model = SwinDetr(num_classes=2)
#inputs = feature_extractor(image, return_tensors="pt")
pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
with torch.no_grad():
    outputs = model(pixel_values)

#last_hidden_states = outputs.last_hidden_state
print(list(outputs))