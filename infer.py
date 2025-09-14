from qwen_vl_utils import process_vision_info
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


# model_path = "Qwen/Qwen2-VL-7B-Instruct"
model_path = "Xiaodong/Open-R1-Video-7B/"

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_path)

# Messages containing a local video path and a text query

# query = """
# What does the person do with the white powdery substance from the larger bowl?
# A. They sprinkle it over the countertop
# B. They pour it into the food processor
# C. They mix it with a liquid
# D. They use a spoon to add it to the food processor
# Answer with the option's letter from the given choices directly.\n
# """
query = """
What does the person do with the white powdery substance from the larger bowl?
A. They sprinkle it over the countertop
B. They pour it into the food processor
C. They mix it with a liquid
D. They use a spoon to add it to the food processor
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.
"""

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "text": "assets/split_5.mp4",
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": query},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    videos=video_inputs,
    fps=1,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)