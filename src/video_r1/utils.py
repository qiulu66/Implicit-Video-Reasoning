import torch

LEARNABLE_QUERY = "<pause>"
BEGIN_COT = "<COT>"
BEGIN_ANS = "<ANS>"
START_DIALOGUE = "<|im_start|>"
END_DIALOGUE = "<|im_end|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"

THINK_START = "<think>"
THINK_END = "</think>"
ANSWER_START = "<answer>"
ANSWER_END = "</answer>"

ADDITIONAL_SPECIAL_TOKENS_LIST = [LEARNABLE_QUERY, THINK_START, THINK_END, ANSWER_START, ANSWER_END, BEGIN_COT, BEGIN_ANS]

SYSTEM_MESSAGE = "You are a helpful assistant"

QUESTION_TEMPLATE = (
    "{Question}\n"
)

TYPE_TEMPLATE = {
    "multiple choice": "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": "Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": "Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": "Please provide your text answer within the <answer> </answer> tags.",
    "regression": "Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
}

def change_attention_mask(inputs, LEARNABLE_QUERY_ID, BEGIN_COT_ID, BEGIN_ANS_ID, PAD_ID):
    dtype = torch.bfloat16

    batch_size, seq_len = inputs["input_ids"].shape
    custom_attention_mask = torch.full((batch_size, seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype)
    
    for b in range(batch_size):
        input_ids = inputs["input_ids"][b]
        valid_indices = torch.where(input_ids != PAD_ID)[0]
        valid_sequence = input_ids[valid_indices].tolist()
        original_seq_len = valid_indices.shape[0]
        
        lq_positions = [i for i, id_val in enumerate(valid_sequence) if id_val == LEARNABLE_QUERY_ID]
        begin_cot_idx = next((i for i, id_val in enumerate(valid_sequence) if id_val == BEGIN_COT_ID), len(valid_sequence))
        begin_ans_idx = next((i for i, id_val in enumerate(valid_sequence) if id_val == BEGIN_ANS_ID), len(valid_sequence))
        
        for j in range(seq_len):
            # default setting for self-attention
            custom_attention_mask[b, j, :j+1] = 0
            
            if j >= original_seq_len:
                continue
            current_token = input_ids[valid_indices[j]].item()
            
            # COT process
            if current_token == BEGIN_COT_ID or (begin_cot_idx < j < begin_ans_idx): # <COT>...<|im_end|>
                custom_attention_mask[b, j, :lq_positions[0]] = torch.finfo(dtype).min
            
            # ANS
            elif current_token == BEGIN_ANS_ID or j >= begin_ans_idx: # <ANS>...<|im_end|>
                custom_attention_mask[b, j, lq_positions[-1]+1:begin_ans_idx] = torch.finfo(dtype).min
    
    return custom_attention_mask.unsqueeze(1)
