import logging
import sys
import datasets
import transformers
from transformers.trainer_utils import get_last_checkpoint
import os
import json
import random
import requests
import torch
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    set_seed,
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
import trl
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info
from src.video_r1.utils import (
    LEARNABLE_QUERY,
    START_DIALOGUE,
    END_DIALOGUE,
    THINK_START,
    THINK_END,
    ANSWER_START,
    ANSWER_END,
    BEGIN_COT,
    BEGIN_ANS,
    change_attention_mask
)
from src.video_r1.utils import SYSTEM_MESSAGE, QUESTION_TEMPLATE, TYPE_TEMPLATE
from src.video_r1.trainer.sft_trainer import CustomTrainer_Seperate_Student, CustomTrainer_Seperate_Distilling

from datasets import Dataset, DatasetDict
from typing import List, Dict, Any
logger = logging.getLogger(__name__)

@dataclass
class SFTConfig(trl.SFTConfig):
    mode: str = field(
        default="distilling", metadata={"help": "The mode of the training (teacher or distilling)."}
    )
    learnable_query_num: int = field(
        default=8, metadata={"help": "The number of learnable queries for cot (n + 1)."}
    )
    dataset_path: str = field(
        default="", metadata={"help": "The path to the video folder."}
    )

def prepare_dataset(example: Dict[str, Any], learnable_query_num: int, end_text: str, mode: str) -> Dict[str, List[Dict[str, Any]]]:
    """Prepare dataset example for training."""

    if example["problem_type"] == 'multiple choice':
        question = example['problem'] + "Options:\n"
        for op in example["options"]:
            question += op + "\n"
    else:
        question = example['problem']


    messages_student_cot = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": example['data_type'],
                    example['data_type']: training_args.dataset_path + example['path'][1:]
                    # "max_pixels": 360*420,
                    # "fps": 1.0
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                }
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": LEARNABLE_QUERY * learnable_query_num + BEGIN_COT + example['process']}]
        }
    ]

    messages_student_ans = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": example['data_type'],
                    example['data_type']: training_args.dataset_path + example['path'][1:]
                    # "max_pixels": 360*420,
                    # "fps": 1.0
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                }
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": LEARNABLE_QUERY * learnable_query_num + BEGIN_ANS + example['solution']}]
        }
    ]

    messages_teacher = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_MESSAGE}]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": example['data_type'],
                    example['data_type']: training_args.dataset_path + example['path'][1:]
                    # "max_pixels": 360*420,
                    # "fps": 1.0
                },
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                }
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example['process'] + example['solution']}]
        }
    ]

    return {"messages_student_cot": messages_student_cot, "messages_student_ans": messages_student_ans, "messages_teacher": messages_teacher, "mode": mode}

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts_student_cot = []
    texts_student_ans = []
    texts_teacher = []

    for i, example in enumerate(examples):
        try:
            mode = example["mode"]
            texts_student_cot.append(processor.apply_chat_template(example["messages_student_cot"], tokenize=False))
            texts_student_ans.append(processor.apply_chat_template(example["messages_student_ans"], tokenize=False))
            texts_teacher.append(processor.apply_chat_template(example["messages_teacher"], tokenize=False))
            image_inputs, video_inputs, video_kwargs = process_vision_info(example["messages_student_cot"], return_video_kwargs=True)
            
        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")
    
    inputs_student_cot = processor(
        text=texts_student_cot,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )
    inputs_student_ans = processor(
        text=texts_student_ans,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )
    inputs_teacher = processor(
        text=texts_teacher,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )

    LEARNABLE_QUERY_ID = processor.tokenizer.convert_tokens_to_ids(LEARNABLE_QUERY)
    BEGIN_COT_ID = processor.tokenizer.convert_tokens_to_ids(BEGIN_COT)
    BEGIN_ANS_ID = processor.tokenizer.convert_tokens_to_ids(BEGIN_ANS)
    PAD_ID = processor.tokenizer.pad_token_id
    START_DIALOGUE_ID = processor.tokenizer.convert_tokens_to_ids(START_DIALOGUE)
    END_DIALOGUE_ID = processor.tokenizer.convert_tokens_to_ids(END_DIALOGUE)
    
    """prepare inputs for teacher"""
    if mode == "teacher" or mode == "distilling":
        # print(f"\ntraining texts for teacher: {texts_teacher[0]}\n")
        # pad question tokens
        labels_teacher = inputs_teacher["input_ids"].clone()
        labels_teacher[labels_teacher == processor.tokenizer.pad_token_id] = -100

        for i in range(labels_teacher.size(0)):
            last_index = (labels_teacher[i] == START_DIALOGUE_ID).nonzero(as_tuple=True)[0][-1].item()
            labels_teacher[i, :last_index+3] = -100 # only for response
        
        # update labels
        inputs_teacher["labels"] = labels_teacher
        # print(f"\nTotal token ids for teacher: {inputs_teacher['labels'][0, :]}")
        # print(f"Response token ids for teacher: {inputs_teacher['labels'][0, last_index+3:]}\n")

    """prepare inputs for student"""    
    if mode == "student" or mode == "distilling":
        # print(f"\ntraining texts for student_cot: {texts_student_cot[0]}\n")
        # pad visual and special tokens
        labels_student_cot = inputs_student_cot["input_ids"].clone()
        labels_student_cot[labels_student_cot == processor.tokenizer.pad_token_id] = -100

        for i in range(labels_student_cot.size(0)):
            last_index = (labels_student_cot[i] == START_DIALOGUE_ID).nonzero(as_tuple=True)[0][-1].item()
            labels_student_cot[i, :last_index+3] = -100 # only for response
        pad_num = len(torch.nonzero(labels_student_cot == -100, as_tuple=True)[1].tolist())
        # print(f"\nResponse token ids for student_cot without padding for special tokens: {labels_student_cot[0, last_index+3:]}")
        
        special_token_ids = [LEARNABLE_QUERY_ID, BEGIN_COT_ID, BEGIN_ANS_ID]
        for special_token_id in special_token_ids:
            # print(f"Special token id: {special_token_id}\nPositions of special tokens: {torch.nonzero(labels_student_cot == special_token_id, as_tuple=True)[1].tolist()}")
            pad_num += len(torch.nonzero(labels_student_cot == special_token_id, as_tuple=True)[1].tolist())
            labels_student_cot[labels_student_cot == special_token_id] = -100
        
        assert pad_num == len(torch.nonzero(labels_student_cot == -100, as_tuple=True)[1].tolist()), f"PAD NUM ERROR: pad num: {pad_num}, actual pad num: {len(torch.nonzero(labels_student_cot == -100, as_tuple=True)[1].tolist())}"
        
        # modify attention mask
        custom_attention_mask = change_attention_mask(inputs_student_cot, LEARNABLE_QUERY_ID, BEGIN_COT_ID, BEGIN_ANS_ID, PAD_ID)
        
        # update attention mask & labels
        inputs_student_cot["attention_mask"] = custom_attention_mask
        inputs_student_cot["labels"] = labels_student_cot
        # print(f"Total token ids for student_cot: {inputs_student_cot['labels'][0, :]}")
        # print(f"Response token ids for student_cot: {inputs_student_cot['labels'][0, last_index+3:]}\n")


        ############################################################################
        # print(f"\ntraining texts for student_ans: {texts_student_ans[0]}\n")
        # pad visual and special tokens
        labels_student_ans = inputs_student_ans["input_ids"].clone()
        labels_student_ans[labels_student_ans == processor.tokenizer.pad_token_id] = -100

        for i in range(labels_student_ans.size(0)):
            last_index = (labels_student_ans[i] == START_DIALOGUE_ID).nonzero(as_tuple=True)[0][-1].item()
            labels_student_ans[i, :last_index+3] = -100 # only for response
        pad_num = len(torch.nonzero(labels_student_ans == -100, as_tuple=True)[1].tolist())
        # print(f"\nResponse token ids for student_ans without padding for special tokens: {labels_student_ans[0, last_index+3:]}")
        
        special_token_ids = [LEARNABLE_QUERY_ID, BEGIN_COT_ID, BEGIN_ANS_ID]
        for special_token_id in special_token_ids:
            # print(f"Special token id: {special_token_id}\nPositions of special tokens: {torch.nonzero(labels_student_ans == special_token_id, as_tuple=True)[1].tolist()}")
            pad_num += len(torch.nonzero(labels_student_ans == special_token_id, as_tuple=True)[1].tolist())
            labels_student_ans[labels_student_ans == special_token_id] = -100
        
        assert pad_num == len(torch.nonzero(labels_student_ans == -100, as_tuple=True)[1].tolist()), f"PAD NUM ERROR: pad num: {pad_num}, actual pad num: {len(torch.nonzero(labels_student_ans == -100, as_tuple=True)[1].tolist())}"
        
        # update labels
        inputs_student_ans["labels"] = labels_student_ans
        # print(f"Total token ids for student_ans: {inputs_student_ans['labels'][0, :]}")
        # print(f"Response token ids for student_ans: {inputs_student_ans['labels'][0, last_index+3:]}\n")

    

    if mode == "teacher":
        return inputs_teacher
    elif mode == "student":
        return {
            "inputs_student_cot": inputs_student_cot,
            "inputs_student_ans": inputs_student_ans
        }
    elif mode == "distilling":
        return {
            "inputs_student_cot": inputs_student_cot,
            "inputs_student_ans": inputs_student_ans,
            "inputs_teacher": inputs_teacher,
            "ans_start_tokens": [[processor.tokenizer.convert_tokens_to_ids(ANSWER_START)]],
            "ans_end_tokens": [[processor.tokenizer.convert_tokens_to_ids(ANSWER_END)]]
        }

if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_config}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Load dataset
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Setup model
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_config.model_name_or_path, **model_kwargs
        )
    
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )
    
    # Add special tokens for COT
    print(f"before add special tokens: {len(processor.tokenizer)}, model: {model.get_input_embeddings().weight.shape[0]}") # 151665; 152064
    processor.tokenizer.add_tokens([LEARNABLE_QUERY, THINK_START, THINK_END, ANSWER_START, ANSWER_END, BEGIN_COT, BEGIN_ANS]) # 151665 -> 151672
    print(f"after add special tokens: {len(processor.tokenizer)}, model: {model.get_input_embeddings().weight.shape[0]}") # 151672; 152064 (no need to resieze embedding layer)

    # Prepare dataset
    end_text = processor.tokenizer.decode(198, skip_special_tokens=False)
    prepared_dataset = [prepare_dataset(example, training_args.learnable_query_num, end_text, training_args.mode) for example in dataset['train']]

    # Initialize trainer
    if training_args.mode == "teacher":
        CustomTrainer = SFTTrainer
        print("\nUsing SFTTrainer for teacher training.")
    elif training_args.mode == "student":
        CustomTrainer = CustomTrainer_Seperate_Student
        print("\nUsing CustomTrainer_Seperate_Student for student training.")
    elif training_args.mode == "distilling":
        CustomTrainer = CustomTrainer_Seperate_Distilling
        print("\nUsing CustomTrainer_Seperate_Distilling for distilling training.")
    else:
        raise ValueError(f"Invalid mode: {training_args.mode}")
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=prepared_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
        # tokenizer=processor.tokenizer
    )

    # Train model
    logger.info("*** Train ***")
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
 
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    logger.info(f"Checkpoint detected, resuming training at {checkpoint}.")

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset['train'])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Save final model
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
