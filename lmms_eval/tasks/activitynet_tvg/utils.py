import datetime
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from loguru import logger as eval_logger

import lmms_eval.tasks._task_utils.file_utils as file_utils
import datasets
   

hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "activitynet_tvg.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)

dataset_kwargs = yaml.safe_load("".join(safe_data))["dataset_kwargs"]
cache_name = dataset_kwargs["cache_dir"]
cache_dir = os.path.join(base_cache_dir, cache_name)

# video_dir can be a relative path or an absolute path
# video_dir = dataset_kwargs["video_dir"]
# video_dir = video_dir if os.path.isabs(video_dir) else os.path.join(cache_dir, video_dir)
video_dir = os.path.join(cache_dir, "videos")

def temporal_grounding_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    # Before entering the evaluation loop, check if all video files exist
    eval_logger.info("Checking for missing video files")
    has_missing_files = False
    for i, doc in enumerate(dataset):
        video_filename = doc["video"]
        video_path = os.path.join(video_dir, video_filename)
        if not os.path.exists(video_path):
            eval_logger.error(f"Missing video file: {video_path}")
            has_missing_files = True
    
    if has_missing_files:
        sys.exit("Cannot proceed evaluation due to missing video files. Please check the logs above.")
    else:
        eval_logger.info("All video files are present. Continue evaluation.")


def temporal_grounding_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    video_filename = doc["video"]
    video_path = os.path.join(video_dir, video_filename)
    return [video_path]


def temporal_grounding_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    question = doc["caption"]

    input_text = question
    if not (pre_prompt is None or pre_prompt == ""):
        input_text = f"{pre_prompt} {input_text}"
    if not (post_prompt is None or post_prompt == ""):
        input_text = f"{input_text} {post_prompt}"
        
    return input_text


def temporal_grounding_doc_to_target(doc):
    return doc["timestamp"]

def temporal_grounding_process_results_generation(doc, result):
    pred = result[0]
    return {"submission": {f'{doc["video"]}>>>{doc["caption"]}>>>{doc["timestamp"]}': pred}}


def temporal_grounding_aggregate_activitynet_tvg(results, args):
    temporal_grounding_aggregate_submissions(results, args, "activitynet_tvg")


def temporal_grounding_aggregate_submissions(results, args, task):
    now_date_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    submission_file_name = f"inference_results_temporal_grounding_{task}_{now_date_time}.json"
    path = file_utils.generate_submission_file(submission_file_name, args)

    # results is a list of 5031 dict,
    # need to convert results into a single dict with 5031 key-value pairs
    combined_submission = {}

    for submission_dict in results:
        combined_submission.update(submission_dict)

    with open(path, "w") as f:
        json.dump(combined_submission, f, indent=4)

    eval_logger.info(f"Submission file saved to {path}")
