# import time
import json
# import math
import argparse
# import Levenshtein
import logging
# import openai
# import traceback
# import os
# import pickle
import io
import torch
from typing import Optional, Any
from llama_2_wrapper import Llama2Wrapper
# from my_utils.my_utils import PredictionRes, print_namedtuple
# from langchain.utilities import WikipediaAPIWrapper
# from tools_config import tools_config
# from keys import openai_key
from flask import Flask, send_file
from flask import request
# from collections import namedtuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

app = Flask(__name__)
logger = logging.getLogger(__name__)


def setup_logger(logger: logging.Logger, debug_mode: bool, log_file: Optional[str]):
    if debug_mode:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    formatter = logging.Formatter("--------%(asctime)s-%(name)s[line:%(lineno)d]-%(levelname)s--------\n%(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)


@app.route("/get-log-prob")
@torch.no_grad()
def get_log_prob():
    text: str = request.args.get("text")
    if args.model == "llama2-70b-chat":
        text = json.loads(text)
        logger.info("Function: get_log_prob")
        logger.info("Text: {}".format(text))
        res = llama2_wrapper.chat_completion(
            [text],
            max_gen_len=1024,  # no use
            temperature=0.0,  # no use
            top_p=1.0,  # no use
            return_prob=True)
        logger.info("Token Length: {}".format(len(res["tokenized_tokens"])))
    else:
        tokenized_res = tokenizer(text, return_tensors="pt")
        input_ids = tokenized_res["input_ids"][0]
        tokenized_tokens = tokenizer.convert_ids_to_tokens(tokenized_res["input_ids"][0])
        output_log_probs = model(**tokenized_res).logits[0].cpu()
        output_log_probs = torch.log(torch.softmax(output_log_probs, dim=-1))
        input_ids = input_ids[1:]
        output_log_probs = output_log_probs[range(len(input_ids)), input_ids]
        res = {
            "tokenized_tokens": tokenized_tokens,
            "output_log_probs": output_log_probs,
        }
    to_send = io.BytesIO()
    torch.save(
        res,
        to_send,
    )
    to_send.seek(0)
    logger.info(torch.cuda.memory_summary(device="cuda:0"))
    return send_file(to_send, mimetype="application/octet-stream")


@app.route("/generate")
@torch.no_grad()
def generate():
    temperature: float = float(request.args.get("tempreture", 0.0))
    max_tokens: int = int(request.args.get("max_tokens", 1024))
    top_p: float = float(request.args.get("top_p", 0.9))
    text = request.args.get("text")
    # generated_results = text_generation_pipeline(
    #     text,
    #     temperature=temperature,
    #     top_p=top_p,
    #     max_new_tokens=max_tokens,
    #     return_full_text=False,
    #     eos_token_id=tokenizer.convert_tokens_to_ids("\n")
    # )
    if args.model == "llama2-70b-chat":
        text = json.loads(text)
        logger.info("Function: generate")
        logger.info("Text: {}".format(text))
        logger.info(torch.cuda.memory_summary(device="cuda:0"))
        return llama2_wrapper.chat_completion(
            [text],
            max_gen_len=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    else:
        generated_results = text_generation_pipeline(
            text,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
            return_full_text=False,
        )
        generated_text = generated_results[0]["generated_text"]
        return generated_text


parser = argparse.ArgumentParser()
parser.add_argument("--log-file", default="serve_logs/log.txt")
parser.add_argument("--port", type=int, default=55055)
parser.add_argument("--model", type=str, default="llama2-70b-chat")
args = parser.parse_args()
setup_logger(logger, True, args.log_file)
if args.model == "llama2-70b-chat":
    llama2_wrapper = Llama2Wrapper(
        "../llama-2-70b-chat-hf",
        is_chat_model=True,
        debug_mode=True,
        load_4bit=True,
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir="./hf-models-cache/",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir="./hf-models-cache/",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

app.run(port=args.port, debug=True, host="0.0.0.0", use_reloader=False, threaded=False)
