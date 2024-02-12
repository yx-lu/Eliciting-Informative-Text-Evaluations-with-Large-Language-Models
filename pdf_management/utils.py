import os
import sys
import openai
import csv
import numpy as np
import tiktoken
import json
import hashlib
import requests
import argparse
# import torch
import io
import pickle
import unicodedata
import threading

openai_client = openai.OpenAI(api_key="Your Key")
openai_chat = openai_client.chat.completions
openai_compl = openai_client.completions

openrouter_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="Your Key",
)
openrouter_chat = openrouter_client.chat.completions

# enc = tiktoken.get_encoding("cl100k_base")
cache_lock = threading.Lock()  # lock to keep the cache operations safe


def generate_cache_key(params):
    '''
    # generate a unique cache key based on the request parameters
    ## md5 seems to be enough
    params: dict, request parameters
    '''
    params_string = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_string.encode('utf-8')).hexdigest()


def fetch_from_cache(cache_key, cache_dir="./cache"):
    '''
    # try to fetch the response from cache
    ## thread safe
    ## return none if not find
    cache_key: str, md5 fingerprint
    cache_dir: str, the folder for cache
    '''
    with cache_lock:  # lock the cache lock
        cache_file = os.path.join(cache_dir, cache_key)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as file:
                return pickle.load(file)
        return None


def save_to_cache(cache_key, response, cache_dir="./cache", overwrite=True):
    '''
    # save a response to the cache
    ## thread safe
    cache_key: str, md5 fingerprint
    response: anything, just the response
    cache_dir: str, the folder for cache
    overwrite: whether to overwrite the cache
    '''
    with cache_lock:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_file = os.path.join(cache_dir, cache_key)

        if os.path.exists(cache_file):
            if not overwrite:
                return None

        with open(cache_file, 'wb') as file:
            pickle.dump(response, file)
        return response


def call_api(compl, params, use_cache=True):  #
    '''
    # call the api with given parameters
    ## thread safe
    compl: str, 'gpt_compl' / 'gpt_chat' / 'llama_chat'
    params: dict, the parameters
    use_cache: bool, true if try to load the output in cache 
    '''
    cache_key = generate_cache_key(params)

    if use_cache:
        cached_response = fetch_from_cache(cache_key)
        if cached_response is not None:
            return cached_response

    response = compl.create(**params)

    save_to_cache(cache_key, response)

    return response


def simple_call_api(system_text, user_text, model, max_tokens=2000, temperature=0, use_cache=True):
    '''
    # simplified api call
    ## thread safe
    system_text: str, system prompt
    user_text: str, user prompt
    model: str, 'gpt4'/'gpt35'/'llama2'/'local' (do not use 'local')
    max_token: int, maximum amout of output tokens
    use_cache: bool, true if try to load the output in cache 
    '''
    api_option = "openrouter"  # openrouter is always better if you do not need logprob

    if api_option == "openrouter":
        compl = openrouter_chat  # choose openrouter_chat or openai_chat!
        model_names = {
            "gpt4": "openai/gpt-4-1106-preview",
            "gpt35": "openai/gpt-3.5-turbo-1106",
            "llama2": "meta-llama/llama-2-70b-chat",
            "local": "local"
        }
    elif api_option == "openai":
        compl = openai_chat
        model_names = {"gpt4": "gpt-4-1106-preview", "gpt35": "gpt-3.5-turbo-1106", "local": "local"}
    model = model_names[model]

    messages = [{
        "role": "system",
        "content": system_text,
    }, {
        "role": "user",
        "content": user_text,
    }] if system_text != "" else [{
        "role": "user",
        "content": user_text,
    }]

    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    }

    response = call_api(compl, params, use_cache)
    if response.choices is None:  # an error occurs (with high probability, the error is caused by censorship of openai)
        error_code = response.error["code"]
        print("Error occurs (%d)" % (error_code))
        print("Cache key:", generate_cache_key(params))
        # assert (0)
        if error_code != 403:  # censorship
            response = call_api(compl, params, use_cache=False)
            print("Try again!")
        if response.choices is None:
            params["model"] = model_names["llama2"]
            response = call_api(compl, params, use_cache)
            print("Using alternative model: Llama 2")
        else:
            print("Success")
    return response.choices[0].message.content


def read_csv(input_filename):
    """
    # read anything from csv files
    ## return a dict
    dataset: str, name of the dataset
    """
    ret = []
    with open(input_filename, mode="r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        ret = [row for row in reader]
    return ret


def read_reviews(dataset):
    '''
    # read the OpenReview data from csv files (for reviews)
    ## return a dict
    dataset: str, name of the dataset
    '''
    # !!! check the relative location of 'utils.py' and the folder 'openreview/'
    input_filename = "../openreview/ICLR2020review_" + dataset + ".csv"
    reviews = read_csv(input_filename)

    def to_ascii(input_str):  # change the utf-8 characters to ascii
        normalized = unicodedata.normalize('NFKD', input_str)
        ascii_str = ''.join(c for c in normalized if c.isascii())
        return ascii_str

    for review in reviews:
        # always use ' instead of " to avoid error in json
        review["review"] = to_ascii(review["review"]).replace('"', "'")
    return reviews


def read_papers(dataset):
    '''
    # read the OpenReview data from csv files (for papers)
    ## return a dict
    dataset: str, name of the dataset
    '''
    # !!! check the relative location of 'utils.py' and the folder 'openreview/'
    input_filename = "../openreview/ICLR2020paper_" + dataset + ".csv"
    reviews = read_csv(input_filename)

    def to_ascii(input_str):  # change the utf-8 characters to ascii
        normalized = unicodedata.normalize('NFKD', input_str)
        ascii_str = ''.join(c for c in normalized if c.isascii())
        return ascii_str

    for review in reviews:
        # always use ' instead of " to avoid error in json
        review["title"] = to_ascii(review["title"]).replace('"', "'")
        review["abstract"] = to_ascii(review["abstract"]).replace('"', "'")
    return reviews