from utils import *
import re
import math
import random
import argparse
import subprocess
import h5py
import torch
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from scipy import stats

random.seed(19260817)  # fix random seed
parser = argparse.ArgumentParser()

# which context, see folder '../openreview/' for openreview and '../yelp/' for yelp
parser.add_argument("--context", type=str, required=True)
# which dataset, use 'rand' for openreview and 'restaurant' for yelp.
parser.add_argument("--dataset", type=str, required=True)
# which model, 'gpt4' always works, try 'gpt35' for cheaper api call
parser.add_argument("--model", type=str, required=True)
# which prompt, always use 'bullet'
parser.add_argument("--prompt", type=str, required=True)
# which task, for experiment: 'experiment_same_diff_item' / 'experiment_one_side_degrade' / 'experiment_llm_gen_review' / 'experiment_bad_llm_review'
parser.add_argument("--task", type=str, required=True)
# whether conditioning on catagory / paper
parser.add_argument("--conditional", action='store_true')
parser.add_argument("--mod8", type=int, default=-1)
parser.add_argument("--test", action='store_true')
args = parser.parse_args()

with open("../%s/%s_item_%s.json" % (args.context, args.context, args.dataset), "r") as json_file:
    items = json.load(json_file)
with open("../%s/%s_review_%s.json" % (args.context, args.context, args.dataset), "r") as json_file:
    reviews = json.load(json_file)
has = {item["id"]: [] for item in items}
for i, review in enumerate(reviews):
    has[review["belong_id"]].append(i)
cond_material = {item["id"]: item["abstract" if args.context == "openreview" else "categories"] for item in items}

with open("./prompts/%s/%s.json" % (args.context, args.prompt), "r") as json_file:
    prompts = json.load(json_file)
logs_path = "./logs/%s/%s__%s__%s__%s%s%s/" % ("direct", args.context, args.dataset, args.model, args.prompt,
                                               "__cond" if args.conditional == True else "", "__test" if args.test == True else "")
if not os.path.exists(logs_path):
    os.makedirs(logs_path)


def get_review_summary(review):
    if args.context == 'yelp':
        return simple_call_api("", prompts["review_preprocess_1"] + "\n\n[Review begins]\n\n" + review["review"], args.model)
    elif args.context == 'openreview':
        return simple_call_api(prompts["review_preprocess_1"], review["review"], args.model)
    else:
        assert (0)


def parse_review_summary(summary):
    parsed_review = []
    for line in summary.split('\n'):
        line = line.strip()
        if line.startswith('- '): line = line[2:]  # remove '- ' at the beginning
        line = line.replace('"', "'")  # change " to '
        line = line.replace('\\', '')  # remove \
        if line == '': continue
        if not line.startswith('The reviewer'): continue
        parsed_review.append(line)
    return '\n'.join(parsed_review)


def remove_half_lines(summary):
    parsed_review = summary.split('\n')
    parsed_review = [parsed_review[i] for i in range(len(parsed_review)) if i % 2 == 0]
    return '\n'.join(parsed_review)


def find_assistant_indexes(li_token):
    '''
    # Find the indexes of assistant feedbacks
    ## Output: list, a sorted list of indexes 
    li_token: list, a list of tokens
    '''
    ret = []
    inside = False
    for i in range(1, len(li_token)):  # the last
        if i >= 4 and li_token[i - 4] == '▁[' and li_token[i - 3] == '/' and li_token[i - 2] == 'INST' and li_token[i - 1] == ']':
            inside = False
        if i + 2 < len(li_token) and li_token[i] == '▁[' and li_token[i + 1] == 'INST' and li_token[i + 2] == ']':
            inside = True
        if not inside:
            ret.append(i)
    return ret


def get_conditional_logprob(abstract, review1, review2, use_cache=True):
    '''
    # get the logprob of review2 conditioning on paper and review1
    ## paper is none -> not conditioning on paper
    ## review1 is none -> not conditioning on review1
    ## review1 and review2 should NOT start from '\u2581'
    paper: str(none), the paper text
    review1: str(none), the review1 text
    review2: str, the review2 text
    use_cache: bool, whether trying to load the output from cache 
    '''

    if abstract is None: abstract = "Not given"
    if review1 is None: review1 = "Not given"
    system = prompts["gen_review_system"]
    user = prompts["gen_review_user"] % (abstract, review1)
    assistant = "\u2581" + review2

    chat = [{
        "role": "system",
        "content": system,
    }, {
        "role": "user",
        "content": user,
    }, {
        "role": "assistant",
        "content": assistant,
    }, {
        "role": "user",
        "content": "",
    }]

    url = "http://127.0.0.1:55055/get-log-prob"  # You should luanch your own Llama-2 server according to the folder ``server''
    params = {"text": json.dumps(chat)}
    cache_key = generate_cache_key({"model": "Llama-2-4bit", "params": params})

    response = fetch_from_cache(cache_key) if use_cache else None
    if response is None:
        response = requests.get(url, params=params, proxies={"http": None, "https": None})
        save_to_cache(cache_key, response)

    res = torch.load(io.BytesIO(response.content))

    li_token = res["tokenized_tokens"]
    logprobs = [0.] + res["output_log_probs"].tolist()
    indexes = find_assistant_indexes(li_token)  # indexes of assistant's output
    sum_logprob = 0.  # the sum of all tokens' logprobs
    for index in indexes:
        sum_logprob += logprobs[index]

    return sum_logprob


def get_pridiction_score(abstract, review1, review2, use_cache=True):
    '''
    # get the pridiction score of review2 conditioning on paper and review1
    ## paper is none -> not conditioning on paper
    ## review1 is none -> not conditioning on review1
    paper: str(none), the paper text
    review1: str(none), the review1 text
    review2: str, the review2 text
    use_cache: bool, whether trying to load the output from cache 
    '''

    if abstract is None: abstract = "Not given"
    if review1 is None: review1 = "Not given"
    lines = review2.split('\n')
    numbered_lines = [f"[{i+1}] {line}" for i, line in enumerate(lines)]
    modified_review2 = '\n'.join(numbered_lines)

    system = prompts["predict_review_system"]
    user = prompts["predict_review_user"] % (abstract, review1, modified_review2)

    response = simple_call_api(system, user, model="llama2" if args.prompt == 'predictionllama' else "gpt4", use_additional_key=args.mod8)
    lines = response.split('\n')
    sum = 0.
    for line in lines:
        if line == '': continue
        found = False
        for score in range(-3, 4):
            for target in [
                    "SCORE=%d" % (score),
                    "Score=%d" % (score),
                    "SCORE= %d" % (score),
                    "Score= %d" % (score),
                    "SCORE = %d" % (score),
                    "Score = %d" % (score),
                    "SCORE: %d" % (score),
                    "Score: %d" % (score)
            ]:
                if target in line:
                    if not found:
                        sum += score
                        found = True

    return sum


def experiment(experiment_id):
    def gen_two_pos(l):  # generate two positions in [0,l)
        assert (l > 1)
        pos1 = random.randint(0, l - 1)
        pos2 = random.randint(0, l - 2)
        if pos2 >= pos1: pos2 += 1
        return (pos1, pos2)

    if args.test == True:
        num_samples = 200
    elif args.context == "openreview" and (experiment_id == "experiment_llm_gen_review" or experiment_id == "experiment_bad_llm_review"):
        num_samples = 500
    else:
        num_samples = 500

    # [[ generate review indices]]

    random.seed(19260817)  # reset seed
    random_review_samples = []
    for t in range(num_samples):
        # review1 and review2 are of the same item, review1 and review3 are of different items
        item1_pos, item2_pos = gen_two_pos(len(items))
        item1_id, item2_id = items[item1_pos]["id"], items[item2_pos]["id"]
        review1_pos, review2_pos = gen_two_pos(len(has[item1_id]))
        review3_pos = gen_two_pos(len(has[item2_id]))[0]
        review1 = reviews[has[item1_id][review1_pos]]
        review2 = reviews[has[item1_id][review2_pos]]
        review3 = reviews[has[item2_id][review3_pos]]
        random_review_samples.append((review1, review2, review3))

    # [[ load the LLM answer in cache ]]

    multi_threading = 30  # the number of threads
    threads = []
    for t in range(num_samples * 3):
        review = random_review_samples[t // 3][t % 3]
        threads.append(threading.Thread(target=get_review_summary, args=(review, )))
        threads[-1].start()

        if t % multi_threading == multi_threading - 1:
            for thread in threads:
                thread.join()
            threads = []
            print("===%d===" % (t))
    for thread in threads:
        thread.join()
    threads = []

    # [[build the query dataset]]

    ret = []
    sum_better = 0
    for t in range(num_samples):
        if t % 8 != args.mod8 and args.mod8 != -1: continue
        if t % 1 == 0:
            print("===%d===" % (t))
        # review1 and review2 are of the same item, review1 and review3 are of different items
        review1, review2, review3 = random_review_samples[t]

        if experiment_id == 'experiment_same_diff_item':
            # Experiment 1: signals of same or distinct items
            ## review1 and review2 are from the same paper
            ## review3 is from different paper
            review1_summary = parse_review_summary(get_review_summary(review1))
            review2_summary = parse_review_summary(get_review_summary(review2))
            review3_summary = parse_review_summary(get_review_summary(review3))
        elif experiment_id == 'experiment_oneside_degrade_same':
            # Experiment 2: one-sided degradation (same paper)
            ## review1 and review2 are from the same paper
            ## review3 is the degraded version of review2
            review1_summary = parse_review_summary(get_review_summary(review1))
            review2_summary = parse_review_summary(get_review_summary(review2))
            review3_summary = remove_half_lines(review2_summary)
        elif experiment_id == 'experiment_llm_gen_review' or 'experiment_bad_llm_review':
            review1_summary = parse_review_summary(get_review_summary(review1))
            review2_summary = parse_review_summary(get_review_summary(review2))
            with open("../pdf_management/files/%s.txt" % (review2["belong_id"]), "r", encoding="utf-8") as f:
                paper2_content = f.read()
            with open("./prompts/%s/%s.json" % (args.context, "llmreview"), "r") as json_file:
                prompts_gen = json.load(json_file)
            reviewllm = simple_call_api(prompts_gen["llm_gen_review_new"], paper2_content,
                                        "gpt4" if experiment_id == 'experiment_llm_gen_review' else "gpt35")
            review3_summary = parse_review_summary(get_review_summary({"review": reviewllm}))
        else:
            assert (0)

        cond1 = cond_material[review1["belong_id"]] if args.conditional == True else None

        if args.prompt == "prediction" or args.prompt == 'predictionllama':
            pmi21 = get_pridiction_score(cond1, review2_summary, review1_summary)
            pmi31 = get_pridiction_score(cond1, review3_summary, review1_summary)
            ret.append((pmi21, pmi31))
            print("t = %d (%.3f, %.3f)" % (t, pmi21, pmi31))

        else:
            logprob1 = get_conditional_logprob(cond1, None, review1_summary)
            logprob21 = get_conditional_logprob(cond1, review2_summary, review1_summary)
            logprob31 = get_conditional_logprob(cond1, review3_summary, review1_summary)
            ret.append((logprob21 - logprob1, logprob31 - logprob1))
            print("t = %d (%.3f, %.3f, %.3f)" % (t, logprob1, logprob21, logprob31))

    if args.mod8 != -1: return

    with open(logs_path + experiment_id + ".json", 'w') as json_file:
        json.dump(ret, json_file)


if __name__ == "__main__":

    if args.task == "experiment_same_diff_item":
        experiment('experiment_same_diff_item')
    elif args.task == "experiment_oneside_degrade_same":
        experiment('experiment_oneside_degrade_same')
    elif args.task == "experiment_llm_gen_review":
        experiment('experiment_llm_gen_review')
    elif args.task == "experiment_bad_llm_review":
        experiment('experiment_bad_llm_review')
    elif args.task == "all":
        experiment('experiment_same_diff_item')
        experiment('experiment_oneside_degrade_same')
        if args.context == "openreview":
            experiment('experiment_llm_gen_review')
            experiment('experiment_bad_llm_review')