from utils import *
import re
import math
import random
import argparse
import subprocess
import h5py
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from scipy import stats

random.seed(19260817)  # fix random seed
parser = argparse.ArgumentParser()

# which context, see folder '../openreview/' for openreview and '../yelp/' for yelp
parser.add_argument("--context", type=str, required=True)
# which dataset, use 'rl' for openreview and 'pizza' for yelp.
parser.add_argument("--dataset", type=str, required=True)
# which model, 'gpt4' always works, try 'gpt35' for cheaper api call
parser.add_argument("--model", type=str, required=True)
# which prompt, always use 'bullet'
parser.add_argument("--prompt", type=str, required=True)
# which task, for experiment: 'experiment_same_diff_item' or 'experiment_one_side_degrade', and for others 'train', etc
parser.add_argument("--task", type=str, required=True)
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

with open("./prompts/%s/%s.json" % (args.context, args.prompt), "r") as json_file:
    prompts = json.load(json_file)
logs_path = "./logs/clustering/%s__%s__%s__%s/" % (args.context, args.dataset, args.model, args.prompt)
if not os.path.exists(logs_path):
    os.makedirs(logs_path)


def get_review_summary(review):
    if args.context == 'yelp':
        return simple_call_api("", prompts["review_preprocess_1"] + "\n\n[Review begins]\n\n" + review["review"], args.model)
    elif args.context == 'openreview':
        tmp_str = simple_call_api(prompts["review_preprocess_1"], review["review"], args.model)
        return simple_call_api(prompts["review_preprocess_2"], tmp_str, args.model)
    else:
        assert (0)


def parse_review_summary(summary):
    '''
    Note the return value is different from the direct method
    '''
    parsed_review = []
    for line in summary.split('\n'):
        line = line.strip()
        if line.startswith('- '): line = line[2:]  # remove '- ' at the beginning
        line = line.replace('"', "'")  # change " to '
        line = line.replace('\\', '')  # remove \
        if line == '': continue
        if not line.startswith('The reviewer'): continue
        parsed_review.append(line)
    return parsed_review


def remove_half_lines(parsed_review):
    return [parsed_review[i] for i in range(len(parsed_review)) if i % 2 == 0]


def read_llm_generated_review(paper):
    with open("../pdf_management/files/%s.txt" % (paper["id"]), "r", encoding="utf-8") as f:
        return f.read()


def unique_in_order(iterable):
    seen = set()
    return [x for x in iterable if not (x in seen or seen.add(x))]


def init_embedding():
    # this task build the training set for the short text embedding
    ## the training set is the first review of each item

    # [[ build training set for text embedding]]
    if args.context == "yelp":
        # use the last two review of each item to build the training set for text embedding
        empirical_reviews = [reviews[has[item["id"]][-1]] for item in items] + [reviews[has[item["id"]][-2]] for item in items]
    else:
        # the number of data is too small, use all reviews to build the training set for text embedding
        empirical_reviews = reviews

    # [[ load the LLM answer in cache ]]
    multi_threading = 40  # the number of threads
    threads = []
    for i, review in enumerate(empirical_reviews):

        thread = threading.Thread(target=get_review_summary, args=(review, ))
        threads.append(thread)
        thread.start()
        if i % multi_threading == multi_threading - 1:
            for thread in threads:
                thread.join()
            threads = []
            print("===%d===" % (i))
    for thread in threads:
        thread.join()
    threads = []

    # [[ parse the review bullets ]]
    dataset = []
    for i, review in enumerate(empirical_reviews):
        summary = get_review_summary(review)
        parsed_review = parse_review_summary(summary)
        for line in parsed_review:
            dataset.append('{"task": "%s", "input": "%s"}' % (args.context, line))
        if i % 100 == 99:
            print("===%d===" % (i))

    dataset = unique_in_order(dataset)  # unique
    with open("./ClusterLLM-main/datasets/%s/empirical.jsonl" % (args.context), 'w', encoding='utf-8') as f:
        ouf = '\n'.join(dataset)
        print(ouf, file=f)

    checkpoint_exist = True

    if not checkpoint_exist:

        target_dir = "./ClusterLLM-main/perspective/2_finetune"
        command = "bash scripts/get_embedding_universal.sh %s %s %s" % (args.context, "empirical", "original")  # dataset,scale,checkpoint/original
        subprocess.run(command, shell=True, cwd=target_dir, text=True)

        target_dir = "./ClusterLLM-main/perspective/1_predict_triplet"
        command = "bash scripts/triplet_sampling_universal.sh %s %s %s" % (args.context, "empirical", "original")  # dataset,scale,checkpoint/original
        subprocess.run(command, shell=True, cwd=target_dir, text=True)

    else:
        target_dir = "./ClusterLLM-main/perspective/2_finetune"
        command = "bash scripts/get_embedding_universal.sh %s %s %s" % (args.context, "empirical", "checkpoint")  # dataset,scale,checkpoint/original
        subprocess.run(command, shell=True, cwd=target_dir, text=True)


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
        num_samples = 1000

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

    multi_threading = 50  # the number of threads
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

    query_dataset = []
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

        for parsed_review in [review1_summary, review2_summary, review3_summary]:
            for line in parsed_review:
                query_dataset.append('{"task": "%s", "input": "%s"}' % (args.context, line))

    if args.mod8 != -1: return

    query_dataset = unique_in_order(query_dataset)  # unique
    with open("./ClusterLLM-main/datasets/%s/query.jsonl" % (args.context), 'w', encoding='utf-8') as f:  #
        ouf = '\n'.join(query_dataset)
        print(ouf, file=f)

    # [[run the finetuned embeder to embed the query dataset]]

    target_dir = "./ClusterLLM-main/perspective/2_finetune"
    # ### add checkpoint after training
    # command = "bash scripts/get_embedding_universal.sh %s %s %s" % (args.context, "empirical", "checkpoint")  # dataset,scale,checkpoint/original
    # subprocess.run(command, shell=True, cwd=target_dir, text=True)
    command = "bash scripts/get_embedding_universal.sh %s %s %s" % (args.context, "query", "checkpoint")  # dataset,scale,checkpoint/original
    subprocess.run(command, shell=True, cwd=target_dir, text=True)

    # # [[calculate empirical frequency]]

    with h5py.File("./ClusterLLM-main/datasets/%s/empirical_checkpoint_embeds.hdf5" % (args.context), 'r') as f:
        empirical_embeds = np.asarray(f['embeds'])
    with h5py.File("./ClusterLLM-main/datasets/%s/query_checkpoint_embeds.hdf5" % (args.context), 'r') as f:
        query_embeds = np.asarray(f['embeds'])
    with open("./ClusterLLM-main/datasets/%s/empirical.jsonl" % (args.context), 'r') as f:
        empirical_dataset = [json.loads(l) for l in f]
    query_dataset = [json.loads(l) for l in query_dataset]
    print(len(empirical_dataset), len(query_dataset))

    clustering = MiniBatchKMeans(n_clusters=30, random_state=100).fit(empirical_embeds)
    preds = clustering.labels_
    n_clusters = len(set(preds))

    cluster_belong = {}
    for i in range(n_clusters):
        class_member_mask = preds == i
        class_member_inds = np.where(class_member_mask)[0]
        for index in list(class_member_inds):
            cluster_belong[empirical_dataset[index]['input']] = i

    joint0 = [0. for i in range(n_clusters)]
    joint1 = [0. for i in range(n_clusters)]
    if args.context == "yelp":
        # use the last two review of each item to build the training set for text embedding
        empirical_reviews = [reviews[has[item["id"]][-1]] for item in items] + [reviews[has[item["id"]][-2]] for item in items]
    else:
        # the number of data is too small, use all reviews to build the training set for text embedding
        empirical_reviews = reviews
    # for review in empirical_reviews:
    #     review_summary = parse_review_summary(get_review_summary(review))
    #     cluster_indices = list(set([cluster_belong[line] for line in review_summary]))
    #     for cluster_index in cluster_indices:
    #         frequency[cluster_index] += 1. / len(empirical_reviews)
    cnt_joint = 0
    for i in range(len(empirical_reviews)):
        review1 = empirical_reviews[i]
        review1_summary = parse_review_summary(get_review_summary(review1))
        indices1 = set([cluster_belong[line] for line in review1_summary])
        for j in range(i + 1, len(empirical_reviews)):
            review2 = empirical_reviews[j]
            if review1["belong_id"] == review2["belong_id"]:
                cnt_joint += 1
                review2_summary = parse_review_summary(get_review_summary(review2))
                indices2 = set([cluster_belong[line] for line in review2_summary])
                for ind in range(n_clusters):
                    joint0[ind] += int((ind not in indices1) and (ind not in indices2))
                    joint1[ind] += int((ind in indices1) and (ind in indices2))
    for ind in range(n_clusters):
        joint0[ind] /= cnt_joint
        joint1[ind] /= cnt_joint

    # [[finally, answer the query]]

    lines = [data["input"] for data in query_dataset]
    results = clustering.predict(query_embeds)
    cluster_belong = {line: results[i] for i, line in enumerate(lines)}  # reset cluster_belong to the query dataset

    ret = []
    for t in range(num_samples):
        if t % 100 == 99:
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

        cluster_indices1 = set([cluster_belong[line] for line in review1_summary])
        cluster_indices2 = set([cluster_belong[line] for line in review2_summary])
        cluster_indices3 = set([cluster_belong[line] for line in review3_summary])

        def calc_lsr(indices_source, indices_target):
            ret = 0.
            for i in range(n_clusters):
                joint = ((joint0[i], (1 - joint0[i] - joint1[i]) / 2), ((1 - joint0[i] - joint1[i]) / 2, joint1[i]))
                marg = (joint[0][0] + joint[0][1], joint[1][0] + joint[1][1])
                sig_source = int(i in indices_source)
                sig_target = int(i in indices_target)
                ret += np.log(joint[sig_source][sig_target]) - np.log(marg[sig_source])
            return ret

        lsr12 = calc_lsr(cluster_indices2, cluster_indices1)
        lsr13 = calc_lsr(cluster_indices3, cluster_indices1)

        ret.append((lsr12, lsr13))

    print("lsr12:", np.mean([x[0] for x in ret]), np.std([x[0] for x in ret]))
    print("lsr13:", np.mean([x[1] for x in ret]), np.std([x[1] for x in ret]))
    print("delta:", np.mean([x[0] - x[1] for x in ret]), np.std([x[0] - x[1] for x in ret]))
    with open(logs_path + experiment_id + ".json", 'w') as json_file:
        json.dump(ret, json_file)


if __name__ == "__main__":

    if args.task == "init_embedding":
        init_embedding()
    elif args.task == "experiment_same_diff_item":
        experiment('experiment_same_diff_item')
    elif args.task == "experiment_oneside_degrade_same":
        experiment('experiment_oneside_degrade_same')
    elif args.task == "experiment_llm_gen_review":
        experiment('experiment_llm_gen_review')
    elif args.task == "all":
        init_embedding()
        experiment('experiment_same_diff_item')
        experiment('experiment_oneside_degrade_same')
        if args.context == "openreview":
            experiment('experiment_llm_gen_review')
            experiment('experiment_bad_llm_review')