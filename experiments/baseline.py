import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_curve, auc
import random
import argparse
import json
import os

random.seed(19260817)  # fix random seed
parser = argparse.ArgumentParser()

# which context, see folder '../openreview/' for openreview and '../yelp/' for yelp
parser.add_argument("--context", type=str, required=True)
# which dataset, use 'rl' for openreview and 'pizza' for yelp.
parser.add_argument("--dataset", type=str, required=True)
args = parser.parse_args()
experiment_id = "experiment_same_diff_item"

with open("../%s/%s_item_%s.json" % (args.context, args.context, args.dataset), "r") as json_file:
    items = json.load(json_file)
with open("../%s/%s_review_%s.json" % (args.context, args.context, args.dataset), "r") as json_file:
    reviews = json.load(json_file)

has = {item["id"]: [] for item in items}
for i, review in enumerate(reviews):
    has[review["belong_id"]].append(i)

logs_path = "./logs/baseline/%s__%s/" % (args.context, args.dataset)
if not os.path.exists(logs_path):
    os.makedirs(logs_path)

score_mapping = {
    "openreview": {
        '1: Reject': 0,
        '3: Weak Reject': 1,
        '6: Weak Accept': 2,
        '8: Accept': 3
    },
    "yelp": {
        1.0: 0,
        2.0: 1,
        3.0: 2,
        4.0: 3,
        5.0: 4
    }
}

# generate data


def gen_two_pos(l):  # generate two positions in [0,l)
    assert (l > 1)
    pos1 = random.randint(0, l - 1)
    pos2 = random.randint(0, l - 2)
    if pos2 >= pos1: pos2 += 1
    return (pos1, pos2)


def get_score(review):
    rating = review["rating"] if args.context == "openreview" else review["stars"]
    return score_mapping[args.context][rating]


joint = [[0 for j in range(5)] for i in range(5)]
marg = [0 for i in range(5)]
max_peer = 3 if args.context == "openreview" else 100

for li in has.values():
    for i in range(min(len(li), max_peer)):
        review1 = reviews[li[i]]
        score1 = get_score(review1)
        marg[score1] += 1
        cnt_peer = 0
        for j in range(i + 1, min(len(li), max_peer)):
            review2 = reviews[li[j]]
            score2 = get_score(review2)
            joint[score1][score2] += 1
            joint[score2][score1] += 1

su = sum(marg)
for i in range(len(marg)):
    marg[i] /= su
su = sum(sum(row) for row in joint)
for i in range(len(marg)):
    for j in range(len(marg)):
        joint[i][j] /= su

# [[ generate review indices]]

num_samples = 1000
random.seed(19260817)  # reset seed
ret = []
for t in range(num_samples):
    # review1 and review2 are of the same item, review1 and review3 are of different items
    item1_pos, item2_pos = gen_two_pos(len(items))
    item1_id, item2_id = items[item1_pos]["id"], items[item2_pos]["id"]
    review1_pos, review2_pos = gen_two_pos(len(has[item1_id]))
    review3_pos = gen_two_pos(len(has[item2_id]))[0]
    review1 = reviews[has[item1_id][review1_pos]]
    review2 = reviews[has[item1_id][review2_pos]]
    review3 = reviews[has[item2_id][review3_pos]]
    score1 = get_score(review1)
    score2 = get_score(review2)
    score3 = get_score(review3)
    logprob21 = np.log(joint[score1][score2]) - np.log(marg[score2])
    logprob31 = np.log(joint[score1][score3]) - np.log(marg[score3])
    ret.append((logprob21, logprob31))

with open(logs_path + experiment_id + ".json", 'w') as json_file:
    json.dump(ret, json_file)
