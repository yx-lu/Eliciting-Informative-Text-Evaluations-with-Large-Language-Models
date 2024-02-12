import os
import openai
import csv
import random
import json
import argparse
import unicodedata


def to_ascii(input_str):  # change the utf-8 characters to ascii
    normalized = unicodedata.normalize('NFKD', input_str)
    normalized = normalized.replace('"', "'")  # always use ' instead of " to avoid error in json
    ascii_str = ''.join(c for c in normalized if c.isascii())
    return ascii_str


random.seed(19260817)
parser = argparse.ArgumentParser()
parser.add_argument("--topic", type=str, default='')
parser.add_argument("--max_count", type=int, default=10**9)
parser.add_argument("--input_paper", type=str, default='./ICLR2020paper_full.csv')
parser.add_argument("--input_review", type=str, default='./ICLR2020review_full.csv')
parser.add_argument("--dataset_prefix", type=str, default="openreview")
parser.add_argument("--dataset_name", type=str, required=True)
args = parser.parse_args()

# read businesses in full dataset, and then rename "uid" key to "id"
with open(args.input_paper, mode="r", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    papers = [row for row in reader]
for paper in papers:
    paper["id"] = paper["uid"]  # rename "uid" -> "id"
    paper.pop("uid", None)

# read reviews in full dataset, and then rename "text" key to "review" and rename "business_id" key to "belong_id"
with open(args.input_review, mode="r", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    reviews = [row for row in reader]
for review in reviews:
    review["belong_id"] = review["paper_uid"]  # rename "business_id" -> "belong_id"
    review.pop("paper_uid", None)

paper_selected = []
has = {}
for paper in papers:

    def contain_str(source, targets):
        # return whether any element in targets is a substring of source
        ## target is seperated by comma, e.g. "Food,Restaurants"
        ## use underline to represent space
        _source = source.lower()
        _targets = targets.lower().replace(" ", "").replace("_", " ").split(",")
        for _target in _targets:
            if _target in _source:
                return True
        return False

    def legal(paper):
        return contain_str(paper['title'] + paper['keywords'], args.topic)

    if legal(paper):
        paper["title"] = to_ascii(paper["title"])
        paper["keywords"] = to_ascii(paper["keywords"])
        paper["abstract"] = to_ascii(paper["abstract"])
        paper_selected.append(paper)
        has[paper["id"]] = []

random.shuffle(paper_selected)
if len(paper_selected) > args.max_count:
    paper_selected = paper_selected[:args.max_count]

print("# items =", len(paper_selected))
with open("%s_%s_%s.json" % (args.dataset_prefix, "item", args.dataset_name), "w", encoding="utf-8") as json_file:
    json.dump(paper_selected, json_file)

for i, review in enumerate(reviews):
    if review["belong_id"] in has:
        has[review["belong_id"]].append(i)
review_selected = []
for paper in paper_selected:
    for i in has[paper["id"]]:
        reviews[i]["review"] = to_ascii(reviews[i]["review"])
        review_selected.append(reviews[i])

print("# reviews =", len(review_selected))
with open("%s_%s_%s.json" % (args.dataset_prefix, "review", args.dataset_name), "w", encoding="utf-8") as json_file:
    json.dump(review_selected, json_file)