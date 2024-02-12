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
parser.add_argument("--state", type=str, default='')
parser.add_argument("--max_count", type=int, default=10**9)
parser.add_argument("--max_reviews", type=int, default=10**9)
parser.add_argument("--review_count", type=int, default=0)
parser.add_argument("--categories", type=str, default='')
parser.add_argument("--input_business", type=str, default='./yelp_academic_dataset_business.json')
parser.add_argument("--input_review", type=str, default='./yelp_academic_dataset_review.json')
parser.add_argument("--dataset_prefix", type=str, default="yelp")
parser.add_argument("--dataset_name", type=str, required=True)
args = parser.parse_args()

print(args.max_count)

# read businesses in full dataset, and then rename "business_id" key to "id"
with open(args.input_business, "r", encoding="utf-8") as json_file:
    businesses = [json.loads(l) for l in json_file]
for business in businesses:
    business["id"] = business["business_id"]  # rename "business_id" -> "id"
    business.pop("business_id", None)

# read reviews in full dataset, and then rename "text" key to "review" and rename "business_id" key to "belong_id"
with open(args.input_review, "r", encoding="utf-8") as json_file:
    reviews = [json.loads(l) for l in json_file]
for review in reviews:
    review["review"] = review["text"]  # rename "text" -> "review"
    review.pop("text", None)
    review["belong_id"] = review["business_id"]  # rename "business_id" -> "belong_id"
    review.pop("business_id", None)

business_selected = []
has = {}
for business in businesses:

    def contain_str(source, targets):
        # return whether any element in targets is a substring of source
        ## target is seperated by comma, e.g. "Food,Restaurants"
        _source = source.lower()
        _targets = targets.lower().replace(" ", "").split(",")
        for _target in _targets:
            if _target in _source:
                return True
        return False

    def legal(business):
        if "state" not in business or "review_count" not in business or "categories" not in business:
            return False
        if type(business["state"]) is not str or type(business["review_count"]) is not int or type(business["categories"]) is not str:
            return False

        return contain_str(business["state"],args.state) and business["review_count"] >= args.review_count\
            and contain_str(business["categories"],args.categories)

    if legal(business):
        business["categories"] = to_ascii(business["categories"])
        business_selected.append(business)
        has[business["id"]] = []

random.shuffle(business_selected)
if len(business_selected) > args.max_count:
    business_selected = business_selected[:args.max_count]
print("# items =", len(business_selected))
with open("%s_%s_%s.json" % (args.dataset_prefix, "item", args.dataset_name), "w", encoding="utf-8") as json_file:
    json.dump(business_selected, json_file)

for i, review in enumerate(reviews):
    if review["belong_id"] in has:
        has[review["belong_id"]].append(i)
for business in business_selected:
    random.shuffle(has[business["id"]])
    if len(has[business["id"]]) > args.max_reviews:
        has[business["id"]] = has[business["id"]][:args.max_reviews]

review_selected = []
for business in business_selected:
    for i in has[business["id"]]:
        if len(reviews[i]["review"]) > 100:
            reviews[i]["review"] = to_ascii(reviews[i]["review"])
            review_selected.append(reviews[i])

print("# reviews =", len(review_selected))
with open("%s_%s_%s.json" % (args.dataset_prefix, "review", args.dataset_name), "w", encoding="utf-8") as json_file:
    json.dump(review_selected, json_file)