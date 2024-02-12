from utils import *
import pdfparser

import argparse
import threading
import time
import sys
import pingouin as pg
import pandas as pd

parser = argparse.ArgumentParser()
# which context, see folder '../openreview/' for openreview and '../yelp/' for yelp
parser.add_argument("--context", type=str, default="openreview")
# which dataset, use 'rl' for openreview and 'pizza' for yelp.
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--tmp", type=int, default=0)
args = parser.parse_args()

with open("./prompt.json", 'r') as json_file:
    prompts = json.load(json_file)
with open("../%s/%s_item_%s.json" % (args.context, args.context, args.dataset), "r") as json_file:
    papers = json.load(json_file)
with open("../%s/%s_review_%s.json" % (args.context, args.context, args.dataset), "r") as json_file:
    reviews = json.load(json_file)

if __name__ == "__main__":

    for i, paper in enumerate(papers):

        path_pdf = "./files/" + paper["id"] + ".pdf"
        path_xml = "./files/" + paper["id"] + ".xml"
        path_txt = "./files/" + paper["id"] + ".txt"

        print(i)

        if not os.path.exists(path_pdf):
            url = "https://openreview.net" + paper["pdf"]
            print(url, path_pdf)
            response = requests.get(url, proxies={"http": None, "https": None})

            if response.status_code == 200:
                with open(path_pdf, "wb") as file:
                    file.write(response.content)
            else:
                print("Failed to download the file. Status code:", response.status_code)
                exit(0)

        if not os.path.exists(path_txt):
            pdfparser.parse_to_text(path_pdf, path_txt)
