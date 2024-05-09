# Eliciting Informative Text Evaluations with Large Language Models
This is the repository for the paper "Eliciting Informative Text Evaluations with Large Language Models". This repository includes the code we use and the experimental results. Here is the guideline of how to rerun the experiment and reproduce our results.

## Step 1: Build the ICLR and Yelp Dataset

**If you do not want to create the dataset by yourself, that's fine. You can just skip this section, since all the dataset we generated is included in the repository.**

### ICLR2020 Data

The data of the ICLR conference is open to everyone and can be obtained by calling the Openreview API. 

Launch a terminal instance with the present working directory as `/openreview`, and then run `crawl.py` to obtain the raw data.

```bash
cd openreview
python3 crawl.py
```

This will create three files, `ICLR2020decision_full.csv`, `ICLR2020paper_full.csv`, and `ICLR2020review_full.csv`, which contains the raw data of ICLR2020.

Then, to build the sub-dataset we use in this project, run the following code to build a random dataset including 300 papers.

```bash
python3 gen_dataset.py --max_count 300 --dataset_name rand
```

This will create two files named `openreview_item_rand.json` and `openreview_review_rand.json` containing the preprocessed data.

Afterwards, we parse the PDFs of the 300 submissions in the dataset for further experiment. To do this, change the working directory to `/pdf_management`. In this project, we use ScienceBeam as the PDF parsing server.

⚠️⚠️⚠️ **ScienceBeam PDF parser only supports x86 Linux operating system. You can also run it in WSL if you use Windows.**

```bash
cd ../pdf_management
conda env create -f conda_environment.yml
conda activate ScienceBeam
python3 -m sciencebeam_parser.service.server --port=8080  # Make sure this is running in the background
```

(The solution and code of parsing PDFs is copied from https://github.com/zhang-yu-wei/ClusterLLM)

After start the PDF parsing server, run `pdf_management.py` to download and parse the PDFs.

```bash
python3 pdf_management.py
```

The original PDFs and the parsed results are saved in directory `/pdf_management/files`.

### Yelp Data

You can download the raw data of the Yelp dataset from https://www.yelp.com/dataset. Please check **YELP DATASET TERMS OF USE** for detailed information.

After download the dataset, you should copy all the files in the dataset to directory `\yelp`.

Then, to build the sub-dataset we use in this project, run the following code. This code will generate a dataset of 1000 restaurants, each of which contains at least 100 reviews.

```bash
cd yelp
python3 gen_dataset.py \
    --review_count 100 \
    --categories restaurant \
    --dataset_name restaurant \
    --max_count 1000 \
    --max_reviews 300
```

This will create two files named `yelp_item_restaurant` and `yelp_review_restaurant` containing the preprocessed data.

## Step 2: Start the Llama-2 server for Logprobs Feedback

To start the server for Logprobs feedback, you should run Llama-2 locally. You can download the model `./llama-2-70b-chat-hf` from `https://huggingface.co/meta-llama/Llama-2-70b-chat-hf`. After download the model, copy it to `/llmserver`.

Then create an environment for the Llama-2 server and start the server.

```bash
cd llmserver
conda env create -f llama_env.yaml
conda activate llama
python3 serve_log_prob.py
```

## Step 3: Run the Experiments

Before the experiments, you need to prepare an API key which can call the GPT-3.5 and GPT-4, and paste the API key into the `Your Key` space in `/experiments/utils.py`.

```python
openai_client = openai.OpenAI(api_key="Your Key")
```

If you cannot access OpenAI, you can use third-party APIs that support the OpenAI library as an alternative, such as OpenRouter.

```python
openrouter_client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="Your Key",
)
```

Up to now, you can start the experiments (as well as your own's). For the experiments appeared in the paper, you should run the following four.

```bash
cd experiments

python direct.py --context openreview --dataset rand --model gpt4 --prompt bullet --task all
python direct.py --context openreview --dataset rand --model gpt4 --prompt bullet --task all --conditional

python direct.py --context yelp --dataset restaurant --model gpt35 --prompt bullet --task all
python direct.py --context yelp --dataset restaurant --model gpt35 --prompt bullet --task all --conditional
```

Here is the explanation of the arguments of the code:

* `--context`: Select a context of the task. You can choose `openreview` for ICLR or `yelp` for Yelp.
* `--dataset`: Select a dataset of the selected context. We have generated the `rand` dataset for `openreview` and `restaurant` dataset for `yelp`. Feel free to generate and use other datasets.
* `--model`: Select a model to generate the review summary. You should choose `gpt4` for `gpt-4-1106-preview` or `gpt35` for `gpt-3.5-turbo-1106`. You can use other models as long as your API supports such models. We recommend `gpt4` for peer-review tasks (`openreview` context) and `gpt35` / `gpt4` for restaurant review tasks (`yelp` dataset).
* `--prompt`: Select a prompt design. We design a set of prompts called `bullet`. You can design other prompts.
* `--task`: There are four tasks in all experiments:
  * `experiment_same_diff_item`: Set `target = human-review-1 for paper-1`. Compare the score for `source = human-review-2 for paper-1`, and `source = human-review for paper-2`. We expect that the former will have a higher score.
  * `experiment_one_side_degrade`: Set `target = human-review-1`. Compare the score for `source = human-review-2`, and `source = human-review2 that removes half of the judgments`. We expect that the former will have a higher score.
  * `experiment_llm_gen_review`: Set `target = human-review-1`. Compare the score for `source = human-review-2`, and `source = gpt4-review`. We expect that the former will have a higher score.
  * `experiment_bad_llm_review`: Set `target = human-review-1`. Compare the score for `source = human-review-2`, and `source = gpt3.5-review`. We expect that the former will have a higher score.
* `--conditional`: set `--conditional` to let the experiment to test GSPPM using the abstract as the synopsis. Otherwise, the experiment tests GPPM (with no synopsis).

Here ends the experiments. You can see the results in the `/logs` folder. You can also run the `processlogs.sh` in `/logs` for figures.

```bash
cd logs
./processlogs.sh
```



