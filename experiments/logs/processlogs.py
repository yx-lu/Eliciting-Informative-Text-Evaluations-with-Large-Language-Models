import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import json
import csv
import numpy as np
import random
import os
import argparse
from scipy import stats


def plot_roc_curve(title, li_posi, li_nega):

    y_scores = np.concatenate([li_posi, li_nega])
    y_true = np.concatenate([np.ones(len(li_posi)), np.zeros(len(li_nega))])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(output_path + title + ".png")
    plt.close("all")


def plot_kde_curve(title, li):
    plt.figure(figsize=(6, 4))
    sns.kdeplot(li, fill=True)
    plt.axvline(0, color='k', linestyle='--', linewidth=4, alpha=0.5)
    p = sum([int(t > 0) + int(t == 0) / 2 for t in li]) / len(li)
    plt.title("Pr$[d > 0] = %.2f$" % (p), fontsize=24)
    plt.xlabel('$d = s_+ - s_-$', fontsize=30)
    plt.ylabel('Density', fontsize=30)
    plt.xticks(fontsize=18)
    plt.yticks([])
    # plt.figtext(0.5, 0.13, , ha="center", fontsize=18, color='black')
    q_large = np.percentile(li, 98) + np.std(li) / 5
    q_small = np.percentile(li, 2) - np.std(li) / 5
    plt.xlim(min(q_small, -q_large), max(q_large, -q_small))

    plt.tight_layout()
    plt.savefig(output_path + title + ".pdf")
    plt.close("all")


parser = argparse.ArgumentParser()
# which method direct/preidction/clustering
parser.add_argument("--method", type=str, required=True)
# which dataset
parser.add_argument("--dataset", type=str, required=True)
# which experiment
parser.add_argument("--experiment", type=str, required=True)
args = parser.parse_args()

input_path = "./%s/%s/experiment_%s.json" % (args.method, args.dataset, args.experiment)
output_path = "./%s/%s/" % (args.method, args.dataset)

with open(input_path, 'r') as json_file:
    data = json.load(json_file)
x_values = [t[0] for t in data]
y_values = [t[1] for t in data]
# plot_roc_curve(args.experiment + "_joint_evaluation", x_values, y_values)
x_values = [t[0] - t[1] for t in data]
y_values = [t[1] - t[0] for t in data]
# plot_roc_curve(args.experiment + "_single_evaluation", x_values, y_values)
t_statistic, p_value = stats.ttest_1samp(x_values, 0)
p = sum([int(t[0] - t[1] > 0) + int(t[0] == t[1]) / 2 for t in data]) / len(data)
se = np.std(x_values) / len(x_values)**.5
with open(output_path + args.experiment + "_ttest_result" + ".txt", "w") as f:
    print("mean = %.3f\nstd = %.3f\nse = %.3f\nts = %.3f\np_value = %.3e\n" % (np.mean(x_values), np.std(x_values), se, t_statistic, p_value), file=f)
    print("%.3f & %.3f & %.3f & %.3f & %.1e\n" % (np.mean(x_values), np.std(x_values), se, t_statistic, p_value), file=f)

plot_kde_curve(args.experiment, x_values)
