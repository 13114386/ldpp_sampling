from __future__ import unicode_literals, print_function, division
import os
from collections import Counter


def repeated_word_stats(folder, inputname, outputname=None):
    ifpath = os.path.join(folder, inputname)
    dup_counter = Counter()
    total_count = 0
    with open(ifpath, "r") as f:
        content = f.read()
    lines = content.splitlines()
    for l in lines:
        tokens = l.split()
        if tokens[-1] == '.':
            tokens = tokens[:-1]
        dup_counter.update({k: c for k, c in Counter(tokens).items() if c > 1})
        total_count += len(tokens)
    repeat_sum = sum(dup_counter.values())
    dup_counter.update({"#total_word_count#": total_count,
                        "#sum_dup_count#": repeat_sum})
    stat = dict(dup_counter)
    stat["#repeat_%#"] = round(float(repeat_sum)/float(total_count), 2)
    if outputname is not None:
        ofpath = os.path.join(folder, outputname)
        with open(ofpath, "w") as f:
            print(stat, file=f)


if __name__ == "__main__":
    folder = "eval_data\\result_2021_12_26_10_47"
    inputname = "eval.result.summary"
    outputname = "eval.result.summary.dup.stat"
    repeated_word_stats(folder, inputname, outputname)
