import os
import numpy as np


def parse_results(filename: str):
    with open(filename) as f:
        epoch_times = []
        final_train_acc = ""
        final_test_acc = ""
        for line in f:
            line = line.strip()
            if line.startswith("Training time/epoch"):
                epoch_times.append(float(line.split(' ')[-1]))
            if line.startswith("Final Train"):
                final_train_acc = line.split(":")[-1]
            if line.startswith("Final Test"):
                final_test_acc = line.split(":")[-1]
        return {"epoch_time": np.array(epoch_times)[-10:].mean(),
                "final_train_acc": final_train_acc,
                "final_test_acc": final_test_acc}



if __name__=='__main__':
    # model = 'gat/'
    # # DATASET='reddit_gat_1236'
    # DATASET='pubmed_gat_1237'
    # DATASET='cora_gat_1238'
    # DATASET='arxiv_gat_1236'
    
    model = 'sage/'
    # DATASET='reddit_sage_1236'
    DATASET='pubmed_sage_1236'
    DATASET='cora_sage_1236'
    DATASET='arxiv_sage_1236'
    DATASET='product_sage_1236'
    res = parse_results(model + DATASET+'.log')
    print(DATASET)
    print(res)
    print()
    