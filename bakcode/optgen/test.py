import torch
import numpy as np
import random
import argparse
from tqdm import tqdm
random.seed(0)

if __name__ == '__main__':
        file1 = open("../dlrm_embedding_temp/fbgemm_t856_bs65536_15_dataset_cache_miss_trace.txt","r")
        file2 = open("../dlrm_embedding_temp/fbgemm_t856_bs65536_15_prefetch_trace.txt","w")
        content_list = file1.readlines()

        start = 0
        end = len(content_list)
        interval = 5
        l = np.arange(start, end , interval)
        for i in tqdm(l):
                new_content = [0,int(float(content_list[i])),random.randint(0, 412403234),int(float(content_list[i+2])),0]
                file2.write(str(new_content))
                file2.write("\n")
                for j in range(4):
                        file2.write(str([0,0,0,0,0]))
                        file2.write("\n")
        file1.close()
        file2.close()

