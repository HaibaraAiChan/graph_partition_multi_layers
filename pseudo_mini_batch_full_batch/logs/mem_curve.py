import numpy as np
import matplotlib.pyplot as plt




def read_mem(filename):
    nvidia_smi=[]
    cuda_cur=[]
    cuda_max=[]
    print('start------')
    with open(filename) as f:
        for line in f:
            # print(line)
            if ('Nvidia-smi'in line) :
                mem=line.split()[-2]
                mem=float(mem)
                nvidia_smi.append(mem)
            elif ('    Memory Allocated'in line) :
                mem=line.strip().split()[-2]
                mem=float(mem)
                cuda_cur.append(mem)
            elif ('Max Memory Allocated' in line) :
                mem=line.split()[-2]
                mem=float(mem)
                cuda_max.append(mem)
    print(nvidia_smi[:10])
    print(len(nvidia_smi))
    return nvidia_smi, cuda_cur, cuda_max


if __name__=='__main__':
    # bench_path = '../../benchmark_full_graph/logs/'
    my_path = '../../my_full_graph/logs/'
    pseudo_mini_batch_path = '../../pseudo_mini_batch(full_batch)/logs/'
    model='sage'
    model_p='sage/'
    # model = 'gat'
    # model_p='gat/'
    
    # bench_file='reddit.log'
    # my_file = 'reddit_full_1238.log'
    # DATASET='reddit'
    
    # bench_file='bench_product_1236.log'
    # my_file = 'products/my_full_graph_products_1236.log'
    # DATASET='ogbn-products_1236'
    # DATASET='reddit'
    # DATASET='arxiv'
    DATASET= "cora"
    # DATASET= 'pubmed'
    seed=1236
    # seed=1237
    runs=1
    DATASET_seed=DATASET+'_'+str(seed)
    # log_file=DATASET_seed+".log"
    DATASET_seed_runs=DATASET+'_'+str(seed)+'_run_'+str(runs)
    full_log_file = DATASET_seed_runs+"_mem_compare.log"
    #------------------------------------------------------------------
    seed=1238
    DATASET_seed=DATASET+'_'+str(seed)
    nb= 4 # number of batches
    
    DATASET_seed_nb_runs = DATASET+'_'+str(seed)+'_nb_'+str(nb)+'_run_'+str(runs)
    nb_log_file = DATASET_seed_nb_runs+"_mem_compare.log"

    # bench_full = read_test_acc(bench_path+model_p+log_file)
    print(my_path + model_p+full_log_file)
    print(pseudo_mini_batch_path + model_p+nb_log_file)

    my_full_nvidia_smi, cuda_mem, cuda_mem_max = read_mem(my_path + model_p+full_log_file)
    pseudo_mini_batch_nvidia_smi, p_cuda_mem, p_cuda_mem_max = read_mem(pseudo_mini_batch_path + model_p+nb_log_file)
    
    # fig=plt.figure(figsize=(12,6))
    fig,ax=plt.subplots(figsize=(24,12))
    # x=range(len(bench_full))
    x=range(len(pseudo_mini_batch_nvidia_smi))
    
    # ax.plot(x, bench_full, label='benchmark '+DATASET )
    
    ax.plot(x, my_full_nvidia_smi, '--',label= "my script full graph nvidia-smi "+ DATASET)
    ax.plot(x, cuda_mem, '--',label= " my script full graph cuda mem " + DATASET)
    ax.plot(x, cuda_mem_max, '--',label= "my script full graph cuda mem max " + DATASET)
    ax.plot(x, pseudo_mini_batch_nvidia_smi, label = "pseudo_mini_batch (full batch) Nvidia-smi nb "+str(nb)+' '+ DATASET )
    ax.plot(x, p_cuda_mem,label= " pseudo_mini_batch cuda mem nb "+str(nb)+' '+  DATASET)
    ax.plot(x, p_cuda_mem_max, label= "pseudo_mini_batch cuda mem max nb "+str(nb)+' '+  DATASET)
    ax.set_title(model+' '+DATASET)
    ax.set_yscale('log')
    
    # plt.ylim([0,10])
    plt.xlabel('epoch*3 (each epoch select 3 point to collect memory consumption)')
    
    # fig,ax=plt.subplots()
    # ax.autoscale(enable=True,axis='y',tight=False)
    # y_pos= np.arange(0,1000,step=100)
    # labels=np.arange(0,1,step=0.1)
    # print(labels)
    # plt.yticks(y_pos,labels=labels)
    plt.ylabel('GPU memory consumption')
    
    plt.legend()
    # plt.savefig('reddit.pdf')
    plt.savefig(DATASET_seed_nb_runs+'.png')
    # plt.show()