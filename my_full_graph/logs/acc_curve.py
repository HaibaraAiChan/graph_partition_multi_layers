import numpy as np
import matplotlib.pyplot as plt




def read_test_acc(filename):
    array=[]
    with open(filename) as f:
        for line in f:
            if ('Run'in line.strip() )and ( 'Test' in line.strip()):
                # print(type(acc))
                acc=line.split()[-1]
                # print(type(acc))
                if '%' in acc:
                    acc=acc[:-1] 
                    acc=float(acc)
                    acc=float("{0:.4f}".format(acc/100))
                else:
                    acc=float(acc)
                array.append(acc)
    print(array[:10])
    print(len(array))
    return array


if __name__=='__main__':
    bench_path = '../../benchmark_full_graph/logs/'
    my_path = '../../my_full_graph/logs/'
    
    # model='sage'
    # model_p='sage/'
    model = 'gat'
    model_p='gat/'
    
    # bench_file='reddit.log'
    # my_file = 'reddit_full_1238.log'
    # DATASET='reddit'
    
    # bench_file='bench_product_1236.log'
    # my_file = 'products/my_full_graph_products_1236.log'
    # DATASET='ogbn-products_1236'
    DATASET='reddit'
    # DATASET='arxiv'
    # DATASET= 'cora'
    # DATASET= 'pubmed'
    seed=1236
    # seed=1238
    # seed=1237
    DATASET_seed=DATASET+'_'+str(seed)
    log_file=DATASET_seed+'.log'
    
    bench_full = read_test_acc(bench_path+model_p+log_file)
    my_full = read_test_acc(my_path + model_p+log_file)
    
    # fig=plt.figure(figsize=(12,6))
    fig,ax=plt.subplots(figsize=(24,6))
    x=range(len(bench_full))
    x2=range(len(my_full))
    ax.plot(x, bench_full, label='benchmark '+DATASET )
    
    ax.plot(x, my_full, label='my script full graph '+DATASET)
    ax.set_title(model+' '+DATASET)
    plt.ylim([0,1])
    plt.xlabel('epoch')
    
    # fig,ax=plt.subplots()
    # ax.autoscale(enable=True,axis='y',tight=False)
    # y_pos= np.arange(0,1000,step=100)
    # labels=np.arange(0,1,step=0.1)
    # print(labels)
    # plt.yticks(y_pos,labels=labels)
    plt.ylabel('Test Accuracy')
    
    plt.legend()
    # plt.savefig('reddit.pdf')
    plt.savefig(DATASET_seed+'.png')
    # plt.show()