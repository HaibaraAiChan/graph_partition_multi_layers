import os
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
# from turtle import Turtle
import numpy as np
import pandas as pd
import argparse
# import time

def colored(r, g, b, text):
	return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)


def parse_results(filename: str):
	with open(filename) as f:
		epoch_times = []
		train_times=[]
		connect_checking_times=[]
		block_gen_times=[]
		batch_gen_times=[]
		final_train_acc = ""
		final_test_acc = ""

		for line in f:
			line = line.strip()
			if line.startswith("Total (block generation + training)time/epoch"):
				epoch_times.append(float(line.split(' ')[-1]))
			if line.startswith("Training time/epoch"):
				train_times.append(float(line.split(' ')[-1]))
			if line.startswith("Final Train"):
				final_train_acc = line.split(":")[-1]
			if line.startswith("Final Test"):
				final_test_acc = line.split(":")[-1]
			if line.startswith("connection checking time:"):
				connect_checking_times.append(float(line.split(' ')[-1]))
			if line.startswith("block generation total time"):
				block_gen_times.append(float(line.split(' ')[-1]))
			if line.startswith("average batch blocks generation time:"):
				batch_gen_times.append(float(line.split(' ')[-1]))
			
			
			
			
		return {"epoch_time": np.array(epoch_times)[-10:].mean(),

				"train_time per epoch": np.array(train_times)[-10:].mean(),
				"connect checking time per epoch: ": np.array(connect_checking_times)[-10:].mean(),
				"block generation time per epoch: ": np.array(block_gen_times)[-10:].mean(),
				"batches generation time per epoch: ": np.array(batch_gen_times)[-10:].mean(),
				"final_train_acc": final_train_acc,
				"final_test_acc": final_test_acc}

def parse_time_results(filename: str):
	OOM_flag=False
	with open(filename) as f:
		epoch_times = []
		pure_train_times=[]
		connect_checking_times=[]
		block_gen_times=[]
		batch_gen_times=[]
		load_block_feature_label_times=[]
		block_to_device_times=[]
		input_nodes_size=[]
		log_input_features_size=[]
		log_block_size_to_device=[]
		infeat_size=0

		for line in f:
			line = line.strip()
			if line.startswith("RuntimeError: CUDA out of memory."):
				OOM_flag=True
			if line.startswith("Total (block generation + training)time/epoch"):
				epoch_times.append(float(line.split(' ')[-1]))
			if line.startswith('Total dataloading + training time/epoch'):
				epoch_times.append(float(line.split(' ')[-1]))

			if line.startswith("in feats:"):
				infeat_size=int(line.split(' ')[-1])

			if line.startswith('Number of first layer input nodes during this epoch:'):
				input_nodes_size.append(float(line.split(' ')[-1]))
			if line.startswith("load block tensor time/epoch"):
				load_block_feature_label_times.append(float(line.split(' ')[-1]))
			if line.startswith("block to device time/epoch"):
				block_to_device_times.append(float(line.split(' ')[-1]))

			if line.startswith('input features size transfer per epoch'):
				log_input_features_size.append(float(line.split(' ')[-1]))
			if line.startswith('blocks size to device per epoch '):
				log_block_size_to_device.append(float(line.split(' ')[-1]))
			

			if line.startswith("Training time without total dataloading part /epoch"):
				pure_train_times.append(float(line.split(' ')[-1]))

			if line.startswith("connection checking time:"):
				connect_checking_times.append(float(line.split(' ')[-1]))
			if line.startswith("block generation total time"):
				block_gen_times.append(float(line.split(' ')[-1]))
			if line.startswith("average batch blocks generation time:"):
				batch_gen_times.append(float(line.split(' ')[-1]))

			
		if OOM_flag:
			res={"epoch_time": None,
				"pure train_time per epoch": None,
				
				"connect checking time per epoch: ": None,
				"block generation time per epoch: ": None,
				"batches generation time per epoch: ": None,

				"first layer input nodes number per epoch": None,
				"first layer num_input nodes * in_feats per epoch": None,

				"logged input_features_size transfer (pointers* Bytes)": None,
				"logged block_size_to_device transfer (pointers*  Bytes)": None,
				"load block tensor time per epoch": None,
				"block to device time per epoch": None}
		else:	
			res={"epoch_time": np.array(epoch_times)[-10:].mean(),
				"pure train_time per epoch": np.array(pure_train_times)[-10:].mean(),
				
				"connect checking time per epoch: ": np.array(connect_checking_times)[-10:].mean(),
				"block generation time per epoch: ": np.array(block_gen_times)[-10:].mean(),
				"batches generation time per epoch: ": np.array(batch_gen_times)[-10:].mean(),

				"first layer input nodes number per epoch": np.array(input_nodes_size)[-10:].mean(),
				"first layer num_input nodes * in_feats per epoch": np.array(input_nodes_size)[-10:].mean()*infeat_size,

				"logged input_features_size transfer (pointers* Bytes)": np.array(log_input_features_size)[-10:].mean()*1024*1024*1024,
				"logged block_size_to_device transfer (pointers*  Bytes)": np.array(log_block_size_to_device)[-10:].mean()*1024*1024*1024,

				"load block tensor time per epoch": np.array(load_block_feature_label_times)[-10:].mean(),
				"block to device time per epoch": np.array(block_to_device_times)[-10:].mean()
			}
		return res

  





def parse_mem_results(filename: str):
	OOM_flag=False
	with open(filename) as f:

		nvidia_smi=[]
		cuda_mem=[]
		cuda_max_mem=[]

		for line in f:
			line = line.strip()
			if line.startswith("RuntimeError: CUDA out of memory."):
				OOM_flag=True
			if line.startswith("Nvidia-smi"):
				nvidia_smi.append(float(line.split()[-2]))
			if line.startswith("Memory Allocated"):
				cuda_mem.append(float(line.split()[-2]))
			if line.startswith("Max Memory Allocated"):
				cuda_max_mem.append(float(line.split()[-2]))
		if OOM_flag:
			res={"Nvidia-smi": 'OOM',
				"CUDA_mem": 'OOM',
				"CUDA_max_mem": 'OOM',}
		else:
			res={"Nvidia-smi": np.array(nvidia_smi)[-10:].mean(),
				"CUDA_mem": np.array(cuda_mem)[-10:].mean(),
				"CUDA_max_mem": np.array(cuda_max_mem)[-10:].mean(),
				}

		return res

def pprint(dd):
	for (key, value) in enumerate(dd.items()):
		print(str(key)+' '+str(value[:]))

def time_full(path,  file_in, color):
	# path = '../../my_full_graph/logs/sage/'
	color=list(color)
	
	for filename in os.listdir(path):
		if filename.endswith(".log"):
			f = os.path.join(path, filename)
			
			if file_in in f:
				res_ = parse_time_results(f)
				
				print(f)
				res=colored(color[0],color[1],color[2],res_)
				print(res)
				return res_

def GPU_mem_full(path,  file_in, color):
	# path = '../../my_full_graph/logs/sage/'
	color=list(color)
	for filename in os.listdir(path):
		if filename.endswith(".log"):
			f = os.path.join(path, filename)
			
			if file_in in f:
				res_ = parse_mem_results(f)

				print(f)
				res=colored(color[0],color[1],color[2],res_)
				print(res)
				return res_


def time_(path,  file_in):
	# path = '../../my_full_graph/logs/sage/'
	# color=list(color)
	res_list=[]
	for filename in os.listdir(path):
		if filename.endswith(".log"):
			f = os.path.join(path, filename)
			
			if file_in in f:
				res_ = parse_time_results(f)
				res_list.append(res_)
				# print(f)
				# res=colored(color[0],color[1],color[2],res_)
				# print(res)
	return res_list


def time_one(path,  file_in, fan_out):
	for filename in os.listdir(path):
		if filename.endswith(".log"):
			f = os.path.join(path, filename)
			if file_in in f and fan_out in f:
				res = parse_time_results(f)
				return res


def GPU_mem(path,  file_in):
	# path = '../../my_full_graph/logs/sage/'
	# color=list(color)
	res_list=[]
	for filename in os.listdir(path):
		if filename.endswith(".log"):
			f = os.path.join(path, filename)
			
			if file_in in f:
				fan_out=filename.split('_')[6]
				# if not_out_of_memory_check(f):
				res_ = parse_mem_results(f)
				res_list.append((fan_out,res_))
				# print(f)
				# res=colored(color[0],color[1],color[2],res_)
				# print(res)
	return res_list

def GPU_mem_one(path,  file_in, fan_out):

	for filename in os.listdir(path):
		if filename.endswith(".log"):
			f = os.path.join(path, filename)
			
			if file_in in f and fan_out in f:
				res = parse_mem_results(f)
				nb=filename.split('_')[8] # number of batches
				return res,nb
	return False



def one_fan_out():
	res_full =[]
	res=[]
	colors=[(255,100,0), (0,255,0),(150,150,100),(200,0,100),(0,200,100)]
	files= ['cora', 'pubmed', 'reddit', 'arxiv', 'products']
	files= ['arxiv']
	model = 'sage/'
	# model = 'gat/'
	
	path_1 = '../../my_full_graph/logs/'+model
	path_2 = model+'1_runs/'

	for i, file_in in enumerate(files):
		tmp_={'Dataset': file_in}
		tmp_m=GPU_mem_full(path_1, file_in, colors[i])
		tmp_t=time_full(path_1, file_in, colors[i])
		tmp_m.update(tmp_t)
		tmp_.update(tmp_m)
		res_full.append(tmp_)
	df = pd.DataFrame(res_full).transpose()
	# print(df.to_json())
	print()
	print(model)
	print(df.to_markdown())

	for i, file_in in enumerate(files):
		tmp_={'Dataset': file_in}
		tmp_m2=GPU_mem_full(path_2, file_in,colors[i])
		tmp_t2=time_full(path_2, file_in, colors[i])
		tmp_m2.update(tmp_t2)
		tmp_.update(tmp_m2)
		res.append(tmp_)
		
	print() 
	df = pd.DataFrame(res).transpose()
	print(df.to_markdown())
	# df.to_csv("pseudo.csv")
'''
+------------------------------------+--------------+------------+------------+------------+-------------+------------+--------------+
|                                    |   full graph |   10,25,15 |   10,25,20 |   10,25,10 |   10,50,100 |   25,35,40 |   50,100,200 |
+====================================+==============+============+============+============+=============+============+==============+
| Nvidia-smi                         |     3.703    |   3.72449  |   3.91394  |   3.16394  |    3.50574  |   3.47058  |     3.57214  |
+------------------------------------+--------------+------------+------------+------------+-------------+------------+--------------+
| CUDA_mem                           |     0.788891 |   0.555289 |   0.564104 |   0.539163 |    0.600046 |   0.591091 |     0.618186 |
+------------------------------------+--------------+------------+------------+------------+-------------+------------+--------------+
| CUDA_max_mem                       |     1.74719  |   1.44232  |   1.47328  |   1.39608  |    1.57691  |   1.54123  |     1.60839  |
+------------------------------------+--------------+------------+------------+------------+-------------+------------+--------------+
| train_time per epoch               |     0.758371 |   0.311036 |   0.321758 |   0.293009 |    0.340648 |   0.331337 |     0.351762 |
+------------------------------------+--------------+------------+------------+------------+-------------+------------+--------------+
| epoch_time                         |   nan        |   6.08479  |   6.21099  |   5.74145  |    7.18847  |   7.52109  |     8.71988  |
+------------------------------------+--------------+------------+------------+------------+-------------+------------+--------------+
| connect checking time per epoch:   |   nan        |   2.37972  |   2.41495  |   2.29112  |    2.75955  |   2.96835  |     3.44894  |
+------------------------------------+--------------+------------+------------+------------+-------------+------------+--------------+
| block generation time per epoch:   |   nan        |   1.96579  |   2.0164   |   1.89419  |    2.33004  |   2.55721  |     3.00784  |
+------------------------------------+--------------+------------+------------+------------+-------------+------------+--------------+
| batches generation time per epoch: |   nan        |   0.982893 |   1.0082   |   0.947097 |    1.16502  |   1.2786   |     1.50392  |
+------------------------------------+--------------+------------+------------+------------+-------------+------------+--------------+
'''

# def not_out_of_memory_check(filename):
# 	with open(filename) as f:
# 		for line in f:
# 			line = line.strip()
# 			if line.startswith("RuntimeError: CUDA out of memory."):
# 				return False
# 	return True

def main():
	
	print("time and memory data collection start ...... " )
	argparser = argparse.ArgumentParser("info collection")
	argparser.add_argument('--file', type=str, default='ogbn-arxiv',
		help="the dataset name we want to collect")
	argparser.add_argument('--model', type=str, default='sage')
	argparser.add_argument('--aggre', type=str, default='mean')
	# argparser.add_argument('--aggre', type=str, default='lstm')
	argparser.add_argument('--selection-method', type=str, default='random')
	# argparser.add_argument('--selection-method', type=str, default='range')
	argparser.add_argument('--eval',type=bool, default=False)
	argparser.add_argument('--epoch-ComputeEfficiency', type=bool, default=False)
	argparser.add_argument('--epoch-PureTrainComputeEfficiency', type=bool, default=True)
	argparser.add_argument('--save-path',type=str, default='./')
	args = argparser.parse_args()
	file_in=args.file
	model=args.model+'/'
	path_1 = '../../full_batch_train/logs/'+model+'1_runs/'
	path_2 = model+'1_runs/'
	if args.eval:
		path_1+='train_eval/'
		path_2+='train_eval/'
	else:
		path_1+='pure_train/'
		path_2+='pure_train/'
	
	
	path_1 += args.aggre +'/'
	path_2 += args.aggre +'/'+args.selection_method+'/'

	tmp_m=GPU_mem(path_1, file_in)
	tmp_t=time_(path_1, file_in)
	for i, (fan_out, item) in enumerate(tmp_m):
		res_full =[]
		item.update(tmp_t[i])
		res_full.append(item)
		column_names=['full batch '+fan_out]

		nb_folder_list=[]
		for f_item in os.listdir(path_2):
			if 'nb_' in f_item:
				nb_size=f_item.split('_')[1]
				nb_folder_list.append(int(nb_size))
		nb_folder_list.sort()
		nb_folder_list=['nb_'+str(i) for i in nb_folder_list]


		res=[]
		column_names_csv=[]
		for f_item in nb_folder_list:

			path_r=path_2+f_item
			tmp_m2=GPU_mem_one(path_r, file_in, fan_out)
			if not tmp_m2:
				continue
			m2, num_batches = tmp_m2
			t2=time_one(path_r, file_in, fan_out)
			m2.update(t2)
			res.append(m2)
			column_names+=['pseudo \n'+str(num_batches)+' batches \n'+fan_out]
			column_names_csv+=['pseudo '+str(num_batches)+' batches '+fan_out]
		df=pd.DataFrame(res_full+res).transpose()
		df.index.name=file_in + ' '+ args.model
		df.columns=column_names
		print(df.to_markdown(tablefmt="grid"))

		df_res=pd.DataFrame(res).transpose()
		df_res.columns=column_names_csv
		df_res.index.name=file_in + ' '+ args.model
		df_res.to_csv(args.save_path + "time_and_mem.csv")




if __name__=='__main__':
	# one_fan_out()
	main()

	
	# res=[]
	# colors=[(255,100,0), (0,255,0),(150,150,100),(200,0,100),(0,200,100)]
	# files= ['cora', 'pubmed', 'reddit', 'arxiv', 'products']
	# files= ['arxiv']
	# model = 'sage/'
	# model = 'gat/'
	
	
		

 
 
	