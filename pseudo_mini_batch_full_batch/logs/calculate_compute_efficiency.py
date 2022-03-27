import os
import numpy as np
import pandas as pd
from statistics import mean
import argparse

def get_fan_out(filename):
	fan_out=filename.split('_')[6]
	# print(fan_out)
	return fan_out
def get_num_batch(filename):
	nb=filename.split('_')[8]
	# print(nb)
	return nb

def colored(r, g, b, text):
	return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)


# def parse_results(filename: str):
# 	with open(filename) as f:
# 		epoch_times = []
# 		train_times=[]
# 		final_train_acc = ""
# 		final_test_acc = ""

# 		nvidia_smi=[]
# 		cuda_mem=[]
# 		cuda_max_mem=[]


# 		for line in f:
# 			line = line.strip()
# 			if line.startswith("Total (block generation + training)time/epoch"):
# 				epoch_times.append(float(line.split(' ')[-1]))
# 			if line.startswith("Training time/epoch"):
# 				train_times.append(float(line.split(' ')[-1]))
# 			if line.startswith("Final Train"):
# 				final_train_acc = line.split(":")[-1]
# 			if line.startswith("Final Test"):
# 				final_test_acc = line.split(":")[-1]
# 			if line.startswith("Nvidia-smi"):
# 				nvidia_smi.append(float(line.split()[-2]))
# 			if line.startswith("Memory Allocated"):
# 				cuda_mem.append(float(line.split()[-2]))
# 			if line.startswith("Max Memory Allocated"):
# 				cuda_max_mem.append(float(line.split()[-2]))
				
# 		return {"Nvidia-smi": np.array(nvidia_smi)[-10:].mean(),
# 				"CUDA_mem": np.array(cuda_mem)[-10:].mean(),
# 				"CUDA_max_mem": np.array(cuda_max_mem)[-10:].mean(),
# 				"epoch_time": np.array(epoch_times)[-10:].mean(),

		
# 				"train_time": np.array(train_times)[-10:].mean(),
				
# 				"final_train_acc": final_train_acc,
# 				"final_test_acc": final_test_acc}


def compute_efficiency_full_graph(filename):
	num_nid=[]
	num_output_nid=0
	full_graph_nid=0
	num_layer = 0   
	with open(filename) as f:
		train_times = []
		for line in f:
			line = line.strip()
			if line.startswith("NumTrainingSamples") or line.startswith("# Train:"):
				num_output_nid=int(line.split(':')[-1])

			if line.startswith("Training time/epoch"):
				train_times.append(float(line.split(' ')[-1]))
			if line.startswith("NumNodes:") or line.startswith("#nodes:") or line.startswith("# Nodes:"):
				full_graph_nid=int(line.split(' ')[-1])
			
			if line.startswith("The number of model layers:"):
				num_layer=int(line.split(' ')[-1])


	n_epoch=len(train_times)
	# print("num_output_nid*n_epoch")
	# print(num_output_nid*n_epoch)
	# print(train_times[-1])
	core_efficiency = (num_output_nid*n_epoch)/sum(train_times)
	
	# real_efficiency=0
	# if full_graph_nid:
	# print('full_graph_nid* num_layer*n_epoch')
	# print(full_graph_nid* num_layer*n_epoch)
	real_efficiency = (full_graph_nid * num_layer * n_epoch)/sum(train_times)
	return core_efficiency, real_efficiency, mean(train_times)




def get_full_batch_size(filename):
	first_layer_num_input_nid=[]
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith('Number of first layer input nodes during this epoch:'):
				first_layer_num_input_nid.append(int(line.split(' ')[-1])) 
	if len(first_layer_num_input_nid)==0:
		return 0
	return int(mean(first_layer_num_input_nid))

def compute_efficiency_full(filename, args,full_input_size=0):
	compute_num_nid=[]
	first_layer_num_input_nid=[]
	num_output_nid=0
	real_efficiency=0
	real_eff_wo_block_to_device=0
	redundancy =0
	sum_pseudo_input_size=0
	if full_input_size==0:
		redundancy=1
	epoch_times = []
	train_wo_to_device =[]
	pure_train_times=[]
	load_block_feature_label_times=[]
	block_to_device_times=[]

	core_pure_train=0
	real_pure_train=0
	OOM_flag=False
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("RuntimeError: CUDA out of memory."):
				OOM_flag=True

			if line.startswith("NumTrainingSamples") or line.startswith('# Train:'):
				num_output_nid=int(line.split(':')[-1])
			if line.startswith("Number of nodes for computation during this epoch:"):
				compute_num_nid.append(int(line.split(' ')[-1]))
			if line.startswith("Training time/epoch"):
				epoch_times.append(float(line.split(' ')[-1]))
			if line.startswith("Training time without block to device /epoch"):
				train_wo_to_device.append(float(line.split(' ')[-1]))
			if line.startswith("Training time without total dataloading part /epoch"):
				pure_train_times.append(float(line.split(' ')[-1]))
			
			if line.startswith("load block tensor time/epoch"):
				load_block_feature_label_times.append(float(line.split(' ')[-1]))
			if line.startswith("block to device time/epoch"):
				block_to_device_times.append(float(line.split(' ')[-1]))

			if line.startswith('Number of first layer input nodes during this epoch:'):
				first_layer_num_input_nid.append(float(line.split(' ')[-1]))


	if not OOM_flag:
		if len(train_wo_to_device)==0:
			# print(' there is no Training time without block to device /epoch !!!!!')
			return 0
		if len(compute_num_nid)!=len(train_wo_to_device) and len(train_wo_to_device)>0:
			# print('num_nid ', len(num_nid))
			# print('train_wo_to_device ', len(train_wo_to_device))
			nn_epoch=len(train_wo_to_device)
			compute_num_nid=compute_num_nid[-nn_epoch:]
			real_efficiency = sum(compute_num_nid)/sum(epoch_times)
			real_eff_wo_block_to_device = sum(compute_num_nid)/sum(train_wo_to_device) 
			core_eff_wo_block_to_device = (num_output_nid*nn_epoch)/sum(train_wo_to_device)
			real_pure_train=sum(compute_num_nid)/sum(pure_train_times)
			core_pure_train=(num_output_nid*nn_epoch)/sum(pure_train_times)
			
		
		n_epoch=len(epoch_times)
		core_efficiency = (num_output_nid*n_epoch)/sum(epoch_times)


	res={}
	if args.epoch_ComputeEfficiency and not OOM_flag:
		res.update({
			'final layer output nodes/epoch time':core_efficiency,
			'all layers input nodes/epoch time':real_efficiency, 
			# 'final layer output nodes/epoch time without block to device':core_eff_wo_block_to_device, 
			# 'all layers input nodes/epoch time without block to device':real_eff_wo_block_to_device,
			'average epoch time':mean(epoch_times)})
	if OOM_flag:		
		res.update({
			'final layer output nodes/pure train time':None, 
			'all layers input nodes//pure train time': None,
			
			# 'average epoch time w/o block to device':mean(train_wo_to_device),
			'average train time per epoch':None,
			# 'average dataloading time per epoch': (mean(epoch_times)-mean(pure_train_times)),
			'average number of nodes for computation':None,
			'average first layer num of input nodes':mean(first_layer_num_input_nid),
		})
	else:
		res.update({
				'final layer output nodes/pure train time':core_pure_train, 
				'all layers input nodes//pure train time': real_pure_train,
				
				# 'average epoch time w/o block to device':mean(train_wo_to_device),
				'average train time per epoch':mean(pure_train_times),
				# 'average dataloading time per epoch': (mean(epoch_times)-mean(pure_train_times)),
				'average number of nodes for computation':mean(compute_num_nid),
				'average first layer num of input nodes':mean(first_layer_num_input_nid),
		})

	if redundancy ==1:
		res.update({"redundancy rate (first layer input)":redundancy})

	if full_input_size >1:
		if not OOM_flag:
			res.update({"redundancy rate (first layer input)":mean(first_layer_num_input_nid)/full_input_size})
		else:
			res.update({"redundancy rate (first layer input)":None})

	if not OOM_flag:
		res.update({
			'average load block input feature time per epoch':mean(load_block_feature_label_times),
			'average block to device time per epoch':mean(block_to_device_times),
			'average dataloading time per epoch': mean(load_block_feature_label_times)+mean(block_to_device_times)
			})
	else:
		res.update({
				'average load block input feature time per epoch':None,
				'average block to device time per epoch':None,
				'average dataloading time per epoch': None
				})
	

	return res
	

def compute_efficiency(filename, args,full_input_size=0):
	compute_num_nid=[]
	first_layer_num_input_nid=[]
	num_output_nid=0
	real_efficiency=0
	real_eff_wo_block_to_device=0
	redundancy =0
	sum_pseudo_input_size=0
	if full_input_size==0:
		redundancy=1
	epoch_times = []
	train_wo_to_device =[]
	pure_train_times=[]
	load_block_feature_label_times=[]
	block_to_device_times=[]

	core_pure_train=0
	real_pure_train=0
	OOM_flag=False
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("RuntimeError: CUDA out of memory."):
				OOM_flag=True

			if line.startswith("NumTrainingSamples") or line.startswith('# Train:'):
				num_output_nid=int(line.split(':')[-1])
			if line.startswith("Number of nodes for computation during this epoch:"):
				compute_num_nid.append(int(line.split(' ')[-1]))
			if line.startswith("Training time/epoch"):
				epoch_times.append(float(line.split(' ')[-1]))
			if line.startswith("Training time without block to device /epoch"):
				train_wo_to_device.append(float(line.split(' ')[-1]))
			if line.startswith("Training time without total dataloading part /epoch"):
				pure_train_times.append(float(line.split(' ')[-1]))
			
			if line.startswith("load block tensor time/epoch"):
				load_block_feature_label_times.append(float(line.split(' ')[-1]))
			if line.startswith("block to device time/epoch"):
				block_to_device_times.append(float(line.split(' ')[-1]))

			if line.startswith('Number of first layer input nodes during this epoch:'):
				first_layer_num_input_nid.append(float(line.split(' ')[-1]))


	if not OOM_flag:
		if len(train_wo_to_device)==0:
			# print(' there is no Training time without block to device /epoch !!!!!')
			return 0
		if len(compute_num_nid)!=len(train_wo_to_device) and len(train_wo_to_device)>0:
			# print('num_nid ', len(num_nid))
			# print('train_wo_to_device ', len(train_wo_to_device))
			nn_epoch=len(train_wo_to_device)
			compute_num_nid=compute_num_nid[-nn_epoch:]
			real_efficiency = sum(compute_num_nid)/sum(epoch_times)
			real_eff_wo_block_to_device = sum(compute_num_nid)/sum(train_wo_to_device) 
			core_eff_wo_block_to_device = (num_output_nid*nn_epoch)/sum(train_wo_to_device)
			real_pure_train=sum(compute_num_nid)/sum(pure_train_times)
			core_pure_train=(num_output_nid*nn_epoch)/sum(pure_train_times)
			
		
		n_epoch=len(epoch_times)
		core_efficiency = (num_output_nid*n_epoch)/sum(epoch_times)


	res={}
	if args.epoch_ComputeEfficiency and not OOM_flag:
		res.update({
			'final layer output nodes/epoch time':core_efficiency,
			'all layers input nodes/epoch time':real_efficiency, 
			# 'final layer output nodes/epoch time without block to device':core_eff_wo_block_to_device, 
			# 'all layers input nodes/epoch time without block to device':real_eff_wo_block_to_device,
			'average epoch time':mean(epoch_times)})
	if OOM_flag:		
		res.update({
			'final layer output nodes/pure train time':None, 
			'all layers input nodes//pure train time': None,
			
			# 'average epoch time w/o block to device':mean(train_wo_to_device),
			'average train time per epoch':None,
			# 'average dataloading time per epoch': (mean(epoch_times)-mean(pure_train_times)),
			'average number of nodes for computation':None,
			'average first layer num of input nodes':None,
		})
	else:
		res.update({
				'final layer output nodes/pure train time':core_pure_train, 
				'all layers input nodes//pure train time': real_pure_train,
				
				# 'average epoch time w/o block to device':mean(train_wo_to_device),
				'average train time per epoch':mean(pure_train_times),
				# 'average dataloading time per epoch': (mean(epoch_times)-mean(pure_train_times)),
				'average number of nodes for computation':mean(compute_num_nid),
				'average first layer num of input nodes':mean(first_layer_num_input_nid),
		})
	if not OOM_flag:
		if redundancy ==1:
			res.update({"redundancy rate (first layer input)":redundancy})
		elif full_input_size >1:
			res.update({"redundancy rate (first layer input)":mean(first_layer_num_input_nid)/full_input_size})
	else:
		if redundancy ==1:
			res.update({"redundancy rate (first layer input)":redundancy})
		else:
			res.update({"redundancy rate (first layer input)":None})
	if not OOM_flag:
		res.update({
			'average load block input feature time per epoch':mean(load_block_feature_label_times),
			'average block to device time per epoch':mean(block_to_device_times),
			'average dataloading time per epoch': mean(load_block_feature_label_times)+mean(block_to_device_times)
			})
	else:
		res.update({
				'average load block input feature time per epoch':None,
				'average block to device time per epoch':None,
				'average dataloading time per epoch': None
				})
	

	return res
	

def cal_compute_eff_full_batch( file_in, model, path_1, path_2, args):
	
	dict_full={}
	
	fan_out=''
	for filename in os.listdir(path_1):
		if filename.endswith(".log"):
			f = os.path.join(path_1,  filename)
			if file_in in f:
				column_names=[]
				full_batch_input_size=0
				fan_out=get_fan_out(filename)
				column_names.append('full batch\n'+fan_out)
				dict_full= compute_efficiency_full(f, args)
				full_batch_input_size=get_full_batch_size(f)
				res_full=[dict_full]

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
					for filename_ in os.listdir(path_r):
						if filename_.endswith(".log"):
							f_ = os.path.join(path_r, filename_)
							if (file_in in f_) and (fan_out in f_):
								nb=get_num_batch(filename_)
								dict2 = compute_efficiency(f_, args, full_batch_input_size)
								res += [dict2]
								column_names.append('pseudo \n'+str(nb)+' batches\n'+fan_out)
								column_names_csv.append('pseudo '+str(nb)+' batches'+fan_out)
				df=pd.DataFrame(res_full+res).transpose()
				df.columns=column_names
				df.index.name=file_in+' '+model
				print(df.to_markdown(tablefmt="grid"))



				df_res=pd.DataFrame(res).transpose()
				df_res.columns=column_names_csv
				df_res.index.name=file_in + ' '+ args.model
				df_res.to_csv(args.save_path + "compute_efficiency.csv")

				


if __name__=='__main__':
	
	print("computation info data collection start ...... " )
	argparser = argparse.ArgumentParser("info collection")
	argparser.add_argument('--file', type=str, default='ogbn-arxiv',
		help="the dataset name we want to collect")
	argparser.add_argument('--model', type=str, default='sage')
	argparser.add_argument('--aggre', type=str, default='lstm')
	argparser.add_argument('--selection-method', type=str, default='range')
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
		
	path_1 += args.aggre  +'/'
	path_2 += args.aggre +'/'+args.selection_method+'/'

	print("full batch vs pseudo minibatch  compute efficiency (output nodes/time, input nodes /time):")
	cal_compute_eff_full_batch(file_in, args.model, path_1, path_2, args)
	
		
	# files= ['cora', 'pubmed', 'reddit', 'arxiv', 'products']
	# for step,file_in in enumerate(files):
	
	# 	cal_compute_eff_full_batch(file_in, model)
	# 	print()
	# 	print()






