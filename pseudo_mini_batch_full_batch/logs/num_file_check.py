import sys
import argparse
import os


# def main(path):
# 	res=0
# 	for filename in os.listdir(path):
# 		if filename.endswith(".log") or filename.endswith(".csv"):
# 			res+=1
# 	if res != 4:
# 		print(path)
# 	return 





if __name__=='__main__':
	argparser = argparse.ArgumentParser("info collection")
	# argparser.add_argument('--file', type=str, default='ogbn-arxiv',
		# help="the dataset name we want to collect")
	argparser.add_argument('--filepath', type=str, default='./res')
	args = argparser.parse_args()
	res=0
	# ${file}/${model}/${aggre}/${pMethod}/layers_${layers}/h_${hidden} 
	for filep in os.listdir(args.filepath):
		for model in os.listdir(args.filepath+'/'+filep):
			for aggre in os.listdir( args.filepath+'/'+filep+'/'+model):
				for pmethod in os.listdir(args.filepath+'/'+filep+'/'+model+'/'+aggre):
					for layers in os.listdir(args.filepath+'/'+filep+'/'+model+'/'+aggre+'/'+pmethod):
						for h in os.listdir(args.filepath+'/'+filep+'/'+model+'/'+aggre+'/'+pmethod+'/'+layers):
							res=0
							for f in os.listdir(args.filepath+'/'+filep+'/'+model+'/'+aggre+'/'+pmethod+'/'+layers+'/'+h):
								if 'error' not in f:
									if f.endswith(".log") or f.endswith(".csv"):
										res+=1
								
								
							if res != 4:
								ef = args.filepath+'/'+filep+'/'+model+'/'+aggre+'/'+pmethod+'/'+layers+'/'+h
								print(ef)
								
	# return 
