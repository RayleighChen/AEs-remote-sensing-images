import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'


root = '/media/dl/7a4fb85e-b20b-4ac9-bcd4-f7e0b49e0b00/rs-data/class/'

#AEs

datasets = ['AID','EuroSAT','NWPU-RESISC45','RSD46-WHU','RSI-CB128','RSI-CB256','UCMerced_LandUse']
models = ['vgg16','resnet50','inceptionresnetv2','inceptionv4','resnext101_32x4d','densenet121', 'pnasnet5large']
tasks = ['val']
types = ['FGSM','BIM','deepfool']
#types = ['FGSM','BIM','deepfool','CW','JSMA']

#data_root| model_root| task_root| types_root | 
#AID/densenet/train/FGSM

def creat_fold(path):
	if os.path.exists(path) == False:
		os.mkdir(path)

for dataset in datasets:
	data = os.path.join(root, dataset)
	for model in models:
		ckp = os.path.join(root, 'models', dataset, model)
		if dataset == 'AID' and model =='vgg16':
			continue
		for task in tasks:
			for _type in types:
				logs = os.path.join(root, 'AEs', dataset, model, task, _type)
				print(dataset, model, task, _type)
				os.system('python attacts.py --data {data} --arch {model} --ckp {ckp} --tasks {task} --attack-type {_type} --logs {logs}'.format(data=data,model=model,ckp=ckp,task=task,_type=_type,logs=logs))


#import os
#
#root = '/media/dl/7a4fb85e-b20b-4ac9-bcd4-f7e0b49e0b00/rs-data/class/'
#
##AEs
#
#datasets = ['AID','EuroSAT/RGB2750','NWPU-RESISC45','RSD46-WHU','RSI-CB128','RSI-CB256','UCMerced_LandUse/Images']
#models = ['vgg16','resnet50','inceptionresnetv2','inceptionv4','resnext101_32x4d','densenet121', 'pnasnet5large']
#tasks = ['train', 'val']
#types = ['FGSM','BIM','deepfool','CW','JSMA']
#
##data_root| model_root| task_root| types_root | 
##AID/densenet/train/FGSM
#
#def creat_fold(path):
#	if os.path.exists(path) == False:
#		os.mkdir(path)
#
#for dataset in datasets:
#	data = os.path.join(root, dataset)
#	for model in models:
#		ckp = os.path.join(root, 'models', dataset, model)
#		for task in tasks:
#			for _type in types:
#				logs = os.path.join(root, 'AEs', dataset, model, task, _type)
#				print(dataset, model, task, _type)
#				os.system('python attacts.py --data {data} --arch {model} --ckp {ckp} --tasks {task} --attack-type {_type} --logs {logs}'.format(data=data,model=model,ckp=ckp,task=task,_type=_type,logs=logs))
