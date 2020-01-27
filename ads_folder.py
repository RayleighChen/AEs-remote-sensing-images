import os

# wqlian@202.114.96.180
# wqlian@202.114.96.180:/home/wqlian/project/rayleigh/data/


root = '/home/wqlian/project/rayleigh/data/'

#AEs

datasets = ['UCMerced_LandUse','AID','UCMerced_LandUse','NWPU-RESISC45','MSRAT','SEN1-2','EuroSAT','EuroSAT-MS']
models = ['alexnet','vgg16','resnet50','inceptionresnetv2','inceptionv4','resnext101_32x4d','densenet121', 'squeezenet1_1','pnasnet5large']
tasks = [ 'val']
_types = ['FGSM','BIM','deepfool','CW']

#data_root| model_root| task_root| _types_root | 
#AID/densenet/train/FGSM

def creat_fold(path):
	if os.path.exists(path) == False:
		os.mkdir(path)

for dataset in datasets:
	data_path = os.path.join(root, dataset)
	classes = os.path.join(data_path,'classes.txt')
	
	class_indices = {}
	for line in open(classes).readlines():
		key,value = line.split(',')[0], int(line.split(',')[1])
		class_indices[key] = value
	class_name = list(class_indices.keys())
	db_path = os.path.join(root, 'AEs', dataset)
	creat_fold(db_path)

	for model in models:
		model_path = os.path.join(db_path, model)
		creat_fold(model_path)

		for task in tasks:
			task_path = os.path.join(model_path, task)
			creat_fold(task_path)

			for _type in _types:
                                
				fold_path = os.path.join(task_path, _type)
				creat_fold(fold_path)
				for _class in class_name:
					_class_fold = os.path.join(fold_path, _class)
					creat_fold(_class_fold)
