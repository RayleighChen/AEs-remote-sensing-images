
import argparse
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets

import pretrainedmodels
import pretrainedmodels.utils
from skimage import img_as_ubyte

import foolbox
import numpy as np
from scipy.special import softmax
import imageio
#from skimage.io import imread
from skimage.external.tifffile import imsave
import re

from util import *

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='Adversarial Examples')
parser.add_argument('--data', metavar='DIR', default="path_to_data",
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: densenet121)')
parser.add_argument('--ckp', metavar='CKP', default="path_to_model",
                    help='path to model')
parser.add_argument('--tasks', metavar='Tasks', default="task_type",
                    help='The type of task')
parser.add_argument('--attack-type', metavar='Attack', default="attack_type",
                    help='The type of attack')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--logs', metavar='Logs', default="logs",
                    help='logs')

def ad_type(model, attack_type):
    if attack_type == 'FGSM':
        return foolbox.attacks.FGSM(model)
    if attack_type == 'BIM':
        return foolbox.attacks.BIM(model, distance=foolbox.distances.Linf)
    if attack_type == 'BIMl1':
        return foolbox.attacks.L1BasicIterativeAttack(model)
    if attack_type == 'BIMl2':
        return foolbox.attacks.L2BasicIterativeAttack(model)
    if attack_type == 'CW':
        return foolbox.attacks.CarliniWagnerL2Attack(model)
    if attack_type == 'deepfool':
        return foolbox.attacks.DeepFoolAttack(model)
    if attack_type == 'deepfooll2':
        return foolbox.attacks.DeepFoolL2Attack(model)
    if attack_type == 'deepfoollinf':
        return foolbox.attacks.DeepFoolLinfinityAttack(model)
    if attack_type == 'JSMA':
        return foolbox.attacks.SaliencyMapAttack(model)

def modify(model, model_name, num_channels):
    if model_name == 'vgg16':
        model._features[0] = nn.Conv2d(num_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    if model_name == 'resnet50':
        model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    if model_name == 'inceptionresnetv2':
        model.conv2d_1a.conv = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    if model_name == 'inceptionv4':
        model.features[0].conv = nn.Conv2d(num_channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    if model_name == 'resnext101_32x4d':
        model.features[0] = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    if model_name == 'densenet121':
        model.features.conv0 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    if model_name == 'pnasnet5large':
        model.conv_0.conv = nn.Conv2d(num_channels, 96, kernel_size=(3, 3), stride=(2, 2), bias=False)
    if model_name == 'squeezenet1_1':
        model.features[0] = nn.Conv2d(num_channels, 64, kernel_size=(3, 3), stride=(2, 2))
    if model_name == 'alexnet':
        model._features[0] = nn.Conv2d(num_channels, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    if model_name == 'nasnetalarge':
        model.conv0.conv = nn.Conv2d(num_channels, 96, kernel_size=(3, 3), stride=(2, 2), bias=False)

def record(model, class_indices):
	#valdir = os.path.join(args.data, 'val')
    #val_data = MSdata(image_dir=valdir, resize_height=max(input_size), resize_width=max(input_size))
    #val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
	#for i, (input, target) in enumerate(val_loader):
		

	class_name = list(class_indices.keys())
	data_name = os.path.basename(args.data)
	ads_path = os.path.join(os.path.dirname(args.data), 'AEs', data_name, args.arch, args.tasks, args.attack_type)
	ads_record = open(os.path.join(ads_path, '{}_{}_{}_{}.txt'.format(data_name, args.arch, args.tasks, args.attack_type)), 'w')

	file_path = os.path.join(args.data, args.tasks)
	files = get_fils(file_path)
	
	mean = np.array([1353.036, 1116.468, 1041.475, 945.344, 1198.498, 2004.878, 2376.699, 2303.738, 732.957, 12.092, 1818.820, 1116.271, 2602.579])
	mean = mean / mean.max()
	std = np.array([65.479, 154.008, 187.997, 278.508, 228.122, 356.598, 456.035, 531.570, 98.947, 1.188, 378.993, 303.851, 503.181])
	std = std / std.max()

	preprocessing = dict(mean=mean, std=std, axis=-3)
	fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=len(class_name), preprocessing=preprocessing)
	attack = ad_type(fmodel, args.attack_type)

	for batch_files in yield_mb(files, args.batch_size, shuffle=False):
	
		categorical_label_from_full_file_name(batch_files, class_indices)
		images, labels = data_generator(batch_files, model.input_size[1:], class_indices)
		labels = np.argmax(labels, axis=1)
		#print(images.shape, labels.shape)		
		cfds = fmodel.forward(images)

		cfds = softmax(cfds, axis=1)
		
		adversarials = attack(images, labels)
		ads_cfds = fmodel.forward(adversarials)
		ads_cfds = softmax(ads_cfds, axis=1)
		
		for i in range(args.batch_size):
			_class = batch_files[i].split('/')[-2]
			file_name = os.path.basename(batch_files[i]).split('.')[-2]
			model_idx = cfds[i].argmax(-1)
			model_cla = class_name[model_idx]
			model_cfs = cfds[i].max() * 100
			model_flg = (model_idx == labels[i])
			
			ads_idx   = ads_cfds[i].argmax(-1)
			ads_cla   = class_name[ads_idx]
			ads_cfs   = ads_cfds[i].max() * 100
			ads_flg   = bool(1-(ads_idx == labels[i]))

			ads_record.write("{_class}+{file_name}+{model_cfs:.2f}+{ads_cla}+{ads_cfs:.2f}.tif,{_class},{model_cla},{model_idx},{model_cfs:.2f},{model_flg},{ads_cla},{ads_idx},{ads_cfs:.2f},{ads_flg}\n".format(_class=_class, file_name=file_name, model_idx=model_idx, model_cla=model_cla, model_cfs=model_cfs, model_flg=model_flg,					ads_idx=ads_idx, ads_cla=ads_cla, ads_cfs=ads_cfs, ads_flg=ads_flg))
			
			if model_flg and ads_flg:
				image_path = os.path.join(ads_path, _class)
				imsave("{image_path}/{_class}+{file_name}+{model_cfs:.2f}+{ads_cla}+{ads_cfs:.2f}.tif".format(image_path=image_path,_class=_class, file_name=file_name,model_cfs=model_cfs,ads_cla=ads_cla,ads_cfs=ads_cfs), img_as_ubyte(np.transpose(adversarials[i], (1, 2, 0))))


def main():

	global args
	args = parser.parse_args()
	
	mode_ckp_path = os.path.join(args.ckp, 'checkpoint.pth.tar')
	
	checkpoint = torch.load(mode_ckp_path,map_location=torch.device('cpu'))
	class_indices = checkpoint['class_to_idx']
	
	num_classes = len(class_indices)
	
	model = pretrainedmodels.__dict__[args.arch](num_classes=1000,pretrained='imagenet').eval()
	
	modify(model, args.arch, 13)

	if 'squeeze' in args.arch:
		model.last_conv = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
		#model.num_classes = len(datas.classes)
	else:
		num_ftrs = model.last_linear.in_features
		model.last_linear = nn.Linear(num_ftrs, num_classes)
	
	state_dict = checkpoint['state_dict']
	remove_data_parallel = True # Change if you don't want to use nn.DataParallel(model)
	
	pattern = re.compile(
		r'^(.*\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
	for key in list(state_dict.keys()):
		match = pattern.match(key)
		new_key = match.group(1) + match.group(2) if match else key
		new_key = new_key[7:] if remove_data_parallel else new_key
		state_dict[new_key] = state_dict[key]
		# Delete old key only if modified.
		if match or remove_data_parallel: 
			del state_dict[key]
	
	model.load_state_dict(state_dict)
	record(model, class_indices)


if __name__ == "__main__":
	main()
	print('OK')


