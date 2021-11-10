#!/usr/bin/python

import numpy as np
import torch

from models import FC, FCSpiking

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 1

# Watchdog option
# https://stackoverflow.com/questions/18599339/watchdog-monitoring-file-for-changes

def getOnv(path):
	onv = np.genfromtxt(path, delimiter=',')
	return onv

def main():

	# TODO: pass path as cmd line arg
	path    = 'C:/Users/taasi/Desktop/process'
	onvPath = f'{path}/onvOut.csv'
	resPath = f'{path}/resultOut.csv'

	# read from ONV file
	onv = getOnv(onvPath)
	print(onv.shape)

	# clear ONV file
	open(onvPath, 'w').close()

	# setup and run the model
	model = FC()
	model.float()
	model.to(device)

	inputs = torch.from_numpy(onv)
	inputs = inputs.float()

	output = model(inputs)
	output = output.cpu().detach().numpy()

	print(output)

	# write results to file
	with open(resPath, 'w') as the_file:
	    the_file.write(f'{ output[0] }, { output[1] }')

if __name__ == "__main__":
	main()