#!/usr/bin/python

import numpy as np
import sys
import torch

from models      import LCN, LCNSpiking
from modelsDummy import FC,  FCSpiking

from data   import CopyRedChannel, OffSpikes, RateEncodeData, LatencyEncodeData, CopyEncodeLabels
from data   import loadData, generateDataloaders, nSteps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 1

# Watchdog option
# https://stackoverflow.com/questions/18599339/watchdog-monitoring-file-for-changes

def main():

	# TODO: pass path as cmd line arg
	path    = 'C:/Users/taasi/Desktop/RunSNN/files'
	onvPath = f'{path}/onvOut.csv'
	resPath = f'{path}/resultOut.csv'

	spiking = True

	# ********* TEMP SETUP ******** #

	fakeOnv =  np.ones((2, 14400))
	np.savetxt(onvPath, fakeOnv, delimiter=",", fmt='%.3e')

	# ****** CREATE DELTAONV ****** #

	# read from ONV file, should be shape (2, #photoreceptors)
	# first row is prev onv, second row is cur one
	
	prevOnv = None
	curOnv  = None

	curOnvString = None		# save string version to easily write back later

	with open(onvPath, 'r') as f:
		for i, line in enumerate(f):
			if (i == 1):
				curOnvString = line
				curOnv  = np.fromstring(line, sep=',')
			else:
				prevOnv = np.fromstring(line, sep=',') 

	deltaOnv = curOnv - prevOnv

	# only keep curOnv in file
	with open(onvPath, 'w') as f:
		f.write(curOnvString)


	# ********* RUN MODEL ********* #
	# TODO: use the real models

	# setup and run the model
	# model = FC()

	if spiking:
		model = LCNSpiking(14400, 2, 15, 2, 5, 0.9, 0.8, True)
	else:
		model = LCN(14400, 2, 15, 2, 5, True)

	model.to(torch.float)
	model.to(device)

	inputs = torch.from_numpy(deltaOnv)
	inputs = inputs.float()

	if spiking:
		rate = RateEncodeData(nSteps, 1, 0)
		inputs = rate(inputs)
		inputs = inputs[None, :]

	print(inputs.shape)
	inputs = inputs.to(device)

	output = model(inputs)

	if spiking:
		output = output[2]

	output = output.cpu().detach().numpy().ravel()
	print(output)

	# write results to file
	with open(resPath, 'w') as the_file:
		the_file.write(f'{ output[0] }, { output[1] }')


if __name__ == "__main__":
	main()