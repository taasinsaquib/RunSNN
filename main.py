#!/usr/bin/python

import numpy as np
import sys, getopt
import torch

from models      import LCN, LCNSpiking, FC
from modelsDummy import FCSpiking

from data   import CopyRedChannel, OffSpikes, RateEncodeData, LatencyEncodeData, CopyEncodeLabels
from data   import loadData, generateDataloaders, nSteps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Watchdog option
# https://stackoverflow.com/questions/18599339/watchdog-monitoring-file-for-changes

def main():

	# *************************************************************************
	# Setup Some Variables
	# *************************************************************************

	deltaData = False		# whether to use normal ONV or delta ONV
	tempData  = False		# generate random data to test the forward pass

	spiking   = False		# whether or not to use an SNN
	useFC     = False		# whether or not to use an FC Network

	# TODO: pass path1, path2, modelDictPath as cmd line args
	path1   = 'C:/Users/taasi/Desktop/RunSNN/files'
	onvPath = f'{path1}/onvOut.csv'

	path2   = 'C:/Users/taasi/Desktop/biomechanical_eye_siggraph_asia_19'
	resPath = f'{path2}/resultOut.csv'

	modelDictPath = 'C:/Users/taasi/Desktop/trainSNNs/model_dicts'
	modelDictName = ""

	# *************************************************************************
	# Command Line Arg Processing
	#
	# d for deltaONV, t for temporary, random forward pass (used for testing)
	# s for LCNSpiking(), l for LCN(), f for FC()
	# 	pass model dict name with the flags
	#
	# *************************************************************************


	argList = sys.argv[1:]
	options = "dfst"
	longOptions = []

	try:
		# Parsing argument
		arguments, values = getopt.getopt(argList, options, longOptions)
		 
		# checking each argument
		for curArg, curVal in arguments:

			if curArg in ("-d"):
				# print ("Using delta ONV")
				deltaData = True

			elif curArg in ("-f"):
				# print("using a FC Net")
				useFC = True

			elif curArg in ("-s"):
				# print("using a Spiking NN")
				spiking = True

			elif curArg in ("-t"):
				# print("Using random data to test forward pass pipeline")
				tempData = True
				 
	except getopt.error as err:
		# output error, and return with an error code
		print (str(err))


	# *************************************************************************
	# Get Data
	# *************************************************************************

	# Populate ONV files for testing
	nPhotoreceptors = 14400

	# ********** Rand ONV ********* #
	if tempData is True:		

		nOnv = 1
		if deltaData:
			nOnv = 2
		onv = np.random.rand(nOnv, nPhotoreceptors)

		np.savetxt(onvPath, onv, delimiter=",", fmt='%.3e')
	

	# **** Read ONV from File ***** #

	# read from ONV file, should be shape (2, #photoreceptors)
	# first row is prev onv, second row is cur one
	
	prevOnv = np.zeros(nPhotoreceptors)
	curOnv  = None

	curOnvString = None		# save string version to easily write back later

	with open(onvPath, 'r') as f:
		for i, line in enumerate(f):
			if (i == 1):
				curOnvString = line
				curOnv  = np.fromstring(line, sep=',')
			else:
				prevOnv = np.fromstring(line, sep=',') 

	if deltaData is not True:
		inputs = prevOnv
	else:
		inputs = curOnv - prevOnv

		# only keep curOnv in file
		with open(onvPath, 'w') as f:
			f.write(curOnvString)

	# *************************************************************************
	# Run Models
	# *************************************************************************
	
	if spiking is True:
		model = LCNSpiking(nPhotoreceptors, 2, 15, 2, 5, 0, 1, True)
	elif useFC is True:
		model = FC()
		model.load_state_dict(torch.load(f'{modelDictPath}/FC_normal_200epoch', map_location=device))
	else:
		model = LCN(nPhotoreceptors, 2, 15, 2, 5, True)

	# model and inputs are float()ed and sent to device
	model.to(torch.float)
	model.to(device)

	inputs = torch.from_numpy(inputs)
	inputs = inputs.float()

	# TODO: change params if needed
	# Encode Spiking Inputs
	if spiking:
		rate = RateEncodeData(nSteps, 1, 0)
		inputs = rate(inputs)
		inputs = inputs[None, :]

	inputs = inputs.to(device)

	# run model, get outputs
	output = model(inputs)
	if spiking:
		output = output[2]

	output = output.cpu().detach().numpy().ravel()
	print(f'EXE output: {output}')

	# write results to file
	with open(resPath, 'w') as the_file:
		the_file.write(f'{ output[0] }, { output[1] }')
		the_file.close()

	# *************************************************************************
	# Cleanup
	# *************************************************************************

	# clear ONV file if it was randomly generated
	if tempData is True:
		file = open(onvPath,"w")
		file.close()


if __name__ == "__main__":
	main()

"""
TODO
	Add model_dict name input on cmd line
	Test delta data
	Test spiking nn

"""