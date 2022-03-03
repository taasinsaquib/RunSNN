#!/usr/bin/python

import numpy as np
import sys, getopt
import torch
from   torchvision import transforms

from models import FC, LCN, LCNChannelStack
from models import LCNSpiking2, LCNSpikingHybrid, LCNSpikingHybrid2

from data   import CopyRedChannel, OffSpikes, RateEncodeData, LatencyEncodeData, CopyEncodeLabels, OnOffChannels, CopyRedChannel
from data   import loadData, generateDataloaders, nSteps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Watchdog option
# https://stackoverflow.com/questions/18599339/watchdog-monitoring-file-for-changes

def main():

	# *************************************************************************
	# Setup Some Variables
	# *************************************************************************

	# "normal" (default) or "delta"
	dataType = ""

	# "FC" (default), "LCN", or "LCNSpiking"
	modelType = ""

	nSpikingLayers = -1

	gain = 2

	tempData  = False		# generate random data to test the forward pass

	# TODO: pass path1, path2, modelDictPath as cmd line args
	path1   = 'C:/Users/taasi/Desktop/RunSNN/files'
	onvPath = f'{path1}/onvOut.csv'

	path2   = 'C:/Users/taasi/Desktop/biomechanical_eye_siggraph_asia_19'
	resPath = f'{path2}/resultOut.csv'

	modelDictPath = 'C:/Users/taasi/Desktop/trainSNNs/model_dicts'
	modelDictName = "FC_normal_200epoch"		# default, for FC, may not work with certain flags


	# *************************************************************************
	# Command Line Arg Processing
	#
	# d for dataType, t for temporary, random forward pass (used for testing)
	# m for modelType
	# n for	pass model dict name with this flag
	#
	# Examples
	# 	python main.py -d normal -m FC -n FC_normal_200epoch
	# 	python main.py -d normal -m LCN -n LCN_normal_200epoch
	# *************************************************************************

	argList = sys.argv[1:]
	options = "d:tm:n:l:g:"
	longOptions = ["data", "temp", "model", "name", "layers", "gain"]

	try:
		# Parsing argument
		arguments, values = getopt.getopt(argList, options, longOptions)

		# checking each argument
		for curArg, curVal in arguments:

			if curArg in ("-d"):
				dataType = curVal
				# print(f'Data type used is - {dataType}')

			elif curArg in ("-t"):
				print("Using random data to test forward pass pipeline")
				tempData = True

			elif curArg in ("-m"):
				modelType = curVal
				# print(f'Model type used is - {modelType}')

			elif curArg in ("-l"):
				nSpikingLayers = int(curVal)
				# print(f'Num spiking layers is - {nSpikingLayers}')

			elif curArg in ("-n"):
				modelDictName = curVal
				# print(f'Model dict name used is - {modelDictName}')	

			elif curArg in ("-g"):
				gain = int(curVal)
				# print(f'Gain is - {gain}')	

	except getopt.error as err:
		# output error, and return with an error code
		print (str(err))
		return


	# *************************************************************************
	# Get Data
	# *************************************************************************

	# Populate ONV files for testing
	nPhotoreceptors = 14400

	# ********** Rand ONV ********* #
	if tempData is True:		

		nOnv = 1
		if dataType == 'delta':
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

	if dataType == 'normal':
		inputs = prevOnv
	elif dataType == 'delta':
		inputs = curOnv - prevOnv

		# only keep curOnv in file
		with open(onvPath, 'w') as f:
			f.write(curOnvString)
	else:
		print("Error: -d must be passed with 'normal' or 'delta'")


	# *************************************************************************
	# Run Models
	# *************************************************************************
	
	# TODO: check or update LCN and LCNSpiking params if needed
	if modelType == "FC":
		model = FC()
	elif modelType == "LCN":
		model = LCN(nPhotoreceptors, 2, 25, 2, 5, True)
	elif modelType == "LCNChannelStack":
		model = LCNChannelStack(14400, 2, 25, 2, 5, True)
	elif modelType == "LCNSpiking":
		model = LCNSpiking2(nPhotoreceptors, 2, 25, 2, 5, 0, 1, True)
	elif modelType == "LCNSpikingHybrid2":
		model = LCNSpikingHybrid2(nSpikingLayers, 43200, 2, 25, 5, 5, 0, 1, True)
	elif modelType == "LCNSpikingHybrid":
		model = LCNSpikingHybrid(nSpikingLayers, 43200, 2, 25, 5, 5, 0, 1, True)
	else:
		print("Error: -m must be passed with 'FC', 'LCN', or 'LCN Spiking'")

	model.load_state_dict(torch.load(f'{modelDictPath}/{modelDictName}', map_location=device))

	# model is float()ed and sent to device
	model.to(torch.float)
	model.to(device)

	# inputs are converted to tensors, float()ed, processed if needed, and sent to device
	inputs = torch.from_numpy(inputs)
	inputs = inputs.float()

	# TODO: change params if needed
	# Encode Spiking Inputs
	if modelType == "LCNSpiking" or modelType == "LCNSpikingHybrid" or modelType == "LCNSpikingHybrid2":
		rate   = RateEncodeData(nSteps, gain, 0)
		rgb = CopyRedChannel()
		rateRGB = transforms.Compose([rgb, rate])
		inputs = rateRGB(inputs)
	if modelType == "LCNChannelStack":
		onOff  = OnOffChannels(14400)
		inputs = onOff(inputs)
	
	inputs = inputs[None, :]
	inputs = inputs.to(device)

	# run model, get outputs
	output = model(inputs)
	if modelType == "LCNSpiking":
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
	Test delta data
	Test spiking nn
	add logging flag to output info
"""