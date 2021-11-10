# RunSNN
Code to interface with a visual studio 10 project

# What it does
* this executable is meant to be run from the main eye code
* we deal with two csv files, onvOut and resultOut
	* both should be in the same directory, which is specified in the main
* read in ONV from a csv file
* run either an ANN or SNN
* write the result to file

# Compile to exe
* pip install pyinstaller
* open cmd as admin
* pyinstaller main.py
* run app from \dist\main\main.exe