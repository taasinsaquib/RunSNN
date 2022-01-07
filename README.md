# RunSNN
Code to interface with a visual studio 10 project

# What it does
* this executable is meant to be run from the main eye code
* we deal with two csv files, `onvOut` and `resultOut`
	* both should be in the same directory (`files`), which is specified in the main
* `onvOut` has 2 lines
	* first line is the previous ONV
	* second line is the current ONV 
* the two angles, theta and phi, are written to `resultOut` at the end

## Steps
* subtract previous ONV from current to get "delta ONV"
	* delete the previous ONV from the file so it now has one line
	* the eye code will write a new ONV next time this code is called
* run either an ANN or SNN with delta ONV as input
* write the result to file

# Generate .exe

## Setup Virtual Env
* `pip install -r requirements.txt`
* You'll probably get an error about: "Could not find a version that satisfies the requirement torch==1.10.0+cu113"
	* install torch from this link: https://pytorch.org/get-started/locally/
		* or just run this command `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`

## Compile

### pyinstaller
* open cmd as admin (on Windows)
	* navigate to this directory and activate the virtual environment that you set up
	* for me, `conda activate exe`
* run `pyinstaller main.py`
	* this might fail if antivirus software doesn't allow permission to access certain files
	* it usually works if you delete the build and dist folders and run the command again
* executable will be in `\dist\main\main.exe`

### cx_freeze
* to install: `pip install --upgrade cx_Freeze`
* create setup.py
