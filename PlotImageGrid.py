import os
import imp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

### Import images from URL
from PIL import Image
import requests
from io import BytesIO

### Progress Bar
try: # Check whether progressbar exists
    imp.find_module('progressbar2')
    from progressbar import ProgressBar, Percentage, Bar, Counter, RotatingMarker, ETA
    pbarFound = True
except(ImportError):
    pbarFound = False


def parse_condition(condStr, colNames):
	"""
	Validates and parses the string representation of a condition into a dictionary format
	
	<param: condStr [String]> - The condition string of the expected format "{colname}{operator}{value}". The "{operator}" must be one of ['>','>=','=','!=,'<=','<'] and the "{value}" must be numeric if operator contains either '<' or '>'.
	<param: colNames [list (Strings)]> - The list of column names in the data table.
	
	<return: cond [dict['col','operator','val']]> - The dictionary representation of the condition.
	"""

	# Check that valid operator was given
	if ('=' in condStr): # check for "=" or ">=" or "<=" or "!="
		if (">" in condStr):
			operator = ">="
		elif ("<" in condStr):
			operator = "<="
		elif ("==" in condStr):
			operator = "=="
		elif ("!=" in condStr):
			operator = "!="
		else:
			operator = "="

	elif (">" in condStr):
		operator = ">"
	elif ("<" in condStr):
		operator = "<"
	else:
		print("\nWARNING: Condition \"{0}\" does not have a valid operator.\n".format(condStr))
		return (None)

	# Check that right side is numeric IF operator contains >|<
	if ("<" in operator or ">" in operator):
		valStr = condStr.split(operator)[-1]
		try:
			val = float(valStr)
		except(ValueError):
			print("\nWARNING: Value \"{0}\" is not numeric for '>' and '<' conditions.\n".format(valStr))
			return (None)
	else:
		val = condStr.split(operator)[-1]

	# Check that the column name exists
	col = condStr.split(operator)[0]
	if (col not in colNames):
		print("\nWARNING: Column name \"{0}\" not in data table.\n".format(col))
		return (None)

	# Return the condition if it was valid, other return None
	return ({"col":col,"operator":operator,"val":val})



### Define Arguments ###
parser = ArgumentParser()
parser.add_argument('-f','--file',
					action='store',dest='inputFile',type=str,default=None,
					help="The path to the file containing both the RA/Dec positional data and the plotting axes for the galaxies.",metavar="FILE")
parser.add_argument('-x','--xaxis',
					action='store',dest='xAxis',type=str,default=None,
					help="The name of the colum to plot on the x-axis.",metavar="XAXIS")
parser.add_argument('-y','--yaxis',
					action='store',dest='yAxis',type=str,default=None,
					help="The name of the colum to plot on the y-axis.",metavar="YAXIS")
parser.add_argument('-c','--conditions',
					action='store',dest='conditions',nargs='*',type=str,default=[],
					help="A set of conditions (format: {col_name}[>|>=|=|!=|<=|<]{value}) that must be met by a galaxy to be plotted (e.g. --conditions ba>0.2 class=1).",metavar="CONDITION")
parser.add_argument('--xlab',
					action='store',dest='xLab',type=str,default=None,
					help="The label of the x-axis.",metavar="XLAB")
parser.add_argument('--ylab',
					action='store',dest='yLab',type=str,default=None,
					help="The label of the y-axis.",metavar="YLAB")
parser.add_argument('-b','--bins',
					action='store',dest='nBins',type=int,nargs='*',default=[10,10],
					help="The number of bins in each axis either as a single number if equal or two values for the x and y-axes, respectively (Defulat: 10)",metavar="NUM")
parser.add_argument('-n','--numcut',
					action='store',dest='numCut',type=int,default=1,
					help="The threshold number of galaxies that must be in a bin in order to display galaxy image.",metavar="NUM")
parser.add_argument('-o','--output',
					action='store',dest='outputFile',type=str,default=None,
					help="The path to the file to save the image grid plot.",metavar="OUTPUT")
parser.add_argument('-v',
					action='count',dest='verbosity',default=0,
					help="The level of verbosity to print to stdout.")

### Parse Arguments ###
args = parser.parse_args()

# Parse verbosity
global vrb;	vrb = args.verbosity

# Validate inputs
if (args.inputFile == None):
	print("\nERROR: No --file argument given! For help, use \"python PlotImageGrid.py -h\".\n  -- ABORTING --\n"); exit()

if (os.path.exists(args.inputFile)):
	inputFile = args.inputFile
else:
	print("\nERROR: --file \"{0}\" down not exist!\n  -- ABORTING --\n".format(args.inputFile)); exit()

# Validate numCut
numCut = max(1,args.numCut)

### Read Table ###
df = pd.read_table(inputFile,sep=',')

nGals = len(df)

colNames = list(df)
if(vrb): print("INFO: colNames: \"{0}\"".format("\", \"".join(colNames)))

idName = colNames[0]

### Check for RA/Dec columns ###
RAName = None; DecName = None
for col in colNames:
	if (col.lower() == 'ra'):
		RAName = col
	elif (col.lower() == 'dec'):
		DecName = col

if (RAName == None): print("\nERROR: No 'RA' column found!\n  -- ABORTING --\n"); exit()
if (DecName == None): print("\nERROR: No 'Dec' column found!\n  -- ABORTING --\n"); exit()


# Check whether x/y-axis names exist
if (args.xAxis in colNames):
	xName = args.xAxis
else:
	print("\nERROR: x-axis name \"{0}\" does not exist in table!\n  -- ABORTING --\n".format(args.xAxis)); exit()

if (args.yAxis in colNames):
	yName = args.yAxis
else:
	print("\nERROR: x-axis name \"{0}\" does not exist in table!\n  -- ABORTING --\n".format(args.yAxis)); exit()


### Validate conditions ###
conditions = [] # A list of dictionaries containing the condition defintions
for condStr in args.conditions:
	cond = parse_condition(condStr,colNames)
	if (cond != None):
		conditions.append(cond)

# Print conditions
if(vrb>0 and len(conditions) > 0):
	print("\nINFO: Specified conditions:")
	for cond in conditions:
		print(" - {0[col]} {0[operator]} {0[val]}".format(cond))


### Mark galaxies as valid based on whether they meet all specified conditions ###
df["valid"] = np.array([True for kk in range(nGals)])

for ii in range(nGals): # Loop over all galaxies
	for cond in conditions: # Check all conditions
		if (cond["operator"] == ">"): # greater-than
			if not df[cond['col']][ii] > cond['val']:
				df["valid"][ii] = False; break
		elif (cond["operator"] == ">="):  # greater-than or equal-to
			if not df[cond['col']][ii] >= cond['val']:
				df["valid"][ii] = False; break
		elif (cond["operator"] == "=" or cond["operator"] == "=="):  # equal-to
			if not df[cond['col']][ii] == cond['val']:
				df["valid"][ii] = False; break
		elif (cond["operator"] == "!="):  # not equal-to
			if not df[cond['col']][ii] != cond['val']:
				df["valid"][ii] = False; break
		elif (cond["operator"] == "<="): # less-than or equal-to
			if not df[cond['col']][ii] <= cond['val']:
				df["valid"][ii] = False; break
		elif (cond["operator"] == "<"): # less-than
			if not df[cond['col']][ii] < cond['val']:
				df["valid"][ii] = False; break

# Print number of galaxies that match all conditions
if(vrb>1 and len(args.conditions) > 0):
	print("INFO: Number of entries that meet specified conditions: {0}/{1}".format(np.sum(df["valid"]),nGals))


### Bin Data in x/y axes ###
if (len(args.nBins) == 1): # If one length, i.e. square grid was specified
	xNum = args.nBins
	yNum = args.nBins
elif (len(args.nBins) > 1): # If two lengths specified
	if (len(args.nBins) > 2): print("\nWARNING: Too many values for --nbins specified!\n")
	xNum = args.nBins[0]
	yNum = args.nBins[1]

# Create the bin edges
xBins = np.linspace(min(df[xName]),max(df[xName]),num=xNum+1) 
yBins = np.linspace(min(df[yName]),max(df[yName]),num=yNum+1)

# Get bin mids
xMids = 0.5*(xBins[:-1]+xBins[1:])
yMids = 0.5*(yBins[:-1]+yBins[1:])


### Initialise the index array. ###
indices = {}
for xx in range(xNum):
	for yy in range(yNum):
		indices[xx,yy] = []

# Fill index array: This is a dictionary of the bins where each contains an array of rows indices that fall into that bin
for ii in range(nGals):
	if (df["valid"][ii] == True):
		for xx in range(0,xNum):
			for yy in range(0,yNum):
				if (xx == 0):
					if (df[xName][ii] >= xBins[xx] and df[xName][ii] <= xBins[xx+1]):
						if (yy == 0):
							if (df[yName][ii] >= yBins[yy] and df[yName][ii] <= yBins[yy+1]):
								indices[xx,yy].append(ii)
								break
						else:			
							if (df[yName][ii] > yBins[yy] and df[yName][ii] <= yBins[yy+1]):
								indices[xx,yy].append(ii)
								break
				else:				
					if (df[xName][ii] > xBins[xx] and df[xName][ii] <= xBins[xx+1]):
						if (yy == 0):
							if (df[yName][ii] >= yBins[yy] and df[yName][ii] <= yBins[yy+1]):
								indices[xx,yy].append(ii)
								break
						else:			
							if (df[yName][ii] > yBins[yy] and df[yName][ii] <= yBins[yy+1]):
								indices[xx,yy].append(ii)
								break

			# The "for, else, continue, break" sequence allows one to break out of a double for loop
			else: # if inner loop was not broken continue to next outer loop iteration
				continue

			break # Inner loop was broken, therefore break outer loop also.


### Get the histogram of entries in each bin ###
hist = np.empty([xNum,yNum])
for xx in range(xNum):
	for yy in range(yNum):
		hist[xx,yy] = len(indices[xx,yy])

hist = hist.T # invert the histogram matrix for plotting

# The physical length (in axis units) of each x/y bin
xBinSize = abs(xBins[-1]-xBins[0])/xNum
yBinSize = abs(yBins[-1]-yBins[0])/yNum

if(vrb>0): print("INFO: x-axis bin size: {0:.3f}".format(xBinSize))
if(vrb>0):  print("INFO: y-axis bin size: {0:.3f}".format(yBinSize))

# Get the require aspect ratio of the plot to ensure square grids.
asp = xBinSize/yBinSize
if(vrb>0): print("INFO: Aspect Ratio = {0:.2f}".format(asp))


### Plot a histogram of the points for diagnostics requires "-vvv" ###
if (vrb>2):
	print("INFO: Displaying galaxy indices in each bin:")
	for xx in range(xNum):
		for yy in range(yNum):
			print("\n",xx,yy,": ",end="")
			for ii in indices[xx,yy]:
				print(ii,end=' ')
	print("\n")


	fig0, axes0 = plt.subplots(figsize=(xNum,yNum))

	X, Y = np.meshgrid(xBins,yBins)
	grid = axes0.pcolormesh(X, Y, hist, cmap="magma")

	fig0.colorbar(grid, ax=axes0)

	#valid = np.array(np.argwhere(df['valid'] == True).T[0])
	#notValid = np.argwhere(df['valid'] == False).T[0]

	#print(df['valid'][valid[0]])
	axes0.plot(df.loc[df['valid'] == False][xName],df.loc[df['valid'] == False][yName],
			  linestyle='',marker='o',markersize=8,markeredgewidth=1.5,markeredgecolor='darkred',color='lightcoral',alpha=0.8)
	axes0.plot(df.loc[df['valid'] == True][xName],df.loc[df['valid'] == True][yName],
			  linestyle='',marker='o',markersize=8,markeredgewidth=1.5,markeredgecolor='darkblue',color='royalblue',alpha=0.8)


	axes0.set_xlabel(args.xLab if args.xLab != None else xName,fontsize=18)
	axes0.set_ylabel(args.yLab if args.yLab != None else yName,fontsize=18)

	axes0.tick_params(labelsize=16)

	axes0.set_xlim(xBins[0],xBins[-1])
	axes0.set_ylim(yBins[0],yBins[-1])

	axes0.set_aspect(asp)

	plt.show()



## Determine plotted image properties
numImages = len(np.where(hist>=numCut)[0])
if(vrb>0): print("INFO: Minimum num. galaxies required per bin: {0}".format(numCut))
if(vrb>0): print("INFO: Total number of galaxies to download: {0}".format(numImages))

## Define SDSS image properties
imgSize = 250
imgScale = (1.5*60.0)/imgSize

fig, axes = plt.subplots(figsize=(xNum,yNum), dpi=250)


## Set up Progress Bar:
if (vrb>0 and pbarFound): 
	pbar = ProgressBar(widgets=["Inserting SDSS RGB images: ",Percentage()," ", Bar(marker=RotatingMarker())," ",ETA()], maxval=numImages).start()

imgIndices = np.empty([xNum,yNum],dtype='int')

### Loop through bins and display SDSS images ###
count = 0
for xx in range(0,xNum):
	for yy in range(0,yNum):
		if (len(indices[xx,yy]) >= numCut):
			index = indices[xx,yy][np.random.randint(len(indices[xx,yy]))]
			imgIndices[xx,yy] = index # update the currently displayed galaxy indices

			# Read in the image data from the URL
			imgURL = "http://skyservice.pha.jhu.edu/DR10/ImgCutout/getjpeg.aspx?ra={ra}&dec={dec}&width={size}&height={size}&scale={scale}".format(ra=df[RAName][index],dec=df[DecName][index],size=imgSize,scale=imgScale)
			response = requests.get(imgURL)
			img = Image.open(BytesIO(response.content))

			# Plot the image
			axes.imshow(img,aspect='auto',extent=(xBins[xx], xBins[xx] + xBinSize, yBins[yy], yBins[yy] + yBinSize), zorder=-1)

			count+=1
			if (vrb>0 and pbarFound): pbar.update(count) # update the progress bar

# Conclude the progress bar
if (vrb>0 and pbarFound): pbar.finish()


#cid = fig.canvas.mpl_connect('button_press_event', onclick)

for xx in xBins: axes.axvline(x=xx,linewidth=0.25,color='dimgrey',zorder=3)
for yy in yBins: axes.axhline(y=yy,linewidth=0.25,color='dimgrey',zorder=3)

axes.set_xlabel(args.xLab if args.xLab != None else xName,fontsize=20)
axes.set_ylabel(args.yLab if args.yLab != None else yName,fontsize=20)

axes.tick_params(labelsize=16)

axes.set_xlim(xBins[0],xBins[-1])
axes.set_ylim(yBins[0],yBins[-1])

xBinSize = abs(xBins[-1]-xBins[0])/xNum
yBinSize = abs(yBins[-1]-yBins[0])/yNum
axes.set_aspect(xBinSize/yBinSize)

axes.set_facecolor('black')

if (args.outputFile != None):
	plt.savefig(args.outputFile,bbox_inches='tight',pad_inches=2)


plt.show()

