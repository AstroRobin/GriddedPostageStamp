# GriddedPostageStamp
A script to plot a grid of galaxy "Postage Stamp" cutouts on a gridded histogram over two parameters.

```
>> python PlotImageGrid.py -h

usage: PlotImageGrid.py [-h] [-f FILE] [-x XAXIS] [-y YAXIS]
                        [-c [CONDITION [CONDITION ...]]] [--xlab XLAB]
                        [--ylab YLAB] [-b [NUM [NUM ...]]] [-n NUM]
                        [-o OUTPUT] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  The path to the file containing both the RA/Dec
                        positional data and the plotting axes for the
                        galaxies.
  -x XAXIS, --xaxis XAXIS
                        The name of the colum to plot on the x-axis.
  -y YAXIS, --yaxis YAXIS
                        The name of the colum to plot on the y-axis.
  -c [CONDITION [CONDITION ...]], --conditions [CONDITION [CONDITION ...]]
                        A set of conditions (format:
                        {col_name}[>|>=|=|!=|<=|<]{value}) that must be met by
                        a galaxy to be plotted (e.g. --conditions ba>0.2
                        class=1).
  --xlab XLAB           The label of the x-axis.
  --ylab YLAB           The label of the y-axis.
  -b [NUM [NUM ...]], --bins [NUM [NUM ...]]
                        The number of bins in each axis either as a single
                        number if equal or two values for the x and y-axes,
                        respectively (Defulat: 10)
  -n NUM, --numcut NUM  The threshold number of galaxies that must be in a bin
                        in order to display galaxy image.
  -o OUTPUT, --output OUTPUT
                        The path to the file to save the image grid plot.
  -v                    The level of verbosity to print to stdout.

```

The --file should be a comma-separated file containing (at least) the positional columns named RA & Dec as well as two additional columns (using --xaxis and --yaxis) for the parameters from which the grid will be generated.
