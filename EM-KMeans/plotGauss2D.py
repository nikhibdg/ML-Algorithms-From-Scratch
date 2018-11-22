import pdb
import math
import pylab as pl
from matplotlib.patches import Ellipse

def plotGauss2D(pos, P, r = 2., color = 'black'):
    U, s , Vh = pl.linalg.svd(P)
    print(U, s, Vh)
    orient = math.atan2(U[1,0],U[0,0])*180/math.pi
    ellipsePlot = Ellipse(xy=pos, width=2*r*math.sqrt(s[0]),
              height=2*r*math.sqrt(s[1]), angle=orient,
              edgecolor=color, fill = False, lw = 3, zorder = 10)
    ax = pl.gca()
    ax.add_patch(ellipsePlot);
    return ellipsePlot

