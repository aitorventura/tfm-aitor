import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image


from time import time
import scipy.io
import cv2
np.seterr('ignore')

class GenLogPolar:
    # 23/3/21: A bit more generalized logpolar transform
    # It does not assume the center of the transform (x0,y0) to be the center of the cartesian image
    # (x0,y0) can be anywhere in the cartesian image
    # rhoMax is given by the max size possible for the cartesian image (i.e. the length of the diagonal)
    # If rhoMax is given as an argument (optional), the minimum is used
    def __init__(self, M, N, x0, y0, rho0, R=None, S=None, rhoMax=None):
        assert M % 2 == 0 and N % 2 == 0, "LogPolar: Either the width or height of the image is odd. Try with even lengths"
        self.M = M  # rows
        self.N = N  # columns
        self.rho0 = rho0
        assert 0 <= y0 < M and 0 <= x0 < N, "center of transform should lie within image bounds"
        self.x0, self.y0 = x0, y0
        self.rhoMax = np.sqrt(M ** 2 + N ** 2)  # diagonal, to cover the image regardless of where (x0,y0) is

        # if rhoMax is not None:
        # self.rhoMax = min(rhoMax,self.rhoMax)
        if R is None or S is None:
            R, S = computeLPsize(rho0, rhoMax, maxOversampling=2)

        self.R = R
        self.S = S
        self.bVerbose = False
        self.bSolveUnmapped = True
        self.computeMap()

    def computeMap(self):

        [xx, yy] = np.meshgrid(np.arange(self.N) - self.x0, np.arange(self.M) - self.y0)
        self.a = np.exp(np.log(self.rhoMax / self.rho0) / self.R)

        # U and V are the mapping from cartesian coordinates (i,j) to their corresponding logpolar coordinates (u,v)
        # U[i,j] = u, V[i,j] = v, 0<=i<M, 0<=j<N
        rho = np.sqrt(xx ** 2 + yy ** 2)
        self.U = np.floor(np.log(rho / self.rho0) / np.log(self.a)).astype(int)

        self.jjOutOfRange, self.iiOutOfRange = np.where(np.logical_or(self.U < 0, self.U >= self.R))
        if self.bVerbose:
            print("num out of range", self.iiOutOfRange.shape, "num pix", self.M * self.N)
        self.U[self.jjOutOfRange, self.iiOutOfRange] = np.floor(np.minimum(self.R - 1, np.maximum(0, np.log(
            rho[self.jjOutOfRange, self.iiOutOfRange] / self.rho0) / np.log(self.a)))).astype(int)
        self.q = self.S / (2.0 * math.pi)
        self.V = np.floor(np.minimum(self.S - 1, np.maximum(0, self.q * (np.arctan2(yy, xx) + math.pi)))).astype(int)

        # this 2d histogramming seems to be the part most computationally costly. Can we do better?
        self.nUVs = np.histogram2d(self.U.ravel(), self.V.ravel(), bins=(range(self.R + 1), range(self.S + 1)))[0]
        self.nUVs[self.nUVs == 0] = 1  # replace 0s by 1s to avoid division by 0 later on
        # nUs = np.histogram(self.u,range(self.R))[0]
        # nVs = np.histogram(self.v,range(self.S))[0]
        '''bVerbose=False
        if bVerbose:
            print "nUs",nUs.shape,nUs
            print "nVs",nVs.shape,nVs
            print "nUVs",nUVs.shape,nUVs
            print nUVs[0,:], np.max(nUVs)
            #        print nUVs[self.R-1,:]
        '''

        if self.bSolveUnmapped:
            self.emptyUU, self.emptyVV = np.where(self.nUVs == 0)
            rho = self.rho0 * self.a ** self.emptyUU
            theta = self.emptyVV / self.q + math.pi
            self.iForEmptyUV = (rho * np.cos(theta) + self.x0).astype(int)
            self.jForEmptyUV = (rho * np.sin(theta) + self.y0).astype(int)
            plt.show(block=True)
            if False:  # self.bVerbose:
                print("empty uu,vv", self.emptyUU, self.emptyVV)
                print("empty ii,jj", self.iForEmptyUV, self.jForEmptyUV)

        if self.bVerbose:
            print("Num empty uv's", np.sum(self.nUVs == 0))
            plt.imshow(np.log(self.nUVs[0:, 0:]), interpolation='none', cmap='gray')
            plt.title('Number of cartesian pixels mapped to each (u,v)')
            plt.show(block=True)

        if self.bVerbose:
            print(self.V.max(), self.V.min())
            print(self.U.max(), self.U.min())
            print(xx.shape, yy.shape)
            print(self.a, self.q)
            plt.imshow(xx, interpolation='none', cmap='gray')
            plt.show(block=True)
            plt.imshow(yy, interpolation='none', cmap='gray')
            plt.show(block=True)
            plt.imshow(self.U, interpolation='none', cmap='gray')
            plt.show(block=True)
            plt.imshow(self.V, interpolation='none', cmap='gray')
            plt.show(block=True)

    # logpolar mapping from input cartesian image im (C->LP conversion)
    def lp(self, im):
        # print type(im[0][0])
        # print "im:",im.mean(),im.min(),im.max()
        assert im.shape == (self.M, self.N), "lp: Input image has size " + str(
            im.shape) + ", which is different to expected " + str(self.M) + "x" + str(self.N)
        lpim = np.zeros((self.R, self.S))
        # print "sizes:", self.U.shape, im.shape

        np.add.at(lpim, (self.U, self.V), im)
        lpim /= self.nUVs
        if self.bSolveUnmapped:
            lpim[self.emptyUU, self.emptyVV] = im[self.jForEmptyUV, self.iForEmptyUV]
        return lpim

    # inverse logpolar mapping from input logpolar image lpim (LP->C conversion)
    def ilp(self, lpim, fillValue=None):
        # print "lpim:",lpim.mean(),lpim.min(),lpim.max()
        # print(lpim.shape,self.R,self.S)
        assert lpim.shape == (self.R, self.S), "ilp: Input image has size different to expected"
        im = np.zeros((self.M, self.N))
        # print type(im[0][0])
        im = lpim[self.U, self.V]
        if fillValue is None:
            valueForUnmappedPixels = 255 if type(im[0][0]) == 'numpy.uint8' else 100
        else:
            valueForUnmappedPixels = fillValue
        im[self.jjOutOfRange, self.iiOutOfRange] = valueForUnmappedPixels
        #        np.add.at(im,(self.u,self.v),lpim)
        return im

    def report(self):
        print("Cartesian image: ", self.M, "x", self.N)
        print("Log-polar image: ", self.R, "x", self.S, "(rings x sectors)")
        print("rho min:", self.rho0)
        print("rho max:", self.rhoMax)

    def getLPSize(self):
        return self.R, self.S

    def getCartSize(self):
        return self.M, self.N


def computeLPsize(rho0, rhoMax, maxOversampling, bShortReturn=True):
    bDisplay = True
    # derive the number of rings (R) and sectors (S) of a logpolar image
    # to meet two design criteria: (1) unit aspect ratio and (2) maximum given oversampling
    # (and forcing rho0 and assuming rhoMax is given by the size of the cartesian image)
    rho0sq = rho0 ** 2
    coeff3 = rho0sq
    coeff2 = -rho0sq
    coeff1 = -rho0sq
    coeff0 = rho0sq - 2.0 / maxOversampling
    a_candidate = np.roots([coeff3, coeff2, coeff1, coeff0])  # solve degree-3 polynomial for a
    if False:
        print("a candidate", a_candidate)

    a = min(a_candidate[a_candidate >= 1])
    S = int(np.round(2 * np.pi / (a - 1)))  # unit aspect ratio
    R = int(np.round(np.log(rhoMax / rho0) / np.log(a)))

    oversamp = S / (np.pi * rho0sq * (a ** 2 - 1))
    nLP = R * S
    nC = (rhoMax * 2) ** 2  # assuming squared input cartesian image and transformation covering all its extent
    compression_ratio = float(nLP) / nC
    aspect_ratio = 2.0 * np.pi / (S * (a - 1.0))
    if bDisplay:
        print("growth rate (a)", a)
        print("aspect ratio", aspect_ratio)
        print("resulting oversampling", oversamp)
        print("R=", R)
        print("S=", S)
        print("Num LP pixels", nLP)
        print("Num Cart pixels", nC)
        print("compression ratio (how large LP image is wrt to C) (%)", 100 * compression_ratio)
        print("compression ratio (how large C is wrt to LP) (x times)", 1.0 / compression_ratio)
        print("rho0", rho0)
        print("rhoMax", rhoMax)
    if bShortReturn:
        return (R, S)
    else:
        return (R, S), oversamp, compression_ratio, aspect_ratio


bTiming = False
# for timing, run ipython from console
# $ ipython
# and at the ipython's prompt, run
# $ run -i logPolarFunction.py

if bTiming:
    from IPython import get_ipython

    ipython = get_ipython()


def test_generalLP():
    bColor = True
    if bColor:
        # imfile='lena.png'
        imfile = 'color.jpg'

    else:
        imfile = 'lena.pgm'

    im_pil = Image.open(imfile)  # .convert('L')
    im = np.array(im_pil)

    if bColor:
        M, N, C = im.shape
    else:
        M, N = im.shape
        C = 1

    print(M, N, C)
    x0, y0 = N / 2, M / 2  # N-1,0 #N-1,M-1 # 0,M-1,
    rho0 = 5.0
    R, S = 30, 60
    rhoMax = min(M, N) / 2
    GLP = GenLogPolar(M, N, x0, y0, rho0, R, S, rhoMax)

    if bColor:
        lp_im = np.zeros((R, S, C))
        for c in range(C):
            lp_im[:, :, c] = GLP.lp(im[:, :, c])
    else:
        lp_im = GLP.lp(im)

    plt.imshow(lp_im / 255, cmap='gray')
    plt.show(block=True)

    if bColor:
        c_im = np.zeros_like(im)
        for c in range(C):
            c_im[:, :, c] = GLP.ilp(lp_im[:, :, c])
    else:
        c_im = GLP.ilp(lp_im)

    plt.imshow(c_im, cmap='gray')
    plt.show(block=True)
