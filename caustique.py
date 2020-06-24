#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def init(args=[], ds=1):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default='caustique', help="Tag")
    parser.add_argument("--figpath", type=str, default='2020-06-19_caustique', help="Folder to store images")
    parser.add_argument("--nx", type=int, default=5*2**8, help="number of pixels (vertical)")
    parser.add_argument("--ny", type=int, default=8*2**8, help="number of pixels (horizontal)")
    parser.add_argument("--bin_dens", type=int, default=4, help="relative bin density")
    parser.add_argument("--nframe", type=int, default=2**7, help="number of frames")
    parser.add_argument("--seed", type=int, default=42, help="seed for RNG")
    parser.add_argument("--H", type=float, default=20., help="depth of the pool")
    parser.add_argument("--sf_0", type=float, default=0.004, help="sf")
    parser.add_argument("--B_sf", type=float, default=0.002, help="bandwidth in sf")
    parser.add_argument("--V_Y", type=float, default=0.3, help="horizontal speed")
    parser.add_argument("--V_X", type=float, default=0.3, help="vertical speed")
    parser.add_argument("--B_V", type=float, default=2.0, help="bandwidth in speed")
    parser.add_argument("--theta", type=float, default=2*np.pi*(2-1.61803), help="angle with the horizontal")
    parser.add_argument("--B_theta", type=float, default=np.pi/12, help="bandwidth in theta")
    parser.add_argument("--fps", type=float, default=18, help="bandwidth in theta")
    parser.add_argument("--verbose", type=bool, default=False, help="Displays more verbose output.")

    opt = parser.parse_args(args=args)

    if opt.verbose:
        print(opt)
    return opt

def make_gif(gifname, fnames, fps):
    import imageio

    with imageio.get_writer(gifname, mode='I', fps=fps) as writer:
        for fname in fnames:
            writer.append_data(imageio.imread(fname))

    from pygifsicle import optimize
    optimize(str(gifname))
    return gifname


class Caustique:
    def __init__(self, opt):
        """
        Image coordinates follow 'ij' indexing, that is, 
        * their origin at the top left, 
        * the X axis is vertical and goes "down",
        * the Y axis is horizontal and goes "right".
        
        """
        self.ratio = opt.ny/opt.nx # ratio between height and width (>1 for portrait, <1 for landscape)
        X = np.linspace(0, 1, opt.nx, endpoint=False) # vertical
        Y = np.linspace(0, self.ratio, opt.ny, endpoint=False) # horizontal
        self.xv, self.yv = np.meshgrid(X, Y, indexing='ij')
        self.opt = opt
        # https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-python-argparse-namespace-as-a-dictionary
        self.d = vars(opt)

    def wave(self):
        # A simplistic model of a wave using https://github.com/NeuralEnsemble/MotionClouds
        import MotionClouds as mc
        fx, fy, ft = mc.get_grids(self.opt.nx, self.opt.ny, self.opt.nframe)
        env = mc.envelope_gabor(fx, fy, ft, V_X=self.opt.V_Y, V_Y=self.opt.V_X, B_V=self.opt.B_V,
                                sf_0=self.opt.sf_0, B_sf=self.opt.B_sf, 
                                theta=self.opt.theta, B_theta=self.opt.B_theta)
        z = mc.rectif(mc.random_cloud(env, seed=self.opt.seed))
        return z

    def transform(self, z_):
        xv, yv = self.xv.copy(), self.yv.copy()
        
        dzdx = z_ - np.roll(z_, 1, axis=0)
        dzdy = z_ - np.roll(z_, 1, axis=1)
        xv = xv + self.opt.H * dzdx
        yv = yv + self.opt.H * dzdy

        xv = np.mod(xv, 1)
        yv = np.mod(yv, self.ratio)

        return xv, yv

    def plot(self, z, gifname=None, dpi=150):
        if gifname is None:
            os.makedirs(self.opt.figpath, exist_ok=True)
            gifname=f'{self.opt.figpath}/{self.opt.tag}.gif'
        binsx, binsy = self.opt.nx//self.opt.bin_dens, self.opt.ny//self.opt.bin_dens

        hist = np.zeros((binsx, binsy, self.opt.nframe))
        for i_frame in range(self.opt.nframe):
            xv, yv = self.transform(z[:, :, i_frame])
            hist[:, :, i_frame], edge_x, edge_y = np.histogram2d(xv.ravel(), yv.ravel(), 
                                                                 bins=[binsx, binsy], 
                                                                 range=[[0, 1], [0, self.ratio]], 
                                                                 density=True)

        hist /= hist.max()
        subplotpars = matplotlib.figure.SubplotParams(left=0., right=1., bottom=0., top=1., wspace=0., hspace=0.,)
        fnames = []
        for i_frame in range(self.opt.nframe):
            fig, ax = plt.subplots(figsize=(binsy/dpi, binsx/dpi), subplotpars=subplotpars)
            ax.pcolormesh(edge_y, edge_x, hist[:, :, i_frame], vmin=0, vmax=1, cmap=plt.cm.Blues_r)
            fname = f'/tmp/{self.opt.tag}_frame_{i_frame}.png'
            fig.savefig(fname, dpi=dpi)
            fnames.append(fname)
            plt.close('all')

        return make_gif(gifname, fnames, fps=self.opt.fps)
