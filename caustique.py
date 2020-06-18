#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def init(args=[], ds=1):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default='2020-06-18_caustique', help="Tag")
    parser.add_argument("--nx", type=int, default=1500//ds, help="number of pixels (vertical)")
    parser.add_argument("--ny", type=int, default=2400//ds, help="number of pixels (horizontal)")
    parser.add_argument("--bin_dens", type=int, default=2, help="relative bin density")
    parser.add_argument("--nframe", type=int, default=120, help="number of frames")
    parser.add_argument("--seed", type=int, default=42, help="seed for RNG")
    parser.add_argument("--H", type=float, default=125., help="depth")
    parser.add_argument("--sf_0", type=float, default=0.002, help="sf")
    parser.add_argument("--B_sf", type=float, default=0.001, help="bandwidth in sf")
    parser.add_argument("--V_Y", type=float, default=0.0, help="horizontal speed")
    parser.add_argument("--V_X", type=float, default=0.5, help="vertical speed")
    parser.add_argument("--B_V", type=float, default=2.0, help="bandwidth in speed")
    parser.add_argument("--theta", type=float, default=np.pi/2, help="angle with the horizontal")
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
        self.ratio = opt.ny/opt.nx # ratio between height and width (>1 for portrait, <1 for landscape)
        X = np.linspace(0, 1, opt.nx, endpoint=False) # vertical
        Y = np.linspace(0, self.ratio, opt.ny, endpoint=False) # horizontal
        self.xv, self.yv = np.meshgrid(X, Y, indexing='ij')
        self.opt = opt
        # https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-python-argparse-namespace-as-a-dictionary
        self.d = vars(opt)

    def wave(self):
        import MotionClouds as mc
        fx, fy, ft = mc.get_grids(self.opt.nx, self.opt.ny, self.opt.nframe)
        env = mc.envelope_gabor(fx, fy, ft, V_X=self.opt.V_Y, V_Y=self.opt.V_X, B_V=self.opt.B_V,
                                sf_0=self.opt.sf_0, B_sf=self.opt.B_sf, theta=self.opt.theta, B_theta=self.opt.B_theta)
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
            os.makedirs(self.opt.tag, exist_ok=True)
            os.makedirs(f'/tmp/{self.opt.tag}', exist_ok=True)
            gifname=f'{self.opt.tag}/{self.opt.tag}.gif'
        binsx, binsy = self.opt.nx//self.opt.bin_dens, self.opt.ny//self.opt.bin_dens
        hist = np.zeros((binsx, binsy, self.opt.nframe))
        for i_frame in range(self.opt.nframe):
            xv, yv = self.transform(z[:, :, i_frame])
            hist[:, :, i_frame], edge_x, edge_y = np.histogram2d(xv.ravel(), yv.ravel(), bins=[binsx, binsy], density=True)

        hist /= hist.max()
        subplotpars = matplotlib.figure.SubplotParams(left=0., right=1., bottom=0., top=1., wspace=0., hspace=0.,)
        fnames = []
        for i_frame in range(self.opt.nframe):
            fig, ax = plt.subplots(figsize=(binsy/dpi, binsx/dpi), subplotpars=subplotpars)
            ax.pcolormesh(edge_x, edge_y, hist[:, :, i_frame].T, vmin=0, vmax=1, cmap=plt.cm.Blues_r)
            fname = f'/tmp/{gifname}_frame_{i_frame}.png'
            fig.savefig(fname, dpi=dpi)
            fnames.append(fname)
            plt.close('all')

        return make_gif(gifname, fnames, fps=self.opt.fps)
