#!/usr/bin/env python
# coding: utf-8
import os
import datetime
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from lambda2color import Lambda2color, xyz_from_xy

def init(args=[], ds=1, PRECISION=7):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default='caustique', help="Tag")
    parser.add_argument("--figpath", type=str, default='2021-12-01', help="Folder to store images")
    parser.add_argument("--nx", type=int, default=5*2**PRECISION, help="number of pixels (vertical)")
    parser.add_argument("--ny", type=int, default=8*2**PRECISION, help="number of pixels (horizontal)")
    parser.add_argument("--nframe", type=int, default=5*2**PRECISION, help="number of frames")
    parser.add_argument("--bin_dens", type=int, default=2, help="relative bin density")
    parser.add_argument("--seed", type=int, default=42, help="seed for RNG")
    parser.add_argument("--H", type=float, default=10., help="depth of the pool")
    parser.add_argument("--sf_0", type=float, default=0.004, help="sf")
    parser.add_argument("--B_sf", type=float, default=0.002, help="bandwidth in sf")
    parser.add_argument("--V_Y", type=float, default=0.3, help="horizontal speed")
    parser.add_argument("--V_X", type=float, default=0.3, help="vertical speed")
    parser.add_argument("--B_V", type=float, default=4.0, help="bandwidth in speed")
    parser.add_argument("--theta", type=float, default=2*np.pi*(2-1.61803), help="angle with the horizontal")
    parser.add_argument("--B_theta", type=float, default=np.pi/3, help="bandwidth in theta")
    parser.add_argument("--min_lum", type=float, default=.2, help="diffusion level for the rendering")
    parser.add_argument("--fps", type=float, default=18, help="frames per second")
    parser.add_argument("--multispectral", type=bool, default=True, help="Compute caustics on the full spectrogram.")
    parser.add_argument("--cache", type=bool, default=True, help="Cache intermediate output.")
    parser.add_argument("--verbose", type=bool, default=False, help="Displays more verbose output.")

    opt = parser.parse_args(args=args)

    if opt.verbose:
        print(opt)
    return opt

def make_gif(gifname, fnames, fps, do_delete=True):
    import imageio

    with imageio.get_writer(gifname, mode='I', fps=fps) as writer:
        for fname in fnames:
            writer.append_data(imageio.imread(fname))

    from pygifsicle import optimize
    optimize(str(gifname))
    if do_delete: 
        for fname in fnames: os.remove(fname)
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
        os.makedirs(self.opt.figpath, exist_ok=True)
        self.cachepath = os.path.join('/tmp', self.opt.figpath)
        if opt.verbose: print(f'{self.cachepath=}')
        os.makedirs(self.cachepath, exist_ok=True)

        # a standard white:
        illuminant_D65 = xyz_from_xy(0.3127, 0.3291), 
        illuminant_sun = xyz_from_xy(0.325998, 0.335354)
        # color conversion class
        self.cs_srgb = Lambda2color(red=xyz_from_xy(0.64, 0.33),
                               green=xyz_from_xy(0.30, 0.60),
                               blue=xyz_from_xy(0.15, 0.06),
                               white=illuminant_sun)
    def wave(self):
        filename = f'{self.cachepath}/{self.opt.tag}_wave.npy'
        if os.path.isfile(filename):
            z = np.load(filename)
        else:
            # A simplistic model of a wave using https://github.com/NeuralEnsemble/MotionClouds
            import MotionClouds as mc
            fx, fy, ft = mc.get_grids(self.opt.nx, self.opt.ny, self.opt.nframe)
            env = mc.envelope_gabor(fx, fy, ft, V_X=self.opt.V_Y, V_Y=self.opt.V_X, B_V=self.opt.B_V,
                                    sf_0=self.opt.sf_0, B_sf=self.opt.B_sf,
                                    theta=self.opt.theta, B_theta=self.opt.B_theta)
            z = mc.rectif(mc.random_cloud(env, seed=self.opt.seed))
            if self.opt.cache: np.save(filename, z)
        return z

    def transform(self, z_, modulation=1.):
        xv, yv = self.xv.copy(), self.yv.copy()

        dzdx = z_ - np.roll(z_, 1, axis=0)
        dzdy = z_ - np.roll(z_, 1, axis=1)
        xv = xv + modulation * self.opt.H * dzdx
        yv = yv + modulation * self.opt.H * dzdy

        xv = np.mod(xv, 1)
        yv = np.mod(yv, self.ratio)

        return xv, yv

    def do_raytracing(self, z):
        filename = f'{self.cachepath}/{self.opt.tag}_hist.npy'
        if os.path.isfile(filename):
            hist = np.load(filename)
        else:
            binsx, binsy = self.opt.nx//self.opt.bin_dens, self.opt.ny//self.opt.bin_dens
        
            if self.opt.multispectral:
                N_wavelengths = len(self.cs_srgb.cmf[:, 0])

                # http://www.philiplaven.com/p20.html
                # 1.40 at 400 nm and 1.37 at 700nm makes a 2% variation
                variation = .02
                variation = .05
                variation = .15
                variation = .40
                
                hist = np.zeros((binsx, binsy, self.opt.nframe, N_wavelengths))
                for i_wavelength in range(N_wavelengths):
                    modulation = 1. + variation/2 - variation*i_wavelength/N_wavelengths
                    # print(i_wavelength, N_wavelengths, modulation)
                    for i_frame in range(self.opt.nframe):
                        xv, yv = self.transform(z[:, :, i_frame], modulation=modulation)
                        hist_, edge_x, edge_y = np.histogram2d(xv.ravel(), yv.ravel(),
                                                               bins=[binsx, binsy],
                                                               range=[[0, 1], [0, self.ratio]],
                                                               density=True)
                        hist[:, :, i_frame, i_wavelength] = hist_
                hist /= hist.max()
            else:
                hist = np.zeros((binsx, binsy, self.opt.nframe))
                for i_frame in range(self.opt.nframe):
                    xv, yv = self.transform(z[:, :, i_frame])
                    hist_, edge_x, edge_y = np.histogram2d(xv.ravel(), yv.ravel(),
                                                           bins=[binsx, binsy],
                                                           range=[[0, 1], [0, self.ratio]],
                                                           density=True)
                #hist /= hist.max()
            if self.opt.cache: np.save(filename, hist)
        return hist
    

    def plot(self, z, do_color=True, gifname=None, dpi=150):
        hist = self.do_raytracing(z)

        if gifname is None:
            gifname=f'{self.opt.figpath}/{self.opt.tag}.gif'

        subplotpars = matplotlib.figure.SubplotParams(left=0., right=1., bottom=0., top=1., wspace=0., hspace=0.,)

        if self.opt.multispectral:
            # multiply by the spectrum of the sky
            if False:
                wavelengths = self.cs_srgb.cmf[:, 0]*1e-9
                intensity5800 = planck(wavelengths, 5800.)
                scatter = scattering(wavelengths)
                spectrum_sky = intensity5800 * scatter
                hist = hist * spectrum_sky[None, None, :, None]
                hist /= hist.max()

            # some magic to only get the hue
            #print(hist.shape)
            #for i_frame in range(self.opt.nframe):
            #    hist[:, :, :, i_frame] /= hist[:, :, :, i_frame].max(axis=2)[:, :, None]
            #hist -= hist.min()
            hist /= hist.max()
            image_rgb = self.cs_srgb.spec_to_rgb(hist)
            image_rgb -= image_rgb.min()
            image_rgb /= image_rgb.max()


        fnames = []
        for i_frame in range(self.opt.nframe):
            fig, ax = plt.subplots(figsize=(self.opt.nx/self.opt.bin_dens/dpi, self.opt.ny/self.opt.bin_dens/dpi), subplotpars=subplotpars)
            if self.opt.multispectral:
                
                ax.imshow(image_rgb[:, :, i_frame] ** (1/1.61803), vmin=0, vmax=1)
            else:
                if do_color:
                    bluesky = np.array([0.268375, 0.283377]) # xyz
                    sun = np.array([0.325998, 0.335354]) # xyz
                    # ax.pcolormesh(edge_y, edge_x, hist[:, :, i_frame], vmin=0, vmax=1, cmap=plt.cm.Blues_r)
                    # https://en.wikipedia.org/wiki/CIE_1931_color_space#Mixing_colors_specified_with_the_CIE_xy_chromaticity_diagram
                    L1 = 1 - hist[:, :, i_frame]
                    L2 = hist[:, :, i_frame]
                    image_denom = L1 / bluesky[1] + L2 / sun[1]
                    image_x = (L1 * bluesky[0] / bluesky[1] + L2 * sun[0] / sun[1]) / image_denom
                    image_y = (L1 + L2) / image_denom 
                    image_xyz = np.dstack((image_x, image_y, 1 - image_x - image_y))
                    image_rgb = self.cs_srgb.xyz_to_rgb(image_xyz)
                    image_L = self.opt.min_lum + (1-self.opt.min_lum)* L2 ** .61803
                    ax.imshow(image_L[:, :, None]*image_rgb, vmin=0, vmax=1)

                else:
                    ax.imshow(1-image_L, vmin=0, vmax=1)

            fname = f'{self.cachepath}/{self.opt.tag}_frame_{i_frame:04d}.png'
            fig.savefig(fname, dpi=dpi)
            fnames.append(fname)
            plt.close('all')

        return make_gif(gifname, fnames, fps=self.opt.fps)

    
# borrowed from https://github.com/gummiks/gummiks.github.io/blob/master/scripts/astro/planck.py

def planck(wav, T):
    import scipy.constants as const
    c = const.c # c = 3.0e+8
    h = const.h # h = 6.626e-34
    k = const.k # k = 1.38e-23
    a = 2.0*h*c**2
    b = h*c/(wav*k*T)
    intensity = a / ( (wav**5) * (np.exp(b) - 1.0) )
    return intensity

def scattering(wav, a=0.005, p=1.3, b=0.45):
    """
    b is  proportionate  to  the  column  density  of  aerosols  
    along  the  path  of  sunlight,  from  outside  the  atmosphere 
    to  the  point  of  observation
    
    N_O3  is  the  ozone  column  density  along  the  path  of  sunlight,  
    sigma_O3 is  the  wavelength dependent ozone absorption cross-section.
    
    """
    # converting wav in µm:
    intensity = np.exp(-a/((wav/1e-6)**4)) # Rayleigh extinction by nitrogen
    intensity *= (wav/1e-6)**-4
    intensity *= np.exp(-b/((wav/1e-6)**p)) # Aerosols
    return intensity

if __name__ == "__main__":
    
    date = datetime.datetime.now().date().isoformat()
    figpath = f'{date}_caustique'
    print(f'Saving our simulations in={figpath}')
    
    opt = init()
    opt.figpath = figpath
    opt.verbose = True

    c = Caustique(opt)
    z = c.wave()
    
    gifname = c.plot(z)
    