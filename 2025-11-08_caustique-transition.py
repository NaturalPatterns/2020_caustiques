PRECISION = 12 # good quality
from dataclasses import dataclass

@dataclass
class init:
    nx: int = 1*2**PRECISION # number of pixels (vertical)
    ny: int = 1*2**PRECISION # number of pixels (horizontal)
    nframe: int = 24*8 # number of frames for each chunk
    bin_dens: int = 2 # relative bin density
    n_bits: int = 16
    gamma: float = 2.4 # Gamma exponant to convert luminosity to luminance
    fps: float = 24

opt = init()

import imageio
import numpy as np
import subprocess as sp
# import shlex               # pour construire la ligne de commande en toute sécurité
import os
from tqdm.auto import trange

def make_movie(
    out_path: str,
    frames: np.ndarray,
    fps: int = 24,
    codec: str = "prores_ks",
    # codec: str = "ffv1", #
    pix_fmt: str = "rgb48le",
    # pix_fmt: str = "yuv444p16le", # "rgb48le", #
    ffmpeg_path: str = "ffmpeg",
) -> str:

    if frames.ndim != 4:
        raise ValueError(f"`frames` doit être 4‑D, shape={frames.shape}")

    h, w, c, n = frames.shape
    cmd = [
        ffmpeg_path,
        "-y",                         # écraser le fichier de sortie s'il existe
        "-f", "rawvideo",             # le format d'entrée est rawvideo
        "-vcodec", "rawvideo",
        # "-preset", "veryslow",         # compression maximale (plus lent)
        # "-level", "3",                 # qualité maximale du codec ffv1
        "-pix_fmt", pix_fmt,          # pixel‑format d'entrée 
        "-pixel_format", "rgb48le",   # 16‑bit, little‑endian, 3 canaux
        "-s", f"{w}x{h}",             # résolution
        "-r", str(fps),               # fps d'entrée
        "-i", "-",                    # lecture depuis stdin
        "-c:v", codec,                # codec de sortie
        "-pix_fmt", pix_fmt,          # pixel‑format de sortie (doit être compatible)
        "-r", str(fps),               # fps de sortie (on le répète pour être sûr)
        out_path,
    ]

    # ---------------------------------------------------------
    # 3️⃣ Lancement du processus ffmpeg
    # ---------------------------------------------------------
    proc = sp.Popen(
        cmd,
        stdin=sp.PIPE,
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        bufsize=10**8,                # buffer large pour éviter les blocages
    )

    # ---------------------------------------------------------
    # 4️⃣ Envoi des frames une par une (ou en bloc)
    # ---------------------------------------------------------
    # On écrit les données brutes en row‑major order (C‑order) = ce que numpy utilise.
    # Chaque frame est un tableau (H, W, C) en uint16 → on le convertit en bytes.
    try:
        for i in trange(n, desc="Encoding frames"):
            # Convertir le slice en bytes (little‑endian, natif numpy)
            frame_bytes = frames[..., i].tobytes()
            proc.stdin.write(frame_bytes)
        # Signaler à ffmpeg que les données sont terminées
        proc.stdin.close()
    except Exception as e:
        proc.kill()
        raise RuntimeError(f"Erreur pendant le pipe vers ffmpeg : {e}")

    return out_path


# make_movie("test_video_08.mp4", np.random.randint(0, 2**8-1, (480, 640, 3, 30), dtype=np.uint8), fps=24)
# make_movie("test_video_16.mkv", np.random.randint(0, 2**16-1, (480, 640, 3, 300), dtype=np.uint16), fps=24)

N_wave = 2
N_wave = 40
np_dtype = np.uint16

output_filename = '2025-11-08_caustique-transition/longue_caustique_5dbd9fe3.mkv'


image_rgb_max = 0.
for i_wave in trange(N_wave, desc='Scanning blocks'):
    fname = f'/Users/laurent/tmp/2025-11-08_caustique-transition/longue_caustique_{i_wave}.npy'
    image_update = np.load(fname)
    image_rgb_max = max(image_rgb_max, image_update.max())    
    del image_update
print(f'{image_rgb_max=}')


all_frames = np.zeros((opt.nx//opt.bin_dens, opt.ny//opt.bin_dens, 3, 0), dtype=np_dtype)
for i_wave in trange(N_wave, desc='Generating movie'):
    fname = f'/Users/laurent/tmp/2025-11-08_caustique-transition/longue_caustique_{i_wave}.npy'
    frames = np.load(fname)

    frames /= image_rgb_max
    frames = frames ** (1/opt.gamma)
    frames = (frames*(2**opt.n_bits-1)).astype(np_dtype)
    all_frames = np.concatenate((all_frames, frames), axis=-1)
output_filename = make_movie(output_filename, all_frames, fps=opt.fps)

del frames
del all_frames

