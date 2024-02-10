# -*- coding: utf-8 -*-
from os import cpu_count
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import scipy
from PIL import Image

OUTDIR = Path("./images_stego").resolve()
MATRICES = [x.resolve() for x in Path("./images_stego").iterdir()]

def mat2png(fname: Path):
	mat = scipy.io.loadmat(fname)
	Image.fromarray(mat["stego"]).convert("L").save(OUTDIR / (fname.stem + ".png"))

def main():
	with ProcessPoolExecutor(max_workers=cpu_count() - 1) as executor:
		executor.map(mat2png, MATRICES, chunksize=4)

if __name__ == "__main__":
	main()
