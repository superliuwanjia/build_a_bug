import numpy as np
from brownian import *

def brownian(vids, sigma=0.7):
	"""
	'vids': batches of videos (batchs, frames, height, length, channels)

	'sigma': sigma for gaussian distribution
	"""
	batchs, frames = vids.shape[:2]

	# horizontal and vertical movements 
	hs = np.random.normal(0, sigma, batchs*frames)
	vs = np.random.normal(0, sigma, batchs*frames)

	# integer the movements
	f_int = np.vectorize(int)
	hs = f_int(hs)
	vs = f_int(vs)

	for i in range(batchs):
		for j in range(frames):
			h = hs[i*batchs + j]
			v = vs[i*batchs + j]

			if h <= 0:
				vids[i][j] = np.lib.pad(vids[i][j], ((0,0),(0,-h),(0,0)), mode='constant')[:,-h:,:]
			else:
				vids[i][j] = np.lib.pad(vids[i][j], ((0,0),(h,0),(0,0)), mode='constant')[:,:-h,:]

			if v <= 0:
				vids[i][j] = np.lib.pad(vids[i][j], ((0,-v),(0,0),(0,0)), mode='constant')[-v:,:,:]
			else:
				vids[i][j] = np.lib.pad(vids[i][j], ((v,0),(0,0),(0,0)), mode='constant')[:-v,:,:]
	return vids

