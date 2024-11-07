import numpy as np


def gen(cent, vars, ns, dim=2, shuffle_flag=False):
	ds = None
	gs = None
	for i, (c, v, n) in enumerate(zip(cent, vars, ns)):
		if ds is None:
			ds = np.random.normal(loc=c, scale=v, size=(n, dim))
			gs = i * np.ones(n, dtype=int)
			continue
		ds = np.vstack((ds, np.random.normal(loc=c, scale=v, size=(n, dim))))
		gs = np.concatenate((gs, i * np.ones(n, dtype=int)))

	if shuffle_flag:
		return shuffle(ds, gs)
	return ds, gs


def shuffle(ds, gs):
	idx = np.arange(ds.shape[0])
	np.random.shuffle(idx)
	return ds[idx], gs[idx]












