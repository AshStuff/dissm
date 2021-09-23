import numpy as np


class Config():
    def __init__(self, trans_center, scale, global_init_mtx_trans):
        self._trans_center = trans_center
        Apx2sdf = np.eye(4)
        Apx2sdf[0, 3] = trans_center[0]
        Apx2sdf[1, 3] = trans_center[1]
        Apx2sdf[2, 3] = trans_center[2]
        Apx2sdf[0, :] *= -1 / scale
        Apx2sdf[1, :] *= -1 / scale
        Apx2sdf[2, :] *= 1 / scale
        self._Apx2sdf = Apx2sdf
        # create initial init matrix which performs translation alone
        global_init_mtx = np.eye(4)
        global_init_mtx[0, 3] = global_init_mtx_trans[0]
        global_init_mtx[1, 3] = global_init_mtx_trans[1]
        global_init_mtx[2, 3] = global_init_mtx_trans[2]
        self._global_init_mtx = global_init_mtx

    @property
    def trans_center(self):
        return self._trans_center

    @property
    def Apx2sdf(self):
        return self._Apx2sdf

    @property
    def global_init_mtx(self):
        return self._global_init_mtx




# Example stats for liver
# trans_center = [-63.5, -63.5, -100.5]
#
# Apx2sdf = np.eye(4)
# Apx2sdf[0, 3] = trans_center[0]
# Apx2sdf[1, 3] = trans_center[1]
# Apx2sdf[2, 3] = trans_center[2]
# Apx2sdf[0, :] *= -1 / 35.0
# Apx2sdf[1, :] *= -1 / 35.0
# Apx2sdf[2, :] *= 1 / 35.0
#
# # create initial init matrix which performs translation alone
# global_init_mtx = np.eye(4)
# global_init_mtx[0, 3] = 77.4
# global_init_mtx[1, 3] = 68.7
# global_init_mtx[2, 3] = 118.2



# Example stats for larynx
# trans_center = [-(161/2), -(161/2), -(611/2)]
#
# Apx2sdf = np.eye(4)
# Apx2sdf[0, 3] = trans_center[0]
# Apx2sdf[1, 3] = trans_center[1]
# Apx2sdf[2, 3] = trans_center[2]
# Apx2sdf[0, :] *= -1 / 20.0
# Apx2sdf[1, :] *= -1 / 20.0
# Apx2sdf[2, :] *= 1 / 20.
#
# # create initial init matrix which performs translation alone
# global_init_mtx = np.eye(4)
# global_init_mtx[0, 3] = 77.4
# global_init_mtx[1, 3] = 68.7
# global_init_mtx[2, 3] = 279.4