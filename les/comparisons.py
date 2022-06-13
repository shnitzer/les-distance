import abc
import numpy as np
import scipy.spatial as spat


class CompareBase:
    def __init__(self, iter_num, r_ratios, dict_keys):
        self.all_distances = {key: np.zeros((iter_num, len(r_ratios))) for key in dict_keys}
    
    @abc.abstractmethod
    def _comp_desc(self, data):
        """
        Compute the algorithm's descriptors per dataset
        :param data: dataset samples organized as [samples x features]
        :return desc: descriptor vector for the dataset
        """

    @abc.abstractmethod
    def _comp_dist(self, desc1, desc2):
        """
        Compute the algorithm's distances a pair of dataset descriptors
        :param desc1, desc2: descriptors of two datasets
        :return dist: distance between the datasets based on the given algorithm
        """
        
    def comp_all_tori_dists(self, ite, i, data_2d_tor, data_2d_tor_sc, data_3d_tor, data_3d_tor_sc):
        """
        Compute the distances between all tori datasets
        :param ite: realization number
        :param i: scale index
        :param data_2d_tor, data_2d_tor_sc, data_3d_tor, data_3d_tor_sc: tori datasets organized as [samples x features]
        """
        desc_2d_tor = self._comp_desc(data_2d_tor)
        desc_2d_tor_sc = self._comp_desc(data_2d_tor_sc)
        desc_3d_tor = self._comp_desc(data_3d_tor)
        desc_3d_tor_sc = self._comp_desc(data_3d_tor_sc)

        self.all_distances['t2D_2DSc'][ite, i] = self._comp_dist(desc_2d_tor, desc_2d_tor_sc)
        self.all_distances['t2D_3D'][ite, i] = self._comp_dist(desc_2d_tor, desc_3d_tor)
        self.all_distances['t2D_3DSc'][ite, i] = self._comp_dist(desc_2d_tor, desc_3d_tor_sc)
        self.all_distances['t3D_2DSc'][ite, i] = self._comp_dist(desc_3d_tor, desc_2d_tor_sc)
        self.all_distances['t3D_3DSc'][ite, i] = self._comp_dist(desc_3d_tor, desc_3d_tor_sc)


class CompareIMD(CompareBase):
    def __init__(self, *args):
        super().__init__(*args)
        imd = __import__('msid')
        self.imd_descriptor = imd.msid.msid_descriptor

        # IMD hyperparameters
        self.T = np.logspace(-1, 1, 256)  # Temperatures for heat kernel approx.
        self.IMD_N_NBRS = 5  # Number of neighbors in graph Laplacian
        self.M_LANCOZ = 10  # Number of Lanczos steps in SLQ

    def _comp_desc(self, data):
        desc = self.imd_descriptor(data, ts=self.T, k=self.IMD_N_NBRS, graph_builder='sparse', m=self.M_LANCOZ)
        return desc

    def _comp_dist(self, desc1, desc2):
        ct = np.exp(-2 * (self.T + 1 / self.T))
        dist = np.amax(ct * np.abs(desc1 - desc2))
        return dist


class CompareTDA(CompareBase):
    def __init__(self, bnum, *args):
        super().__init__(*args)
        ripser = __import__('ripser')
        self.rips = ripser.Rips(maxdim=2)
        self.persim = __import__('persim')

        self.bnum = bnum

    def _comp_desc(self, data):
        desc = self.rips.fit_transform(data)[self.bnum]
        return desc

    def _comp_dist(self, desc1, desc2):
        dist = self.persim.bottleneck(desc1, desc2)
        return dist


class CompareGS(CompareBase):
    def __init__(self, *args):
        super().__init__(*args)
        gs = __import__('gs')
        self.gs = gs

        self.NGS = 200  # Tori results in Figure 1(d) are with NGS=2000, reduced here for speed

    def _comp_desc(self, data):
        desc = self.gs.rlts(data, n=self.NGS)
        return desc

    def _comp_dist(self, desc1, desc2):
        dist = self.gs.geom_score(desc1, desc2)
        return dist


class CompareGW:
    def __init__(self, iter_num, r_ratios, dict_keys):
        self.ot = __import__('ot')
        self.all_distances = {key: np.zeros((iter_num, len(r_ratios))) for key in dict_keys}

    def comp_all_tori_dists(self, ite, i, data_2d_tor, data_2d_tor_sc, data_3d_tor, data_3d_tor_sc):
        """
        Compute the distances between all tori datasets
        :param ite: realization number
        :param i: scale index
        :param data_2d_tor, data_2d_tor_sc, data_3d_tor, data_3d_tor_sc: tori datasets organized as [samples x features]
        """
        n = data_2d_tor.shape[0]
        p = self.ot.unif(n)
        q = self.ot.unif(n)

        dist_mat_2d_tor = spat.distance.cdist(data_2d_tor, data_2d_tor)
        dist_mat_2d_tor_sc = spat.distance.cdist(data_2d_tor_sc, data_2d_tor_sc)
        dist_mat_3d_tor = spat.distance.cdist(data_3d_tor, data_3d_tor)
        dist_mat_3d_tor_sc = spat.distance.cdist(data_3d_tor_sc, data_3d_tor_sc)

        self.all_distances['t2D_2DSc'][ite, i] = self.ot.gromov_wasserstein2(dist_mat_2d_tor, dist_mat_2d_tor_sc, p, q)
        self.all_distances['t2D_3D'][ite, i] = self.ot.gromov_wasserstein2(dist_mat_2d_tor, dist_mat_3d_tor, p, q)
        self.all_distances['t2D_3DSc'][ite, i] = self.ot.gromov_wasserstein2(dist_mat_2d_tor, dist_mat_3d_tor_sc, p, q)
        self.all_distances['t3D_2DSc'][ite, i] = self.ot.gromov_wasserstein2(dist_mat_3d_tor, dist_mat_2d_tor_sc, p, q)
        self.all_distances['t3D_3DSc'][ite, i] = self.ot.gromov_wasserstein2(dist_mat_3d_tor, dist_mat_3d_tor_sc, p, q)


class CompareIMDOurApproach:
    def __init__(self, gamma, iter_num, r_ratios, dict_keys):
        self.T = np.logspace(-1, 1, 256)  # Temperatures for heat kernel approx.
        self.gamma = gamma

        self.all_distances = {key: np.zeros((iter_num, len(r_ratios))) for key in dict_keys}

    def _comp_desc(self, les_desc):
        """
        :param data: Here data should be the LES descriptor
        """
        desc = np.sum((np.exp(les_desc) - self.gamma)[:, None] ** self.T, axis=0)
        return desc

    def _comp_dist(self, desc1, desc2):
        ct = np.exp(-2 * (self.T + 1 / self.T))
        dist = np.amax(ct * np.abs(desc1 - desc2))
        return dist

    def comp_all_tori_dists(self, ite, i, les_2d_tor, les_2d_tor_sc, les_3d_tor, les_3d_tor_sc):
        desc_2d_tor = self._comp_desc(les_2d_tor)
        desc_2d_tor_sc = self._comp_desc(les_2d_tor_sc)
        desc_3d_tor = self._comp_desc(les_3d_tor)
        desc_3d_tor_sc = self._comp_desc(les_3d_tor_sc)

        self.all_distances['t2D_2DSc'][ite, i] = self._comp_dist(desc_2d_tor, desc_2d_tor_sc)
        self.all_distances['t2D_3D'][ite, i] = self._comp_dist(desc_2d_tor, desc_3d_tor)
        self.all_distances['t2D_3DSc'][ite, i] = self._comp_dist(desc_2d_tor, desc_3d_tor_sc)
        self.all_distances['t3D_2DSc'][ite, i] = self._comp_dist(desc_3d_tor, desc_2d_tor_sc)
        self.all_distances['t3D_3DSc'][ite, i] = self._comp_dist(desc_3d_tor, desc_3d_tor_sc)

