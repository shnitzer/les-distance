import numpy as np
import matplotlib.pyplot as plt
from les import les_desc_comp, les_dist_comp
from comparisons import CompareIMD, CompareIMDOurApproach, CompareTDA, CompareGS, CompareGW

# Simulation parameters:
N = 1000  # Number of samples - reduced from N=3000 for faster computation times
ITER_NUM = 2  # Number of trials to average on
R1 = 10  # Major radius
R2 = 3  # Minor/middle radius in 2D/3D
R3 = 1  # Minor radius in 3D
NOISE_VAR = 0.01  # STD of added noise to the tori data
R_RATIOS = np.arange(0.4, 1.01, 0.2)  # Radius ratio (c parameter)
DICT_KEYS = ['t2D_2DSc', 't2D_3D', 't2D_3DSc', 't3D_2DSc', 't3D_3DSc']

# LES hyperparameter
GAMMA = 1e-8  # Kernel regularization parameter
SIGMA = 2  # Kernel scale
NEV = 200  # Number of eigenvalues to estimate

# ========================== Comparisons: ==========================
# List of algorithms to compare with. Possible algorithms: 'imd_ours', 'imd', 'tda', 'gs', 'gw'
ALGS2COMPARE = ['imd_ours']  # ['imd_ours', 'imd', 'gs', 'gw', 'tda']

ALG_CLASS = {'imd': CompareIMD,
             'imd_ours': CompareIMDOurApproach,
             'tda': CompareTDA,
             'gs': CompareGS,
             'gw': CompareGW,
             }

# Initialize the classes that compute the compared algorithms
algs_dists = {}
for alg in ALGS2COMPARE:
    alg = alg.lower()
    if alg == 'tda':
        algs_dists['tda_H0'] = ALG_CLASS[alg](0, ITER_NUM, R_RATIOS, DICT_KEYS)
        algs_dists['tda_H1'] = ALG_CLASS[alg](1, ITER_NUM, R_RATIOS, DICT_KEYS)
        algs_dists['tda_H2'] = ALG_CLASS[alg](2, ITER_NUM, R_RATIOS, DICT_KEYS)
    elif alg == 'imd_ours':
        algs_dists[alg] = ALG_CLASS[alg](GAMMA, ITER_NUM, R_RATIOS, DICT_KEYS)
    else:
        algs_dists[alg] = ALG_CLASS[alg](ITER_NUM, R_RATIOS, DICT_KEYS)

# ========= Initializations and tori equation definitions =========
les_dist = {key: np.zeros((ITER_NUM, len(R_RATIOS))) for key in DICT_KEYS}


def tori_2d_gen(c):
    ang1, ang2, ang3 = 2 * np.pi * np.random.rand(N), 2 * np.pi * np.random.rand(N), 2 * np.pi * np.random.rand(N)
    tor2d = np.concatenate(([(R1 + c * R2 * np.cos(ang2)) * np.cos(ang1)],
                            [(R1 + c * R2 * np.cos(ang2)) * np.sin(ang1)],
                            [c * R2 * np.sin(ang2)]),
                           axis=0)
    tor2d += NOISE_VAR * np.random.randn(3, N)
    return tor2d


def tori_3d_gen(c):
    ang1, ang2, ang3 = 2 * np.pi * np.random.rand(N), 2 * np.pi * np.random.rand(N), 2 * np.pi * np.random.rand(N)
    tor3d = np.concatenate(([(R1 + (R2 + c * R3 * np.cos(ang3)) * np.cos(ang2)) * np.cos(ang1)],
                            [(R1 + (R2 + c * R3 * np.cos(ang3)) * np.cos(ang2)) * np.sin(ang1)],
                            [(R2 + c * R3 * np.cos(ang3)) * np.sin(ang2)],
                            [c * R3 * np.sin(ang3)]),
                           axis=0)
    tor3d += NOISE_VAR * np.random.randn(4, N)
    return tor3d


for ite in range(ITER_NUM):
    print(f'Running iteration number {ite}')

    for i, r_ratio in enumerate(R_RATIOS):
        print(f'Computing radius ratio c = {r_ratio:.1f}')

        # -------------- Generate tori data --------------
        data_2d_tor = tori_2d_gen(1)
        data_2d_tor_sc = tori_2d_gen(r_ratio)
        data_3d_tor = tori_3d_gen(1)
        data_3d_tor_sc = tori_3d_gen(r_ratio)

        # ---- Computing dataset descriptors and distances ----
        print('Computing LES descriptors and distances')
        les_desc_2d_tor = les_desc_comp(data_2d_tor.T, SIGMA, NEV, GAMMA)
        les_desc_2d_tor_sc = les_desc_comp(data_2d_tor_sc.T, SIGMA, NEV, GAMMA)
        les_desc_3d_tor = les_desc_comp(data_3d_tor.T, SIGMA, NEV, GAMMA)
        les_desc_3d_tor_sc = les_desc_comp(data_3d_tor_sc.T, SIGMA, NEV, GAMMA)

        les_dist['t2D_2DSc'][ite, i] = les_dist_comp(les_desc_2d_tor, les_desc_2d_tor_sc)
        les_dist['t2D_3D'][ite, i] = les_dist_comp(les_desc_2d_tor, les_desc_3d_tor)
        les_dist['t2D_3DSc'][ite, i] = les_dist_comp(les_desc_2d_tor, les_desc_3d_tor_sc)
        les_dist['t3D_2DSc'][ite, i] = les_dist_comp(les_desc_3d_tor, les_desc_2d_tor_sc)
        les_dist['t3D_3DSc'][ite, i] = les_dist_comp(les_desc_3d_tor, les_desc_3d_tor_sc)

        for alg in algs_dists:
            print('Computing ' + alg.upper() + ' descriptors')
            if alg == 'imd_ours':
                algs_dists[alg].comp_all_tori_dists(ite, i, les_desc_2d_tor, les_desc_2d_tor_sc, les_desc_3d_tor,
                                                    les_desc_3d_tor_sc)
            else:
                algs_dists[alg].comp_all_tori_dists(ite, i, data_2d_tor.T, data_2d_tor_sc.T, data_3d_tor.T,
                                                    data_3d_tor_sc.T)

# ========================== Plot display ==========================
plt.style.use('seaborn-paper')
line_width = 3
alpha_val = 0.2


def create_distance_plt(var, ylabel='', xlabel=''):
    plt.plot(R_RATIOS, np.mean(var['t2D_2DSc'], axis=0), '-', color='teal', linewidth=line_width,
             label="$d(T_{2},T_{2}^{Sc})$")
    x, y, err = R_RATIOS, np.mean(var['t2D_2DSc'], axis=0), np.std(var['t2D_2DSc'], axis=0)
    plt.fill_between(x, y - err, y + err, alpha=alpha_val, facecolor='teal', linewidth=0)
    plt.plot(R_RATIOS, np.mean(var['t3D_3DSc'], axis=0), '-', color='indigo', linewidth=line_width,
             label="$d(T_{3},T_{3}^{Sc})$")
    y, err = np.mean(var['t3D_3DSc'], axis=0), np.std(var['t3D_3DSc'], axis=0)
    plt.fill_between(x, y - err, y + err, alpha=alpha_val, facecolor='indigo', linewidth=0)
    plt.plot(R_RATIOS, np.mean(var['t2D_3D'], axis=0), '--', color='yellowgreen', linewidth=line_width,
             label="$d(T_{2},T_{3})$")
    y, err = np.mean(var['t2D_3D'], axis=0), np.std(var['t2D_3D'], axis=0)
    plt.fill_between(x, y - err, y + err, alpha=alpha_val, facecolor='yellowgreen', linewidth=0)
    plt.plot(R_RATIOS, np.mean(var['t2D_3DSc'], axis=0), '--', color='plum', linewidth=line_width,
             label="$d(T_{2},T_{3}^{Sc})$")
    y, err = np.mean(var['t2D_3DSc'], axis=0), np.std(var['t2D_3DSc'], axis=0)
    plt.fill_between(x, y - err, y + err, alpha=alpha_val, facecolor='plum', linewidth=0)
    plt.plot(R_RATIOS, np.mean(var['t3D_2DSc'], axis=0), '--', color='tomato', linewidth=line_width,
             label="$d(T_{3},T_{2}^{Sc})$")
    y, err = np.mean(var['t3D_2DSc'], axis=0), np.std(var['t3D_2DSc'], axis=0)
    plt.fill_between(x, y - err, y + err, alpha=alpha_val, facecolor='tomato', linewidth=0)

    plt.ylim(bottom=0)
    plt.xticks([0.4, 0.6, 0.8, 1], ['0.4', '0.6', '0.8', '1'])
    plt.ylabel(ylabel, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)


anum = len(algs_dists) + 1
if anum <= 4:
    sbplt_rc = [1, anum]
else:
    sbplt_rc = [2, int(np.ceil(anum/2))]

fig = plt.figure(figsize=[10, 6])
plt.subplot(sbplt_rc[0], sbplt_rc[1], 1)
create_distance_plt(les_dist, ylabel='LES', xlabel='Radius Scale (c)')
for i, alg in enumerate(algs_dists):
    plt.subplot(sbplt_rc[0], sbplt_rc[1], i+2)
    create_distance_plt(algs_dists[alg].all_distances, ylabel=alg.upper(), xlabel='Radius Scale (c)')

# plt.legend(framealpha=1, frameon=True, handlelength=2.5)
fig.tight_layout()
legendid = plt.legend(framealpha=1, frameon=True, loc='upper right', bbox_to_anchor=(0.95, 2.4), fontsize=14, labelspacing=0.1, handlelength=2, ncol=5)
plt.savefig('Tori_comparisons.pdf')
plt.show()
