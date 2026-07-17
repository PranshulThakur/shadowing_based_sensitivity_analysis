import numpy as np;
import matplotlib.pyplot as plt;

lyp_exp_grid1 = [0.1147021426636166, 0.103617177007911, 0.07910264123307842, 0.05968948631131785, 0.03935807086024172];
lyp_exp_grid2 = [0.161836, 0.140661, 0.122048, 0.111323, 0.0518644];
lyp_exp_grid3 = [0.1780511182364136, 0.1433254072820692, 0.1299182733366628, 0.1185604779210388, 0.1014815864934453];

#lyp_exp_grid1 = [0.144448, 0.139385, 0.109848, 0.0819451, 0.0629324];
#lyp_exp_grid2 = [0.190314, 0.150222, 0.132731, 0.0990697, 0.11198];
#lyp_exp_grid3 = [0.1701512494965841, 0.1784645533778243, 0.176411153017985, 0.1315132856428453, 0.1206764169908358];

grids = [1,2,3];

for i in range(5):
    plt.scatter(grids,[lyp_exp_grid1[i],lyp_exp_grid2[i],lyp_exp_grid3[i]],color='blue', marker='o', s=50);

plt.xticks([1, 2, 3], ['1', '2', '3']);
plt.xlabel("Grid #",fontsize=12);
plt.ylabel("Adjoint Lyapunov exponents",fontsize=12);
plt.ylim([0.02,0.2]);
plt.show();
