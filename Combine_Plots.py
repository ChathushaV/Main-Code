import numpy as np
import matplotlib.pyplot as plt
import thermal_prop_CNT_10_10
import thermal_prop_BNNT_10_10
import thermal_prop_CNT_10_0
import lambda_BNNT
import lambda_CNT

size = 16
params = {
    'axes.labelsize': size,
    'axes.titlesize': size,
    'xtick.labelsize': size*0.75,
    'ytick.labelsize': size*0.75,
    'axes.titlepad': 25,
    'legend.fontsize': 12,
    'figure.figsize': (8,6),
}
plt.rcParams.update(params)

BNNT_1010_k = thermal_prop_BNNT_10_10.k
BNNT_1010_G = thermal_prop_BNNT_10_10.conductance
BNNT_1010_cp = thermal_prop_BNNT_10_10.cp
T_cp = thermal_prop_BNNT_10_10.T_cp
T_k = thermal_prop_BNNT_10_10.T

CNT_1010_k = thermal_prop_CNT_10_10.k
CNT_1010_G = thermal_prop_CNT_10_10.conductance
CNT_1010_cp = thermal_prop_CNT_10_10.cp


CNT_100_k = thermal_prop_CNT_10_0.k
CNT_100_G = thermal_prop_CNT_10_0.conductance
CNT_100_cp = thermal_prop_CNT_10_0.cp

lam_B = lambda_BNNT.lambda_fit
lam_C = lambda_CNT.lambda_fit

fig,ax = plt.subplots()

ax.plot(T_cp,np.log10(CNT_1010_cp), 'b' ,label = 'Specific Heat Capacity for CNT [10,10]')
ax.plot(T_cp,np.log10(CNT_100_cp), 'g',label = 'Specific Heat Capacity for CNT [10,0]')
ax.plot(T_cp,np.log10(BNNT_1010_cp), 'k',label = 'Specific Heat Capacity for BNNT [10,10]')
ax.set_xlabel('Temperature [K]')
ax.set_ylabel(r'Specific Heat Capacity [W/($m^3$K)]')
ax.set_yticks([2,3,4,5,6])
ax.set_yticklabels([r'$10^2$',r'$10^3$',r'$10^4$',r'$10^5$',r'$10^6$'])
ax.set_title('Comparison of Specific Heat Capacity of CNTs and BNNTs')
ax.legend()

fig2,ax2 = plt.subplots()

ax2.plot(T_k,CNT_1010_k, 'b' ,label = 'Back Calculated k for CNT [10,10]')
ax2.plot(T_k,BNNT_1010_k, 'k',label = 'Back Calculated k for BNNT [10,10]')
ax2.set_xlabel('Temperature [K]')
ax2.set_ylabel('Thermal Conductivity [W/(m.K)]')
ax2.set_title('Back Calculated Thermal Conductivities (k) of CNTs and BNNTs')
ax2.legend()

fig3,ax3 = plt.subplots()

ax3.plot(T_k,CNT_1010_k, 'b' ,label = 'Back Calculated k for CNT [10,10]')
ax3.plot(T_k,CNT_100_k, 'g',label = 'Predicted k for CNT [10,0]')
ax3.set_xlabel('Temperature [K]')
ax3.set_ylabel('Thermal Conductivity [W/(m.K)]')
ax3.set_title('Prediction of Thermal Conductivity (k) of [10,0] CNT')
ax3.legend()

fig4,ax4 = plt.subplots()

ax4.plot(T_k,CNT_1010_G, 'b' ,label = 'Back Calculated G for CNT [10,10]')
ax4.plot(T_k,CNT_100_G, 'g',label = 'Predicted G for CNT [10,0]')
ax4.plot(T_k,BNNT_1010_G, 'k',label = 'Back Calculated G for BNNT [10,10]')
ax4.set_xlabel('Temperature [K]')
ax4.set_ylabel('Thermal Conductance [W/K]')
ax4.set_title('Comparison of Thermal Conductance (G) ')
ax4.legend()

plt.show()