import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

size = 16
params = {
    'axes.labelsize': size,
    'axes.titlesize': size,
    'xtick.labelsize': size,
    'ytick.labelsize': size,
    'axes.titlepad': 25,
    'legend.fontsize': 8,
    'figure.figsize': (8,6),
}
plt.rcParams.update(params)

plt.rcParams['font.family'] = 'Arial'

branch_csv = ['BNNT_A1.csv','BNNT_A2.csv','BNNT_A3.csv','BNNT_O1.csv','BNNT_O2.csv']
branch_names = ['Acoustic 1','Acoustic 2','Acoustic 3','Optical 1','Optical 2']
branch_count = len(branch_csv)
poly_order =  8
k_space_res = 10000

branch_data = []
k_invm = []
f_invcm = []
poly_coeff = []
fit_k_invm = []
fit_w_invcm = []

#fig, ax = plt.subplots()

cmap = ['r','g','b','k','m']

for i in range(branch_count):
    extract_data = np.loadtxt(branch_csv[i], delimiter=',', encoding='utf-8-sig')

    branch_data.append(extract_data)

    branch_data[i] = branch_data[i][branch_data[i][:,0].argsort()]

    k_invm.append(branch_data[i][:,0])
    f_invcm.append(branch_data[i][:,1])

    coeffs = np.polyfit(k_invm[i], f_invcm[i], poly_order)
    poly_coeff.append(coeffs)

    fit_k = np.linspace(min(k_invm[i]), max(k_invm[i]), k_space_res)
    fit_k_invm.append(fit_k)

    fit_E = np.polyval(poly_coeff[i], fit_k_invm[i])
    fit_w_invcm.append(fit_E)

    color = cmap[i]

    #ax.plot(k_invm[i], f_invcm[i], linestyle='none', marker ='o', markerfacecolor = 'none',label='Extracted points from '+branch_names[i], color = color)
    #ax.plot(fit_k_invm[i], fit_w_invcm[i],linestyle='-', label=f'{poly_order}th order polynomial fit for '+branch_names[i],color = color)

#ax.set_xlabel('kT')
#ax.set_ylabel(r'Frequency [$cm^{-1}$]')
#ax.set_title('Extracted Dispersion Relationship for (10,10) BNNT')
#plt.legend()
