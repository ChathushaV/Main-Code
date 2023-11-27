import numpy as np
import matplotlib.pyplot as plt

branch_csv = ['A1.csv','A2.csv','A3.csv','O1.csv','O2.csv']
branch_names = ['Acoustic 1','Acoustic 2','Acoustic 3','Optical 1','Optical 2']
branch_count = len(branch_csv)
poly_order =  4
k_space_res = 10000

branch_data = []
k_invm = []
E_meV = []
poly_coeff = []
fit_k_invm = []
fit_w_invcm = []

fig, ax = plt.subplots()

for i in range(branch_count):
    extract_data = np.loadtxt(branch_csv[i], delimiter=',', encoding='utf-8-sig')

    branch_data.append(extract_data)

    branch_data[i] = branch_data[i][branch_data[i][:,0].argsort()]

    k_invm.append(branch_data[i][:,0])
    E_meV.append(branch_data[i][:,1])

    coeffs = np.polyfit(k_invm[i], E_meV[i], poly_order)
    poly_coeff.append(coeffs)

    fit_k = np.linspace(min(k_invm[i]), max(k_invm[i]), k_space_res)
    fit_k_invm.append(fit_k)

    fit_E = np.polyval(poly_coeff[i], fit_k_invm[i])
    fit_w_invcm.append(fit_E)

    ax.plot(k_invm[i], E_meV[i], 'b.', label=branch_csv[i])
    ax.plot(fit_k_invm[i], fit_w_invcm[i],'b--', label=f'{poly_order}th order fit for'+branch_csv[i])

plt.legend()
plt.show()
