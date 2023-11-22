import numpy as np
import matplotlib.pyplot as plt

branch_csv = ['Acoustic_1.csv','Acoustic_2.csv','Acoustic_3.csv']
branch_count = len(branch_csv)
poly_order =  4

branch_data = []
k_invA = []
E_meV = []
poly_coeff = []
fit_k_invA = []
fit_E_meV = []

#fig, ax = plt.subplots()

for i in range(branch_count):
    extract_data = np.loadtxt(branch_csv[i], delimiter=',', encoding='utf-8-sig')

    branch_data.append(extract_data)

    branch_data[i] = branch_data[i][branch_data[i][:,0].argsort()]

    k_invA.append(branch_data[i][:,0])
    E_meV.append(branch_data[i][:,1])

    coeffs = np.polyfit(k_invA[i], E_meV[i], poly_order)
    poly_coeff.append(coeffs)

    fit_k = np.linspace(min(k_invA[i]), max(k_invA[i]), 100)
    fit_k_invA.append(fit_k)

    fit_E = np.polyval(poly_coeff[i], fit_k_invA[i])
    fit_E_meV.append(fit_E)

    #ax.plot(k_invA[i], E_meV[i], 'b.', label=branch_csv[i])
    #ax.plot(fit_k_invA[i], fit_E_meV[i],'b--', label=f'{poly_order}th order fit for'+branch_csv[i])

#plt.legend()
#plt.show()
