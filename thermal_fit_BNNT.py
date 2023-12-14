import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from dispersion_BNNT_10_10 import fit_k_invm, fit_w_invcm, branch_count, branch_names, k_space_res

# Accounting for the degenerate transverse branch
fit_k_invm.insert(1,fit_k_invm[1])
fit_w_invcm.insert(1,fit_w_invcm[1])

branch_count = len(fit_k_invm)

size = 16
params = {
    'axes.labelsize': size,
    'axes.titlesize': size,
    'xtick.labelsize': size,
    'ytick.labelsize': size,
    'axes.titlepad': 25,
    'legend.fontsize': 12,
    'figure.figsize': (8,6),
}
plt.rcParams.update(params)

# Universal Constants
h_ = 1.055e-34
kB = 1.38e-23

# Unit conversion factors for dispersion relationship (DR) data
meV_to_J = 1.6021773e-22
invcm_to_radpers = 2*np.pi*3e10
invA_to_invm = 1e10

# Defining Dimensionality of the system (1=1D, 2=2D, 3=3D)
dimensionality = 2

# Defining chirality of the CNT
chirality = [10,10]
m = chirality[0]
n = chirality[1]

# Defining factors to determine edges of the Brillouin zone
a = np.sqrt(3)*1.452e-10
L = a*np.sqrt(m**2 + n**2 + m*n)
a_2 = 1.42e-10
L_q = 1.5*m*a_2

diameter = L/np.pi
thickness = 0.335e-9
length =  0.5e-6
area = np.pi/4*diameter**2 # Matching Eric Pop's paper

d = np.gcd(2*m+n,2*n+m)
T_chirality = np.sqrt(3)*L/d
q_mul = 1/T_chirality

# Defining functions to calculate thermal conductivity

# Convert the frequency data from units of cm^-1 to rad/s
def omega(w_invcm):
   return w_invcm*invcm_to_radpers

# Convert the normalized wavenumber data from 0-1 to rad/m
def wavenum(norm_q):
    return norm_q*q_mul

def normalize_q(q_val):
    return q_val/q_mul

# Calculate groupvelocity based on frequency (in rad/s) and wavenumber (in rad/m)
def group_velocity(w_invcm,norm_q):
    vg = np.gradient(omega(w_invcm),wavenum(norm_q))
    return vg

# Calculate mean free path using scattering terms (a constant mean fre path is used for now)
def mean_free_path(w_invcm,norm_q,T):
    if T<300:
        lam = L_q
    else:
        lam = group_velocity(w_invcm,norm_q)/omega(w_invcm)**2
    return lam

# Calculate Bose-Einstein distribution for the energies in the DR
def fBE(w_invcm,T):
    f_BE = 1/(np.exp(h_ * omega(w_invcm) / (kB * T))-1)
    return f_BE

# Calculate the derivative of the Bose-Einstein distribution for energies in the DR
def df_dT(w_invcm,T):
    deriv = (fBE(w_invcm,T)**2)*np.exp(h_ *omega(w_invcm)/(kB*T))*(h_ * omega(w_invcm)/(kB*T**2))
    return deriv

# Calculate density of states for each wavenumber according to the dimensionality specified
def DOS(norm_q,dim):
    if dim==1:
        dos = 1/np.pi
    elif dim==2:
        dos = wavenum(norm_q)/(2*np.pi)
    elif dim==3:
        dos = wavenum(norm_q)**2/(2*np.pi**2)
    return dos

# Calculate sum-to-integral conversion factor according to dimensionality and wavenumber
def sum_integral_factor(norm_q,dim):
    if dim==1:
        fac = 1
    elif dim==2:
        fac = 2*np.pi*wavenum(norm_q)
    elif dim==3:
        fac = 4*np.pi*wavenum(norm_q)**2
    return fac

# Calculate the integrand of the thermal conductivity integral
def integrand(norm_q,w_invcm,T):
    return h_*omega(w_invcm)*df_dT(w_invcm,T)*DOS(norm_q,dimensionality)*group_velocity(w_invcm,norm_q)*mean_free_path(w_invcm,norm_q,T)

# Lists to store wavenumber, frequency and group velocity arrays
q = []
w = []
vg = []

# DR plot
# fig,ax = plt.subplots()
# ax.set_title('Dispersion relationship for CNTs')
# ax.set_xlabel('Wavenumber [rad/m]')
# ax.set_ylabel('Frequency [rad/s]')

# Group VElocity plot
# fig2,ax2 = plt.subplots()
# ax2.set_title('Group Velcoity of CNT phonons')
# ax2.set_xlabel('Wavenumber [rad/m]')
# ax2.set_ylabel('Group Velocity [rad/s]')

# Get q, w, vg data for each branch in the DR and plot
for i in range(branch_count):
    q_in_invm = fit_k_invm[i]
    q.append(q_in_invm)

    w_in_radpers = omega(fit_w_invcm[i])
    w.append(w_in_radpers)

    # ax.plot(q[i],w[i],label =branch_names[i])

    vg_in_mpers = group_velocity(fit_w_invcm[i],fit_k_invm[i])
    vg.append(vg_in_mpers)

    # ax2.plot(q[i],vg[i], label =branch_names[i])

# ax.legend()
# ax2.legend()

# Calculate thermal conductivity

# Range of temeperatures to calculate thermal conductivity
T_low = np.linspace(100,300,101,endpoint=False)
T_high = np.linspace(300,700,101)
calc_low = np.zeros(len(T_low))
calc_high = np.zeros(len(T_high))

T_ranges = [T_low,T_high]
calc_ranges = [calc_low,calc_high]

for _ in range(len(T_ranges)):
    for i in range(len(T_ranges[_])):
        sum = 0
        for j in range(branch_count):
            q_values = np.linspace(1e-12, 0.5/T_chirality, k_space_res)
            norm_q_values = normalize_q(q_values)
            integrand_values = integrand(norm_q_values, fit_w_invcm[j], T_ranges[_][i])
            sum += np.trapz(integrand_values, q_values)
        calc_ranges[_][i] = 1/2*sum

k_low = calc_ranges[0]/thickness
k_high = calc_ranges[1]/thickness

#Plot thermal conductivity
# fig3,ax3 = plt.subplots()
# ax3.plot(T_low,k_low)
# ax3.plot(T_high,k_high)
# ax3.set_title('Temperature Dependence of Thermal Conductance of CNTs')
# ax3.set_xlabel('Temperature [K]')
# ax3.set_ylabel('Thermal Conductance [W/K]')

exp_k_data = np.loadtxt('BNNT_kT.csv', delimiter=',')
exp_k_data = exp_k_data[exp_k_data[:,0].argsort()]

temp = exp_k_data[:,0]
conductivity = exp_k_data[:,1]

new_con = [[],[]]

for _ in range(len(T_ranges)):
    for i in range(len(T_ranges[_])):
        new_con[_].append(np.interp(T_ranges[_][i],temp,conductivity))

A_fit = k_high/np.array(new_con[1])
s_fit = 1 - k_low/np.array(new_con[0])

coeffs_A = np.polyfit(T_high,A_fit,4)
coeffs_s = np.polyfit(T_low,s_fit,4)

polyfit_A = np.polyval(coeffs_A,T_high)
polyfit_s = np.polyval(coeffs_s,T_low)

# fig4,ax4 = plt.subplots()

# ax4.plot(T_high,A_fit,color ='r',linestyle ='none',marker ='.', markerfacecolor ='none',label =r'$A_1$ approximations from experimental data')
# ax4.plot(T_high,polyfit_A,color ='b',linestyle='-',label =r'Polyfit for $A_1$' )
# ax4.set_xlabel('Temperature [K]')
# ax4.set_ylabel(r'$A_1$(T) [s]')
# ax4.set_title(r'Fitting $A_1$ parameter to experimental data for (10,10) BNNT')
# ax4.legend()

# fig5,ax5 = plt.subplots()

# ax5.plot(T_low,s_fit,color ='r',linestyle ='none',marker ='.',markerfacecolor ='none',label =r's approximations from experimental data')
# ax5.plot(T_low,polyfit_s,color ='b',linestyle='-',label =r'Polyfit for s')
# ax5.set_xlabel('Temperature [K]')
# ax5.set_ylabel('s')
# ax5.set_ylim(0.99,1)
# ax5.set_title(r'Fitting specularity parameter to experimental data for (10,10) BNNT')
# ax5.legend()

#plt.show()
