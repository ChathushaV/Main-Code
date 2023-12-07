import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from dispersion_BNNT_10_0 import fit_k_invm, fit_w_invcm, branch_count, branch_names, k_space_res
from thermal_fit_BNNT import coeffs_A,coeffs_s

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
chirality = [14,0]
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
area = np.pi*thickness*diameter

d = np.gcd(2*m+n,2*n+m)
T_chirality = np.sqrt(3)*L/d
q_mul = np.pi/T_chirality

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
        lam = L_q/(1-s(T))
    else:
        lam = group_velocity(w_invcm,norm_q)/(A(T)*omega(w_invcm)**2)
    return lam

def A(T):
    return np.polyval(coeffs_A,T)

def s(T):
    return np.polyval(coeffs_s,T)

def mfp_final(w_invcm,norm_q,T):
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

# Calculate the integrand of the thermal conductivity integral
def integrand(norm_q,w_invcm,T):
    return h_*omega(w_invcm)*df_dT(w_invcm,T)*DOS(norm_q,dimensionality)*group_velocity(w_invcm,norm_q)*mean_free_path(w_invcm,norm_q,T)

def heat_capacity(norm_q,w_invcm,T):
    return h_*omega(w_invcm)*df_dT(w_invcm,T)*DOS(norm_q,dimensionality)

# Lists to store wavenumber, frequency and group velocity arrays
q = []
w = []
vg = []

# DR plot
fig,ax = plt.subplots()
ax.set_title('Dispersion relationship for [14,0] BNNTs')
ax.set_xlabel('Wavenumber [rad/m]')
ax.set_ylabel('Frequency [rad/s]')

# Group VElocity plot
fig2,ax2 = plt.subplots()
ax2.set_title('Group Velcoity of [14,0] BNNT phonons')
ax2.set_xlabel('Wavenumber [rad/m]')
ax2.set_ylabel('Group Velocity [m/s]')

cmap = ['r','g','b','k','m']

# Get q, w, vg data for each branch in the DR and plot
for i in range(branch_count):
    q_in_invm = fit_k_invm[i]
    q.append(q_in_invm)

    color = cmap[i]

    w_in_radpers = omega(fit_w_invcm[i])
    w.append(w_in_radpers)

    ax.plot(q[i],w[i],label =branch_names[i],color=color, linewidth = 2)

    vg_in_mpers = group_velocity(fit_w_invcm[i],fit_k_invm[i])
    vg.append(vg_in_mpers)

    ax2.plot(q[i],vg[i], label =branch_names[i],color=color, linewidth = 2)

ax.legend()
ax2.legend()

# Calculate thermal conductivity

# Range of temeperatures to calculate thermal conductivity
T = np.linspace(300,700,101)
T_cp = np.linspace(10,1000,1001)

calc_k = np.zeros(len(T))
calc_cp = np.zeros(len(T_cp))

for i in range(len(T)):
    sum = 0
    for j in range(branch_count):
        q_values = np.linspace(1e-12, 0.5*np.pi/T_chirality, k_space_res)
        norm_q_values = normalize_q(q_values)
        integrand_values = integrand(norm_q_values, fit_w_invcm[j], T[i])
        sum += np.trapz(integrand_values, q_values)
    calc_k[i] = 1/2*sum

for i in range(len(T_cp)):
    sum = 0
    for j in range(branch_count):
        q_values = np.linspace(1e-12, 0.5*np.pi/T_chirality, k_space_res)
        norm_q_values = normalize_q(q_values)
        cp_int_values = heat_capacity(norm_q_values, fit_w_invcm[j], T_cp[i])
        sum += np.trapz(cp_int_values, q_values)
    calc_cp[i] = 1/2*sum
    
k = calc_k/thickness
cp = calc_cp/thickness
conductance = k*area/length

# Plot thermal conductivity
fig3,ax3 = plt.subplots()
ax3.plot(T,k)
ax3.set_title('Temperature Dependence of Thermal Conductivity of (14,0) BNNTs')
ax3.set_xlabel('Temperature [K]')
ax3.set_ylabel('Thermal Conductivity [W/(m.K)]')

# Plot specific heat capacity
fig4,ax4 = plt.subplots()
ax4.plot(T_cp,cp)
ax4.set_title('Temperature Dependence of Specific Heat Capacity of (14,0) BNNTs')
ax4.set_xlabel('Temperature [K]')
ax4.set_ylabel(r'Specific Heat Capacity [W/($m^3$K)]')

# Plot specific heat capacity
fig5,ax5 = plt.subplots()
ax5.plot(T,conductance)
ax5.set_title('Temperature Dependence of Conductance of (14,0) BNNTs')
ax5.set_xlabel('Temperature [K]')
ax5.set_ylabel('Conductance [W/K]')

plt.show()
