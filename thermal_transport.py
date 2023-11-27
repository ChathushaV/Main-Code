import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from dispersion_data import fit_k_invm, fit_w_invcm, branch_count, branch_names, k_space_res

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
a = 2.49e-10
L = a*np.sqrt(m**2 + n**2 + m*n)
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

# Calculate groupvelocity based on frequency (in rad/s) and wavenumber (in rad/m)
def group_velocity(w_invcm,norm_q):
    vg = np.gradient(omega(w_invcm),wavenum(norm_q))
    return vg

# Calculate mean free path using scattering terms (a constant mean fre path is used for now)
def mean_free_path():
    return 100e-9

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
    return h_*omega(w_invcm)*df_dT(w_invcm,T)*DOS(norm_q,dimensionality)*group_velocity(w_invcm,norm_q)*mean_free_path()*sum_integral_factor(norm_q,dimensionality)

"""Write new functions below this line (starting with unit 4)."""
def main():
    # Lists to store wavenumber, frequency and group velocity arrays
    q = []
    w = []
    vg = []

    # DR plot
    fig,ax = plt.subplots()
    ax.set_title('Dispersion relationship for CNTs')
    ax.set_xlabel('Wavenumber [rad/m]')
    ax.set_ylabel('Frequency [rad/s]')

    # Group VElocity plot
    fig2,ax2 = plt.subplots()
    ax2.set_title('Group Velcoity of CNT phonons')
    ax2.set_xlabel('Wavenumber [rad/m]')
    ax2.set_ylabel('Group Velocity [rad/s]')

    # Get q, w, vg data for each branch in the DR and plot
    for i in range(branch_count):
        q_in_invm = fit_k_invm[i]
        q.append(q_in_invm)

        w_in_radpers = omega(fit_w_invcm[i])
        w.append(w_in_radpers)

        ax.plot(q[i],w[i],label =branch_names[i])

        vg_in_mpers = group_velocity(fit_w_invcm[i],fit_k_invm[i])
        vg.append(vg_in_mpers)

        ax2.plot(q[i],vg[i], label =branch_names[i])

    ax.legend()
    ax2.legend()

    # Calculate thermal conductivity

    # Range of temeperatures to calculate thermal conductivity
    T = np.linspace(10,1000,101)

    k = np.zeros(len(T))

    for i in range(len(T)):
        sum = 0
        for j in range(branch_count):
            q_values = np.linspace(1e-12, np.pi/T_chirality, k_space_res)
            integrand_values = integrand(q_values, fit_w_invcm[j], T[i])
            sum += np.trapz(integrand_values, wavenum(q_values))
        k[i] = 1/3*sum

    # Plot thermal conductivity
    fig3,ax3 = plt.subplots()
    ax3.plot(T,k)
    ax3.set_title('Temperature Dependence of Thermal Conductivity of CNTs')
    ax3.set_xlabel('Temperature [K]')
    ax3.set_ylabel('Thermal Conductivity [W/mK]')

    plt.show()

if __name__ == "__main__":
    main()