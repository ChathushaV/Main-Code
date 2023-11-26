import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from dispersion_data import fit_k_invm, fit_w_invcm, branch_count

h_ = 1.055e-34
kB = 1.38e-23
meV_to_J = 1.6021773e-22
invcm_to_radpers = 2*np.pi*3e10
invA_to_invm = 1e10

dimensionality = 2

chirality = [10,10]
m = chirality[0]
n = chirality[1]
a = 2.49e-10
L = a*np.sqrt(m**2 + n**2 + m*n)
d = np.gcd(2*m+n,2*n+m)
T_chirality = np.sqrt(3)*L/d
q_mul = np.pi/T_chirality

def omega(w_invcm):
   return w_invcm*invcm_to_radpers

def wavenum(norm_q):
    return norm_q*q_mul

def group_velocity(w_invcm,norm_q):
    vg = np.gradient(omega(w_invcm),wavenum(norm_q))
    return vg

def mean_free_path():
    return 100e-9

def fBE(w_invcm,T):
    f_BE = 1/(np.exp(h_ * omega(w_invcm) / (kB * T))-1)
    return f_BE

def df_dT(w_invcm,T):
    deriv = (fBE(w_invcm,T)**2)*np.exp(h_ *omega(w_invcm)/(kB*T))*(h_ * omega(w_invcm)/(kB*T**2))
    return deriv

def DOS(norm_q,dim):
    if dim==1:
        dos = 1/np.pi
    elif dim==2:
        dos = wavenum(norm_q)/(2*np.pi)
    elif dim==3:
        dos = wavenum(norm_q)**2/(2*np.pi**2)
    return dos

def sum_integral_factor(norm_q,dim):
    if dim==1:
        fac = 1
    elif dim==2:
        fac = 2*np.pi*wavenum(norm_q)
    elif dim==3:
        fac = 4*np.pi*wavenum(norm_q)**2
    return fac

def integrand(norm_q,w_invcm,T):
    return h_*omega(w_invcm)*df_dT(w_invcm,T)*DOS(norm_q,dimensionality)*group_velocity(w_invcm,norm_q)*mean_free_path()*sum_integral_factor(norm_q,dimensionality)

"""Write new functions below this line (starting with unit 4)."""
def main():
    
    q = []
    w = []
    vg = []

    fig,ax = plt.subplots()
    ax.set_title('Dispersion relationship for CNTs')
    ax.set_xlabel('Wavenumber [1/m]')
    ax.set_ylabel('Frequency [1/s]')
    ax.legend()

    fig2,ax2 = plt.subplots()
    ax2.set_title('Group Velcoity of CNT phonons')
    ax2.set_xlabel('Wavenumber [1/m]')
    ax2.set_ylabel('Group Velocity [m/s]')
    ax2.legend()

    for i in range(branch_count):
        q_in_invm = fit_k_invm[i]
        q.append(q_in_invm)

        w_in_radpers = omega(fit_w_invcm[i])
        w.append(w_in_radpers)

        ax.plot(q[i],w[i],label =f'Branch {i}')

        vg_in_mpers = group_velocity(fit_w_invcm[i],fit_k_invm[i])
        vg.append(vg_in_mpers)

        ax2.plot(q[i],vg[i], label = f'Branch {i}')

    T = np.linspace(1,1000,101)

    k = np.zeros(len(T))

    for i in range(len(T)):
        sum = 0
        for j in range(branch_count):
            q_values = np.linspace(np.min(q[j]), np.max(q[j]), 100)
            integrand_values = integrand(q_values, fit_w_invcm[j], T[i])
            sum += np.trapz(integrand_values, wavenum(q_values))
        k[i] = 1/3*sum

    fig3,ax3 = plt.subplots()

    ax3.plot(T,k)
    ax3.set_title('Temperature Dependence of Thermal Conductivity of CNTs')
    ax3.set_xlabel('Temperature [K]')
    ax3.set_ylabel('Thermal Conductivity [W/mK]')


    plt.show()


if __name__ == "__main__":
    main()