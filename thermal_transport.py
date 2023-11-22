import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from dispersion_data import fit_k_invA, fit_E_meV, branch_count

h_ = 1.055e-34
kB = 1.38e-23
meV_to_J = 1.6021773e-22
invA_to_invm = 1e10

def E_to_omega(E_meV):
    w = E_meV*meV_to_J/h_
    return w

def group_velocity(E_meV,q):
    vg = np.gradient(E_to_omega(E_meV),q)
    return vg

def mean_free_path():
    return 350e-9

def fBE(E_meV,T):
    f_BE = 1/(np.exp(h_ * E_to_omega(E_meV) / (kB * T))-1)
    return f_BE

def df_dT(E_meV,T):
    deriv = (fBE(E_meV,T)**2)*np.exp(h_ *E_to_omega(E_meV)/(kB*T))*(h_ * E_to_omega(E_meV)/(kB*T**2))
    return deriv

def DOS(q):
    dos_3d = q**2/(2*np.pi**2)
    return dos_3d

def integrand(q,E_meV,T):
    return h_*E_to_omega(E_meV)*df_dT(E_meV,T)*DOS(q)*group_velocity(E_meV,q)*mean_free_path()*4*np.pi*q**2

"""Write new functions below this line (starting with unit 4)."""
def main():
    
    q = []
    omega = []
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
        q_in_invm = fit_k_invA[i]*invA_to_invm
        q.append(q_in_invm)

        omega_in_invs = E_to_omega(fit_E_meV[i])
        omega.append(omega_in_invs)

        ax.plot(q[i],omega[i],label =f'Acoustic Branch {i}')

        vg_in_mpers = group_velocity(omega[i],q[i])
        vg.append(vg_in_mpers)

        ax2.plot(q[i],vg[i], label = f'Acoustic Branch {i}')

    T = np.linspace(1,1000,101)

    k = np.zeros(len(T))

    for i in range(len(T)):
        sum = 0
        for j in range(branch_count):
            q_values = np.linspace(np.min(q[j]), np.max(q[j]), 100)
            integrand_values = integrand(q_values, fit_E_meV[j], T[i])
            sum += np.trapz(integrand_values, q_values)
        k[i] = 1/3*sum

    fig3,ax3 = plt.subplots()

    ax3.plot(T,k)
    ax3.set_title('Temperature Dependence of Thermal Conductivity of CNTs')
    ax3.set_xlabel('Temperature [K]')
    ax3.set_ylabel('Thermal Conductivity [W/mK]')


    plt.show()


if __name__ == "__main__":
    main()