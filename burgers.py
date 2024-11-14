import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import time
from utils import error
from scipy.signal import gausspulse


def BurgersFOM(dx, dt, mu, plot=False):
    dx = dx
    dt = dt

    L = 1
    Tf = 1

    Nx = int(L / dx)+1
    m = Nx-2
    Nt = int(Tf / dt) + 1

    x_space = np.linspace(0, L, Nx)
    T = np.linspace(0, Tf, Nt)

    mu = mu
    nu = 1 / mu


    u0 = x_space / (1 + np.sqrt(1 / np.exp(mu / 8)) * np.exp(mu * x_space ** 2 / 4))
    # u0 = gausspulse((x_space - .5)/.5, fc=3)


    plt.plot(u0)
    plt.show()

    A = diags([-1, 0, 1], [-1, 0, 1], shape=(m, m)).toarray()
    # A[0, :] = 0
    # A[-1, :] = 0
    A = 1 / (2 * dx) * A

    L = diags([1, -2, 1], [-1, 0, 1], shape=(m, m)).toarray()
    # L[0, :] = 0
    # L[-1, :] = 0
    L = 1 / (mu * dx ** 2) * L

    X = np.zeros((m, Nt))
    u_ = u0[1:-1]
    X[:, 0] = u_

    # def RK4(u, A, L, dt):
    #    k1 = -np.diag(u) @ A @ u + L @ u
    #    k2 = -np.diag(u + dt / 2 * k1) @ A @ (u + dt / 2 * k1) + L @ (u + dt / 2 * k1)
    #    k3 = -np.diag(u + dt / 2 * k2) @ A @ (u + dt / 2 * k2) + L @ (u + dt / 2 * k2)
    #    k4 = -np.diag(u + dt * k3) @ A @ (u + dt * k3) + L @ (u + dt * k3)

    # def RK4(u, A, L, dt):
    #    k1 = -A @ (u*u)/2 + L @ u
    #    k2 = -A @ ((u + dt / 2 * k1)*(u + dt / 2 * k1))/2 + L @ (u + dt / 2 * k1)
    #    k3 = -A @ ((u + dt / 2 * k2)*(u + dt / 2 * k2))/2 + L @ (u + dt / 2 * k2)
    #    k4 = -A @ ((u + dt * k3)*(u + dt * k3))/2 + L @ (u + dt * k3)

       # return u + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * dt

    progress_bar = tqdm(total=Nt, desc=f'FOM Solve    mu: {mu}', position=0, leave=True)

    start_time = time.perf_counter()

    for i in range(Nt - 1):
       # Ud = np.diag(U[:, i])
       # B = np.linalg.inv(I + dt/2*Ud@A - dt/2*L)
       # C = I - dt/2*Ud@A + dt/2*L
       # X[:, i + 1] = RK4(X[:, i], A, L, dt)
       # u = X[:, i]
       X[:, i+1] = np.linalg.solve(np.eye(m)-dt*L, X[:, i] - dt*A@(X[:, i]*X[:, i])/2)
       # residual = X[:, i+1] - X[:, i] - dt*(-A@(X[:, i]*X[:, i])/2 + L@X[:, i+1])
       # norm = np.linalg.norm(residual)
       # U[:, i+1] = B@C@U[:, i]
       progress_bar.update(1)

    end_time = time.perf_counter()

    print(f'\n Time elapsed FOM: {end_time - start_time}')

    X_ = np.zeros((Nx, Nt))
    X_[1:-1] = X

    if plot is True:
       fig, ax = plt.subplots()
       line, = ax.plot(x_space, X_[:, 0])
       ax.set_xlabel('x')
       ax.set_ylabel('u')
       time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

       def update(frame):
           line.set_ydata(X_[:, frame])
           time_text.set_text(f'Time Step: {frame * dt:.3f}')
           return line, time_text

       ani = FuncAnimation(fig, update, frames=Nt, interval=1, blit=True)

       # ani.save('Burgersviscid.gif', writer='Pillow', fps=30)

       plt.show()
    return X_


def ROM(r, plot=False):

    dx = 2 ** (-8)
    dt = 1e-3

    L = 1
    Tf = 1

    Nx = int(L / dx) + 1
    Nt = int(Tf / dt) + 1

    x_space = np.linspace(0, L, Nx)
    T = np.linspace(0, Tf, Nt)

    mu = 500
    nu = 1 / mu

    X = np.load('Files/Burgers_FOM.npy')

    U, s = np.linalg.svd(X)[:2]

    np.save('Files/BurgersBasis.npy', U)

    U = np.load('Files/BurgersBasis.npy')
    #
    # r_bases = np.arange(0, 105, 5)
    # error_proj = np.zeros(r_bases.shape[0])
    # for i, r in enumerate(r_bases):
    #    Phi = U[:, :r]
    #
    #    error_proj[i] = np.linalg.norm(X - Phi @ Phi.T @ X)
    #
    # plt.plot(r_bases, error_proj, '-o', markersize=4)
    # plt.yscale('log')
    # plt.show()

    # r = 10
    Phi = U[:, :r]

    A = diags([-1, 0, 1], [-1, 0, 1], shape=(Nx, Nx)).toarray()
    A[0, :] = 0
    A[-1, :] = 0
    A = 1 / (2 * dx) * A

    L = diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)).toarray()
    L[0, :] = 0
    L[-1, :] = 0
    L = 1 / (mu * dx ** 2) * L

    # Ar = A @ Phi
    Lr = Phi.T @ L @ Phi

    T = np.zeros((Nx, r**2))

    for i in range(Nx):
       T[i, :] = np.kron(Phi[i, :], Phi[i, :])

    Br = 1/2*Phi.T @ A @ T

    # def RK4_ROM(u, Br, Lr, dt, Phi=Phi):
    #    k1 = -Br @ np.kron(u, u) + Lr @ u
    #    k2 = -Br @ np.kron((u + dt / 2 * k1), (u + dt / 2 * k1)) + Lr @ (u + dt / 2 * k1)
    #    k3 = -Br @ np.kron((u + dt / 2 * k2), (u + dt / 2 * k2)) + Lr @ (u + dt / 2 * k2)
    #    k4 = -Br @ np.kron((u + dt * k3), (u + dt * k3)) + Lr @ (u + dt * k3)
    #
    #    return u + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * dt

    # def RK4_ROM(u, Br, Lr, dt, Phi=Phi):
    #     k1 = -Phi.T @ np.diag(Phi @ u) @ Ar @ u + Lr @ u
    #     k2 = -Phi.T @ np.diag(Phi @ (u + dt / 2 * k1)) @ Ar @ (u + dt / 2 * k1) + Lr @ (u + dt / 2 * k1)
    #     k3 = -Phi.T @ np.diag(Phi @ (u + dt / 2 * k2)) @ Ar @ (u + dt / 2 * k2) + Lr @ (u + dt / 2 * k2)
    #     k4 = -Phi.T @ np.diag(Phi @ (u + dt * k3)) @ Ar @ (u + dt * k3) + Lr @ (u + dt * k3)
    #
    #     return u + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * dt


    Xr = np.zeros((r, Nt))

    Xr[:, 0] = Phi.T @ X[:, 0]

    C = np.linalg.inv(np.eye(r) - dt * Lr)

    progress_bar = tqdm(total=Nt, desc='ROM Solve', position=0, leave=True)
    start_time = time.perf_counter()

    for i in range(Nt - 1):
       # Ud = np.diag(U[:, i])
       # B = np.linalg.inv(I + dt/2*Ud@A - dt/2*L)
       # C = I - dt/2*Ud@A + dt/2*L
       Xr[:, i+1] = C @ (Xr[:, i] - dt * Br @ (np.kron(Xr[:, i], Xr[:, i])))
       # U[:, i+1] = B@C@U[:, i]
       progress_bar.update(1)

    end_time = time.perf_counter()

    print(f'\n Time elapsed POD-Galerkin: {end_time - start_time}')

    np.save('Files/BurgersROM.npy', Xr)

    Xr = np.load('Files/BurgersROM.npy')

    Xrec = Phi @ Xr

    err = error(X, Xrec, dx, dt)

    print(f'Error POD-Galerkin: {err}')

    # plt.plot(x_space, X[:, 100], label='FOM')
    # plt.plot(x_space, Xrec[:, 100], label='ROM')
    # plt.legend()
    # plt.show()
    if plot is True:

       fig, ax = plt.subplots()
       line1, = ax.plot(x_space, X[:, 0], label='FOM')
       line2, = ax.plot(x_space, Xrec[:, 0], label='POD-Galerkin')
       ax.set_title('Viscous Burgers Wave Equation')
       ax.legend()
       ax.set_xlabel('x')
       ax.set_ylabel('u')
       r_text = ax.text(0.02, 0.9, f'r: {r}', transform=ax.transAxes)
       time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)


       def update(frame):
           line1.set_ydata(X[:, frame])
           line2.set_ydata(Xrec[:, frame])

           time_text.set_text(f't: {frame * dt:.3f}')
           return line1, line2, time_text


       ani = FuncAnimation(fig, update, frames=Nt, interval=1, blit=True)

       # ani.save('Files/BurgersPODGalerkin.gif', writer='Pillow', fps=30)

       plt.show()

    return err

# r_bases = np.arange(5, 105, 5)
#
# err = np.zeros((1, 20))
# for i, r in enumerate(r_bases):
#    print(r)
#    err[0,i] = ROM(r)
#
# np.save('Files/errorBurgersGal.npy', err)
# # err = np.load()
# plt.scatter(r_bases, err[0])
# plt.yscale('log')
# plt.show()

if __name__ == '__main__':
    dx = 2 ** (-8)
    dt = 1e-3
    mu = 500
    X = BurgersFOM(dx = dx, dt = dt, mu = mu, plot=True)
    np.save('Files/Burgers_FOM.npy', X)
    X = np.load('Files/Burgers_FOM.npy')
    U = np.linalg.svd(X)[0]

    ROM(10, plot=True)

    r_bases = np.arange(5, 105, 5)

    err = np.zeros((1, 20))
    for i, r in enumerate(r_bases):
        print(r)
        err[0, i] = ROM(r, plot=False)

    np.save('Files/errorBurgersGal.npy', err)
    # err = np.load()
    plt.scatter(r_bases, err[0])
    plt.yscale('log')
    plt.show()