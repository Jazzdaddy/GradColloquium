import numpy as np
import scipy as sp
from scipy.sparse import diags
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import time

from utils import error


Lx = 1
Ly = 1

Tf = 1

a = 1


def FOM(dx, dy, dt, a, plot=False):
    Nx = int(Lx / dx) + 1
    Ny = int(Ly / dy) + 1
    Nt = int(Tf / dt) + 1

    x_space = np.linspace(0, Lx, Nx)
    y_space = np.linspace(0, Ly, Ny)
    T = np.linspace(0, Tf, Nt)

    X, Y = np.meshgrid(x_space, y_space)

    U0 = a * np.exp(5*(-(X - 1 / 2) ** 2 - (Y - 1 / 2) ** 2))

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, U0, cmap='coolwarm')
    ax.set_zlim(0,1)

    plt.show()

    n = Nx-2
    D = (a / dx ** 2) * diags([1, -2, 1], [-1, 0, 1], (Nx, Nx)).toarray()
    D[0, :] = 0
    D[-1, :] = 0
    # bd = np.zeros((Nx, Nx))
    # bd[0, :] = a * np.exp(-(X[0, :] - 1 / 2) ** 2 - (Y[0, :] - 1 / 2) ** 2)
    # bd[-1, :] = a * np.exp(-(X[-1, :] - 1 / 2) ** 2 - (Y[-1, :] - 1 / 2) ** 2)
    # bd[:, 0] = a * np.exp(-(X[:, 0] - 1 / 2) ** 2 - (Y[:, 0] - 1 / 2) ** 2)
    # bd[:, -1] = a * np.exp(-(X[:, -1] - 1 / 2) ** 2 - (Y[:, -1] - 1 / 2) ** 2)
    f = np.zeros((Nx, Nx))
    f[int(Nx/4): int(Nx*3/4), int(Nx/4): int(Nx*3/4)] = 0.001*(np.sin(X[int(Nx/4): int(Nx*3/4),
                                int(Nx/4): int(Nx*3/4)]-1/2)**2
                                 + np.cos(Y[int(Nx/4): int(Nx*3/4),
                                int(Nx/4): int(Nx*3/4)]-1/2)**2)
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, f, cmap='coolwarm')
    ax.set_zlim(0, 1)

    plt.show()

    f_ = sparse.csc_matrix(f.reshape(Nx**2, 1))

    # D[0, :] = 0
    # D[-1, :] = 0
    # D[0, 0] = 1
    # D[-1, -1] = 1

    U_0 = sparse.csc_matrix(U0.reshape((Nx ** 2, 1)))

    D = sparse.csc_matrix(D)

    I = sparse.eye(Nx)
    A = sparse.kron(I, D) + sparse.kron(D, I)
    m = A.shape[0]
    Iv = sparse.eye(m)
    # check1 = Iv - dt/2*A
    # B = sp.sparse.linalg.spsolve((Iv - dt/2*A), Iv)

    U = np.zeros((Nx ** 2, Nt))
    U[:, 0] = U0.reshape((Nx ** 2,))

    progress_bar = tqdm(total=Nt, desc=f'FOM Solve    a: {a}', position=0, leave=True)

    for i in range(Nt-1):
        U_n = sparse.csc_matrix(U[:, i]).T
        U_n_1 = sparse.linalg.spsolve((Iv - dt/2*A), (Iv + dt/2*A)@U_n + f_)
        U[:, i+1] = U_n_1

        progress_bar.update(1)

    np.save('HeatFOM.npy', U)
    # U = np.load('Files/HeatFOM.npy')

    Ugraph = U.reshape((Nx, Nx, Nt))

    diff = U0 - Ugraph[:, :, 0]

    if plot is True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        U_min, U_max = np.min(Ugraph), np.max(Ugraph)

        surf = ax.plot_surface(X, Y, Ugraph[:, :, 0], cmap='coolwarm', vmin=U_min, vmax=U_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')

        time_text = ax.text3D(0.02, 0.95, 1, '', transform=ax.transAxes)

        def update(frame):
            ax.clear()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            ax.set_zlim(0, 1)

            surf = ax.plot_surface(X, Y, Ugraph[:, :, frame], cmap="coolwarm", vmin=U_min, vmax=U_max)

            time_text.set_text(f'Time Step: {frame * dt:.3f}')
            return surf, time_text

        ani = FuncAnimation(fig, update, frames=Nt, interval=1, blit=False)

        plt.show()

    return U

U = FOM(dx=1e-2, dy=1e-2, dt=1e-3, a=1, plot=True)
U = np.load('Files/HeatFOM.npy')

# Usp = sparse.csc_matrix(U)
# Phi, s = np.linalg.svd(U)[:2]
#
# np.save('HeatFOMbasis.npy', Phi)
# np.save('HeatFOMsv.npy', s)
Phi = np.load('Files/HeatFOMbasis.npy')
s = np.load('Files/HeatFOMsv.npy')[:100]

plt.plot(s)
plt.yscale('log')
plt.show()

def ROM(r, dx=1e-2, dy=1e-2, dt=1e-3, a=1, plot=False):
    Nx = int(Lx / dx) + 1
    Ny = int(Ly / dy) + 1
    Nt = int(Tf / dt) + 1

    x_space = np.linspace(0, Lx, Nx)
    y_space = np.linspace(0, Ly, Ny)
    # T = np.linspace(0, Tf, Nt)

    X, Y = np.meshgrid(x_space, y_space)

    Phi = np.load('Files/HeatFOMbasis.npy')[:, :r]

    D = (a / dx ** 2) * diags([1, -2, 1], [-1, 0, 1], (Nx, Nx)).toarray()
    D[0, :] = 0
    D[-1, :] = 0

    D = sparse.csc_matrix(D)

    I = sparse.eye(Nx)
    A = sparse.kron(I, D) + sparse.kron(D, I)

    A = A.toarray()

    Ar = Phi.T@A@Phi

    f = np.zeros((Nx, Nx))
    f[int(Nx / 4): int(Nx * 3 / 4), int(Nx / 4): int(Nx * 3 / 4)] = 0.001 * (np.sin(X[int(Nx / 4): int(Nx * 3 / 4),
                                                                                    int(Nx / 4): int(
                                                                                        Nx * 3 / 4)] - 1 / 2) ** 2
                                                                             + np.cos(Y[int(Nx / 4): int(Nx * 3 / 4),
                                                                                      int(Nx / 4): int(
                                                                                          Nx * 3 / 4)] - 1 / 2) ** 2)
    f = f.reshape(Nx**2, 1)
    fr = Phi.T@f

    U0 = np.load('Files/HeatFOM.npy')[:, 0]

    y0 = Phi.T@U0

    K = np.zeros((r, Nt))
    K[:, 0] = y0

    I = np.eye(r)

    B = np.linalg.inv((I-dt/2*Ar))
    C = B@(I + dt/2*Ar)
    F = (B@fr).flatten()

    # progress_bar = tqdm(total=Nt, desc=f'FOM Solve    a: {a}', position=0, leave=True)
    start_time = time.perf_counter()
    for i in range(Nt-1):
        K[:, i+1] = C@K[:, i] + F
        # solve = C@K[:, i].reshape(r, 1) + F
        # K[:, i+1] = solve.reshape(r, )

        # progress_bar.update(1)
    end_time = time.perf_counter()

    Uapprox = Phi@K


    cpu_time = end_time - start_time
    print(f"\n CPU time: {cpu_time:.6f} seconds")

    Ugraph = Uapprox.reshape(Nx, Nx, Nt)

    if plot is True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        U_min, U_max = np.min(Ugraph), np.max(Ugraph)

        surf = ax.plot_surface(X, Y, Ugraph[:, :, 0], cmap='coolwarm', vmin=U_min, vmax=U_max)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')

        time_text = ax.text3D(0.02, 0.95, 1, '', transform=ax.transAxes)

        def update(frame):
            ax.clear()  
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            ax.set_zlim(0, 1)

            surf = ax.plot_surface(X, Y, Ugraph[:, :, frame], cmap="coolwarm", vmin=U_min, vmax=U_max)

            time_text.set_text(f'Time Step: {frame * dt:.3f}')
            return surf, time_text

        ani = FuncAnimation(fig, update, frames=Nt, interval=1, blit=False)

        plt.show()

    return Uapprox

Uapprox = ROM(10, plot=False)

print(error(U, Uapprox, dx=1e-2, dt=1e-3))

errors = np.zeros((11))

r_vals = np.arange(22, step=2)

for i,r in enumerate(r_vals):
    Uapprox = ROM(r)
    errors[i] = error(U, Uapprox, dx=1e-2, dt=1e-3)

plt.plot(r_vals, errors)
plt.yscale('log')
plt.xlabel('r')
plt.ylabel('error')
plt.show()





