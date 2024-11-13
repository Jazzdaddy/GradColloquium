import numpy as np
import scipy as sp
from scipy.sparse import diags
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

Lx = 1
Ly = 1

Tf = 2

a = 1

def FOM(dx, dy, dt, a, plot=False):
    Nx = int(Lx/dx) + 1
    Ny = int(Ly/dy) + 1
    Nt = int(Tf/dt) + 1

    x_space = np.linspace(0, Lx, Nx)
    y_space = np.linspace(0, Ly, Ny)
    T = np.linspace(0, Tf, Nt)

    X, Y = np.meshgrid(x_space, y_space)

    U0 = a*np.exp(-(X-1/2)**2 -(Y-1/2)**2)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, U0)

    plt.show()

    D = (a/dx**2)*diags([1, -2, 1], [-1, 0, 1], (Nx, Nx))

    U_0 = sparse.csc_matrix(U0.reshape((Nx**2, 1)))

    # D = sparse.csc_matrix(D)

    I = sparse.eye(Nx)
    A = sparse.kron(I, D) + sparse.kron(D, I)
    m = A.shape[0]
    Iv = sparse.eye(m)
    # check1 = Iv - dt/2*A
    # B = sp.sparse.linalg.spsolve((Iv - dt/2*A), Iv)

    U = np.zeros((Nx**2, Nt))
    U[:, 0] = U0.reshape((Nx**2, ))

    progress_bar = tqdm(total=Nt, desc=f'FOM Solve    a: {a}', position=0, leave=True)

    for i in range(Nt-1):
        U_n = sparse.csc_matrix(U[:, i]).T
        U_n_1 = sparse.linalg.spsolve((Iv - dt/2*A), (Iv + dt/2*A)@U_n)
        U[:, i+1] = U_n_1

        progress_bar.update(1)

    Ugraph = U.reshape((Nx, Nx, Nt))

    if plot is True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Ugraph[:, :, 0])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        # time_text = ax.text(0.02, 0.95, f'{0}', transform=ax.transAxes)

        def update(frame, surf):
            surf.remove()
            surf = ax.plot_surface(X, Y, Ugraph[:, :, frame])
            # time_text.set_text(f'Time Step: {frame * dt:.3f}')
            return surf,

        ani = FuncAnimation(fig, update, frames=Nt, fargs= (surf),interval=1, blit=True)


        plt.show()

    print('check')

FOM(dx = 1e-2, dy = 1e-2, dt = 1e-3, a = 1, plot=True)