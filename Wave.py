import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from scipy.signal import gausspulse, gauss_spline
import time

from utils import error


def wave(dx, dt, plot=False):
    L = 1
    Tf = 2

    c = 1

    Nx = int(L / dx)
    Nt = int(Tf / dt) + 1

    mesh = np.linspace(0, L, Nx + 1)
    x = mesh[:-1]

    T = np.linspace(0, Tf, Nt)

    # u0 = np.sin(2*np.pi * x + 1)
    # u0 = gausspulse((x - .5) / .5, fc=1)
    # u0 = gauss_spline((x-.5)/.5, n=2)
    def cubic_spline(pos, u=10.0):
        s = u * np.abs(pos - 0.5)
        h = np.zeros_like(s)

        mask_range1 = (0 <= s) & (s <= 1)
        mask_range2 = (1 < s) & (s <= 2)

        h[mask_range1] = 1 - 1.5 * s[mask_range1] ** 2 + 0.75 * s[mask_range1] ** 3
        h[mask_range2] = 0.25 * (2 - s[mask_range2]) ** 3

        return h

    u0 = cubic_spline(x)

    plt.plot(x, u0)
    plt.show()

    v0 = np.zeros(Nx)

    X = np.zeros((2 * Nx, Nt))

    X[:Nx, 0] = u0
    X[Nx:, 0] = v0

    A = diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)).toarray()
    A[0, -1] = 1
    A[-1, 0] = 1

    A = (c ** 2 / dx ** 2) * A

    zero = np.zeros((Nx, Nx))

    J = np.block([[zero, np.eye(Nx)], [-np.eye(Nx), zero]])
    dH = np.block([[-A, zero], [zero, np.eye(Nx)]])

    # b = np.block([[zero, np.eye(Nx)], [A, zero]])

    I = np.eye(2 * Nx)

    L = I - .5 * dt * (J @ dH)
    R = I + .5 * dt * (J @ dH)

    print('Solving B')
    B = np.linalg.solve(L, R)

    progress_bar = tqdm(total=Nt, desc=f'FOM Solve', position=0, leave=True)

    start_time = time.perf_counter()
    for i in range(Nt - 1):
        X[:, i + 1] = B @ X[:, i]
        progress_bar.update(1)

    end_time = time.perf_counter()

    print(f'\n Time Elapsed FOM: {end_time-start_time}')

    X_mesh, Y_mesh = np.meshgrid(x, T)

    if plot is True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(X_mesh, Y_mesh, X[:Nx, :].T, cmap='coolwarm', antialiased=True)

        plt.show()

        fig, ax = plt.subplots()
        line, = ax.plot(x, X[:Nx, 0])
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def update(frame):
            line.set_ydata(X[:Nx, frame])
            time_text.set_text(f't: {frame * dt:.3f}')
            return line, time_text

        ani = FuncAnimation(fig, update, frames=Nt, interval=1, blit=True)

        plt.show()

    return X

def Hamiltonian(X, dx, Nt, c=1):
    H = np.zeros(Nt)
    Nx = int(X.shape[0]/2)
    A = diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)).toarray()
    A[0, -1] = 1
    A[-1, 0] = 1

    A = (c ** 2 / dx ** 2) * A

    progress_bar = tqdm(total=Nt, desc=f'Hamiltonian', position=0, leave=True)

    for j in range(Nt):
        U = X[:Nx, j]
        V = X[Nx:, j]
        H[j] = 1/2*(-U.T@A@U*dx + V.T@V*dx)
        progress_bar.update(1)
    return H





X = wave(dx=1e-3, dt=1e-3, plot=True)
np.save('Files/WaveFOM.npy', X)
X = np.load('Files/WaveFOM.npy')
HFOM = Hamiltonian(X, dx=1e-3, Nt = 2001)

plt.plot(HFOM)
plt.show()

plt.plot(HFOM)
plt.ylim(7.49, 7.50)
plt.show()



def WaveROM(r, dx=1e-3, dt=1e-3, Tf=2, plot=False):
    L = 1
    # Tf = 2

    c = 1

    Nx = int(L / dx)
    Nt = int(Tf / dt) + 1

    mesh = np.linspace(0, L, Nx + 1)
    x = mesh[:-1]

    T = np.linspace(0, Tf, Nt)

    # u0 = np.sin(2*np.pi * x + 1)
    # u0 = gausspulse((x - .5) / .5, fc=1)
    # v0 = np.zeros(Nx)

    X = np.load('Files/WaveFOM.npy')
    np.save('Files/WaveBasisu.npy', np.linalg.svd(X[:Nx, :])[0])
    np.save('Files/WaveBasisv.npy', np.linalg.svd(X[Nx:, :])[0])

    Phi_u = np.load('Files/WaveBasisu.npy')[:, :r]
    Phi_v = np.load('Files/WaveBasisv.npy')[:, :r]

    zero_ = np.zeros((Nx, r))
    Phi = np.block([[Phi_u, zero_], [zero_, Phi_v]])

    Y = np.zeros((2*r, Nt))
    Y[:, 0] = Phi.T @ X[:, 0]

    A = diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)).toarray()
    A[0, -1] = 1
    A[-1, 0] = 1

    A = (c ** 2 / dx ** 2) * A

    zero = np.zeros((Nx, Nx))

    J = np.block([[zero, np.eye(Nx)], [-np.eye(Nx), zero]])
    dH = np.block([[-A, zero], [zero, np.eye(Nx)]])

    Jr = Phi.T@J@Phi
    dHr = Phi.T@dH@Phi

    I = np.eye(2*r)

    L = I - .5 * dt * (Jr @ dHr)
    R = I + .5 * dt * (Jr @ dHr)

    print('Solving B')
    B = np.linalg.solve(L, R)

    progress_bar = tqdm(total=Nt, desc=f'FOM Solve', position=0, leave=True)

    start_time = time.perf_counter()
    for i in range(Nt - 1):
        Y[:, i + 1] = B @ Y[:, i]
        progress_bar.update(1)

    end_time = time.perf_counter()

    print(f'\n Time Elapsed ROM: {end_time-start_time}')


    X_mesh, Y_mesh = np.meshgrid(x, T)

    Yapprox = Phi@Y

    if plot is True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_surface(X_mesh, Y_mesh, Yapprox[:Nx, :].T, cmap='coolwarm', antialiased=True)

        plt.title('ROM')

        plt.show()

        fig, ax = plt.subplots()
        line1, = ax.plot(x, X[:Nx, 0], label='FOM')
        line2, = ax.plot(x, Yapprox[:Nx, 0], label='ROM')
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend()
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

        def update(frame):
            line1.set_ydata(X[:Nx, frame])
            line2.set_ydata(Yapprox[:Nx, frame])

            time_text.set_text(f't: {frame * dt:.3f}')
            return line1, line2, time_text

        ani = FuncAnimation(fig, update, frames=Nt, interval=1, blit=True)


        plt.show()

    return Yapprox



Y = WaveROM(10, plot=True)
HROM = Hamiltonian(Y, dx=1e-3, Nt = 2001)
plt.plot(HROM, label='ROM')
# plt.plot(HFOM, label='FOM')
plt.legend()
plt.show()

r_bases = np.arange(5, 105, 5)


err = np.zeros((1, 20))
for i, r in enumerate(r_bases):
    print(r)

    err[0, i] = error(X, WaveROM(r, plot=False),dx=1e-3, dt=1e-3)
    print(f'error: {err[0, i]}')


plt.scatter(r_bases, err[0])
plt.yscale('log')
plt.show()
#
# Y = WaveROM(10, Tf = 100, plot=False)
#
# H = Hamiltonian(Y, dx=1e-3, Nt = 100001)
# np.save('Files/WaveLTHam.npy', H)
# HLT = np.load('Files/WaveLTHam.npy')
# plt.plot(HLT)
# plt.show()


