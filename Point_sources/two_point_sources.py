import numpy as np
from numpy import sqrt, exp
import matplotlib.pyplot as plt
from matplotlib import animation
# import myfig
plt.rc('lines', linewidth=2)
plt.rc('font', size=14)
plt.rc('axes', linewidth=1.5, labelsize=14)
plt.rc('legend', fontsize=14)

class Monopole():
    def __init__(self, coords, Q, phase=0):
        """monopole fonction

        Args:
            coords (tuple): position m
            Q (float): debit volumique m^3/s
            phase (float, optional): rad. Defaults to 0.
        """
        self.x, self.y = coords  # m
        self.Q = Q  # m^3/s
        self.phase = phase  # rad

    def func(self, freq, x, y, c, rho, t):
        r = sqrt((x - self.x)**2+(y - self.y)**2)
        k = 2*np.pi*freq/c
        omega = 2*np.pi*freq
        mono = 1j*rho*omega*self.Q / \
            (4*np.pi) * exp(1j*(omega*t - k*r + self.phase))/r
        return mono

name_gif = 'two_sources.gif'
# Grid parameters
N = 200
xmin, xmax = -5, 5  # m
ymin, ymax = -5, 5  # m
x, y = np.linspace(xmin, xmax, N), np.linspace(ymin, ymax, N)
X, Y = np.meshgrid(x, y)

# Phisical parameters
c = 343  # m/s
freq = 250  # Hz
rho = 1.225  # kg/m3
d = 2.5  # distance between sources m
Q = 1e-2  # debit m^3/s
position_1 = (0, d/2)
position_2 = (0, -d/2)

# case 1
pressure_phi = np.zeros(X.shape)

source_1_1 = Monopole(position_1, Q)
source_2_1 = Monopole(position_2, Q)

pressure_phi = source_1_1.func(
    freq, X, Y, c, rho, t=0) + source_2_1.func(freq, X, Y, c, rho, t=0)

# case 2
pressure_outphi = np.zeros(X.shape)

source_1_2 = Monopole(position_1, Q)
source_2_2 = Monopole(position_2, -Q)

pressure_outphi = source_1_2.func(
    freq, X, Y, c, rho, t=0) + source_2_2.func(freq, X, Y, c, rho, t=0)


# Animation
clim = (-2, 2)  # color lim

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))

case1 = ax.imshow(np.real(pressure_phi), vmin=clim[0], vmax=clim[1], origin='lower', aspect='equal',
                  extent=[xmin, xmax, ymin, ymax])

case2 = ax2.imshow(np.real(pressure_outphi), vmin=clim[0], vmax=clim[1], origin='lower', aspect='equal',
                   extent=[xmin, xmax, ymin, ymax])


ax.set_xlabel('$x~(m)$')
ax2.set_xlabel('$x~(m)$')
ax.set_ylabel('$y~(m)$')
ax2.set_ylabel('$y~(m)$')
fig.suptitle('Pression rayonnée par deux points sources')
ax.set_title('1')
ax2.set_title('2')
# fig.tight_layout()
# plt.close()


def init():
    case1.set_array(np.real(pressure_phi))
    case2.set_array(np.real(pressure_outphi))
    return (case1, case2)


def animate(i):
    case1.set_array(
        np.array(np.real(pressure_phi*np.exp(1j*(i/24)*2*np.pi))))
    case2.set_array(
        np.array(np.real(pressure_outphi*np.exp(1j*(i/24)*2*np.pi))))
    return (case1, case2,)


my = animation.FuncAnimation(fig, animate, init_func=init,
                             frames=24, interval=30, blit=False, repeat=True)
plt.show()
# my.save(name_gif, writer='imagemagick', fps=24, dpi=150)# saving as gif, need to install imagemagick
