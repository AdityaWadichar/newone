import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
time=np.arange(0, 10, 0.01)
theta=np.zeros(1000)
thd=np.pi/20
thdr=np.ones(1000)*thd
uex=np.zeros(1000)
kp=500
ki=2
kd=0
dt=0.01

m=5
g=9.8
l=1

th=0
dth=0
e=0
old_e=0
E=0
e_dot=0

for i in range(1000):
    # if i>100 and i<105:
    #     uex[i]=5
    if i>500 and i<505:
        uex[i]=-10

for i in range(1000):
    e=thd-th



    u= kp*e+kd*e_dot+ki*E

    ddth=(u+uex[i]-m*g*l*np.sin(th))/m*l**2
    dth=dth+ddth*dt
    th=th+dth*dt
    th=np.arctan2(th, 1)
    theta[i]=th

    e_dot = e - old_e
    old_e=e
    E = E + e
print(u)
#theta=np.arctan2(theta, np.ones(1000))

plt.plot(time, theta, time, thdr)
plt.grid(True)
plt.show()

x1 = 0.0
y1 = 0.0

x2 = l*np.sin(theta)
y2 = -l*np.cos(theta)
# print(y2)
# print(cos(th))

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False,aspect='equal',
					 xlim=(-1.5*l, 1.5*l), ylim=(-1.5*l, 1.5*l))
ax.grid()

line, = ax.plot([], [], '-o', lw=2)
time_template = 'time = %.01f s'
thet_template = 'theta = %1f'
thetd_template = 'desired = %1f'
uext_template = 'disturbance = %.1f N.m'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
thet = ax.text(0.05, 0.85, '', transform=ax.transAxes)
thetd = ax.text(0.05, 0.8, '', transform=ax.transAxes)
uext = ax.text(0.05, 0.75, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    thet.set_text('')
    thetd.set_text('')
    uext.set_text('')
    return line, time_text, thet, thetd, uext

def animate(i):
    global N
    global T
    thisx = [x1, x1+x2[i]]
    thisy = [y1, y2[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template%(i*dt))
    thet.set_text(thet_template%theta[i])
    t=thdr*np.ones(1000)
    thetd.set_text(thetd_template % t[i])
    uext.set_text(uext_template % uex[i])
    return line, time_text, thet, thetd, uext

ani = animation.FuncAnimation(fig, animate,
    interval=1, blit=True, init_func=init)


plt.show()


