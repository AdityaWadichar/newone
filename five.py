import casadi as ca
import numpy as np
from celluloid import Camera
from matplotlib import pyplot as plt

################
# main code starts after line 360
################


###############
# class five_link for five link biped model
###############

class five_link():
    def __init__(self, start, stop, qin, dqin, uin, N, T):
        self.opti= ca.Opti()

        #masses
        self.m1= 0.5
        self.m2= 0.5
        self.m3= 0.5
        self.m4= self.m2
        self.m5= self.m1

        #lengths
        self.l1= 0.5
        self.l2= 0.5
        self.l3= 0.5
        self.l4= self.l2
        self.l5= self.l1

        self.P0=[0,0]

        self.g=-9.81
        self.T= T
        self.N= N
        self.h= self.T/(self.N)

        self.initial= start
        self.final= stop
        self.qin=qin
        self.dqin=dqin
        self.uin=uin

        self.max_tau= np.inf
        self.max_angle= np.pi
        self.step_max=1

        self.i1= self.m1*(self.l1**2)/12
        self.i2= self.m2*(self.l2**2)/12
        self.i3= self.m3*(self.l3**2)/12
        self.i4= self.m4*(self.l4**2)/12
        self.i5= self.m5*(self.l5**2)/12

        self.state=[]
        self.u=[]

        for i in range(self.N):
            q=[]
            tau=[]

            q.append(self.opti.variable(5))
            q.append(self.opti.variable(5))
            tau.append(self.opti.variable(4))

            self.state.append(q)
            self.u.append(tau)

        self.pos = []
        self.com = []
        self.ddq = []
        for i in range(self.N):
            P, dP, G, dG, ddG, ddq = self.Dynamics(self.state[i], self.u[i])
            self.pos.append(P)
            self.com.append(G)
            self.ddq.append(ddq)

            if i == 0:
                self.dp0 = dP
            if i == self.N - 1:
                self.dpN = dP
            self.impactmap = self.heel_strike(self.state[i][0], self.state[i][1], P, dP, G, dG)


    def Kinematics(self, q, dq):
        p1x = ca.MX.sym('p1x', 1)
        p1y = ca.MX.sym('p1y', 1)
        p2x = ca.MX.sym('p2x', 1)
        p2y = ca.MX.sym('p2y', 1)
        p3x = ca.MX.sym('p3x', 1)
        p3y = ca.MX.sym('p3y', 1)
        p4x = ca.MX.sym('p4x', 1)
        p4y = ca.MX.sym('p4y', 1)
        p5x = ca.MX.sym('p5x', 1)
        p5y = ca.MX.sym('p5y', 1)

        g1x = ca.MX.sym('g1x', 1)
        g1y = ca.MX.sym('g1y', 1)
        g2x = ca.MX.sym('g2x', 1)
        g2y = ca.MX.sym('g2y', 1)
        g3x = ca.MX.sym('g3x', 1)
        g3y = ca.MX.sym('g3y', 1)
        g4x = ca.MX.sym('g4x', 1)
        g4y = ca.MX.sym('g4y', 1)
        g5x = ca.MX.sym('g5x', 1)
        g5y = ca.MX.sym('g5y', 1)

        p1x = -self.l1*ca.sin(q[0])
        p1y = self.l1*ca.cos(q[0])
        g1x = -self.l1 * ca.sin(q[0]) / 2
        g1y = self.l1 * ca.cos(q[0]) / 2

        p2x = p1x + (-self.l2*ca.sin(q[1]))
        p2y = p1y + (self.l2 * ca.cos(q[1]))
        g2x = p1x + (-self.l2 * ca.sin(q[1])) / 2
        g2y = p1y + (self.l2 * ca.cos(q[1])) / 2

        p3x = p2x + (self.l3 * ca.sin(q[2]))
        p3y = p2y + (self.l3 * ca.cos(q[2]))
        g3x = p2x + (self.l3 * ca.sin(q[2])) / 2
        g3y = p2y + (self.l3 * ca.cos(q[2])) / 2

        p4x = p2x + (self.l4 * ca.sin(q[3]))
        p4y = p2y + (-self.l4 * ca.cos(q[3]))
        g4x = p2x + (self.l4 * ca.sin(q[3])) / 2
        g4y = p2y + (-self.l4 * ca.cos(q[3])) / 2

        p5x = p4x + (self.l5 * ca.sin(q[4]))
        p5y = p4y + (-self.l5 * ca.cos(q[4]))
        g5x = p4x + (self.l5 * ca.sin(q[4])) / 2
        g5y = p4y + (-self.l5 * ca.cos(q[4])) / 2

        dp1x, dp1y = ca.jtimes(p1x, q, dq), ca.jtimes(p1y, q, dq)
        dp2x, dp2y = ca.jtimes(p2x, q, dq), ca.jtimes(p2y, q, dq)
        dp3x, dp3y = ca.jtimes(p3x, q, dq), ca.jtimes(p3y, q, dq)
        dp4x, dp4y = ca.jtimes(p4x, q, dq), ca.jtimes(p4y, q, dq)
        dp5x, dp5y = ca.jtimes(p5x, q, dq), ca.jtimes(p5y, q, dq)

        dg1x, dg1y = ca.jtimes(g1x, q, dq), ca.jtimes(g1y, q, dq)
        dg2x, dg2y = ca.jtimes(g2x, q, dq), ca.jtimes(g2y, q, dq)
        dg3x, dg3y = ca.jtimes(g3x, q, dq), ca.jtimes(g3y, q, dq)
        dg4x, dg4y = ca.jtimes(g4x, q, dq), ca.jtimes(g4y, q, dq)
        dg5x, dg5y = ca.jtimes(g5x, q, dq), ca.jtimes(g5y, q, dq)

        ddg1x, ddg1y = ca.jtimes(g1x, dq, dq), ca.jtimes(g1y, dq, dq)
        ddg2x, ddg2y = ca.jtimes(g2x, dq, dq), ca.jtimes(g2y, dq, dq)
        ddg3x, ddg3y = ca.jtimes(g3x, dq, dq), ca.jtimes(g3y, dq, dq)
        ddg4x, ddg4y = ca.jtimes(g4x, dq, dq), ca.jtimes(g4y, dq, dq)
        ddg5x, ddg5y = ca.jtimes(g5x, dq, dq), ca.jtimes(g5y, dq, dq)


        P = [[p1x, p1y], [p2x, p2y], [p3x, p3y], [p4x, p4y], [p5x, p5y]]
        dP = [[dp1x, dp1y], [dp2x, dp2y], [dp3x, dp3y], [dp4x, dp4y], [dp5x, dp5y]]

        G = [[g1x, g1y], [g2x, g2y], [g3x, g3y], [g4x, g4y], [g5x, g5y]]
        dG = [[dg1x, dg1y], [dg2x, dg2y], [dg3x, dg3y], [dg4x, dg4y], [dg5x, dg5y]]
        ddG = [[ddg1x, ddg1y], [ddg2x, ddg2y], [ddg3x, ddg3y], [ddg4x, ddg4y], [ddg5x, ddg5y]]

        return P, dP, G, dG, ddG


    ########################


    def Dynamics(self, state, u):
        q= state[0]
        dq= state[1]
        u= u[0]
        m=[self.m1, self.m2, self.m3, self.m4, self.m5]
        P0=[0, 0]

        P, dP, G, dG, ddG = self.Kinematics(q, dq)

        eq = [0, 0, 0, 0, 0]

        eq[4] = (G[4][0]-P[3][0])*(-self.m5*self.g) - ((G[4][0]-P[3][0])*self.m5*ddG[4][1] - (G[4][1]-P[3][1])*self.m5*ddG[4][0])

        for i in [3, 4]:
            eq[3] =(G[i][0]-P[1][0])*(-m[i]*self.g) - ((G[i][0]-P[1][0])*m[i]*ddG[i][1] - (G[i][1]-P[1][1])*m[i]*ddG[i][0])

        for i in [2, 3, 4]:
            eq[2] =(G[i][0]-P[1][0])*(-m[i]*self.g) - ((G[i][0]-P[1][0])*m[i]*ddG[i][1] - (G[i][1]-P[1][1])*m[i]*ddG[i][0])

        for i in [1, 2, 3, 4]:
            eq[1] =(G[i][0]-P[0][0])*(-m[i]*self.g) - ((G[i][0]-P[0][0])*m[i]*ddG[i][1] - (G[i][1]-P[0][1])*m[i]*ddG[i][0])

        for i in [0, 1, 2, 3, 4]:
            eq[0] =(G[i][0]-P0[0])*(-m[i]*self.g) - ((G[i][0]-P0[0])*m[i]*ddG[i][1] - (G[i][1]-P0[1])*m[i]*ddG[i][0])


        ddq5 = (u[3] + eq[4])/self.i5
        ddq4 = (u[2] + eq[3] - ddq5*self.i5)/self.i4
        ddq3 = (u[1] + eq[2] - ddq5*self.i5 - ddq4*self.i4)/self.i3
        ddq2 = (u[0] + eq[1] - ddq5*self.i5 - ddq4*self.i4 - ddq3*self.i3)/self.i2
        ddq1 = (eq[0] - ddq5*self.i5 - ddq4*self.i4 - ddq3*self.i3 - ddq2*self.i2)/self.i1



        ddq = [ddq1, ddq2, ddq3, ddq4, ddq5]
        return P, dP, G, dG, ddG, ddq




    #############################


    def heel_strike(self, q, dq, p, dP, G, dG):
        qn= q[::-1]
        P0=[0, 0]
        P= [P0, p[0], p[1], p[2], p[3], p[4]]
        Pn=[P[5], P[4], P[2], P[3], P[1], P[0]]
        Gn= G[::-1]
        dGn= dG[::-1]
        m=[self.m1, self.m2, self.m3, self.m4, self.m5]
        I=[self.i1, self.i2, self.i3, self.i4, self.i5]
        # if masses are not symmetric the we will need another m_next (mn) & I_next (In)

        eq= [0, 0, 0, 0, 0]

        i=0
        j=4-i
        eq[4] = ((G[i][0] - P[1][0])*m[i]*dG[i][1]) - ((G[i][1] - P[1][1])*m[i]*dG[i][0]) + dq[i]*I[i] - (
                (Gn[j][0] - Pn[4][0])*m[j]*dGn[j][1] - (Gn[j][1] - Pn[4][1])*m[j]*dGn[j][0])

        for i in [0, 1]:
            j=4-1
            eq[3] = ((G[i][0] - P[2][0]) * m[i] * dG[i][1]) - ((G[i][1] - P[2][1]) * m[i] * dG[i][0]) + dq[i] * I[i] - (
                        (Gn[j][0] - Pn[2][0]) * m[j] * dGn[j][1] - (Gn[j][1] - Pn[2][1]) * m[j] * dGn[j][0])

        for i in [0, 1, 2]:
            j=4-1
            eq[2] = ((G[i][0] - P[2][0]) * m[i] * dG[i][1]) - ((G[i][1] - P[2][1]) * m[i] * dG[i][0]) + dq[i] * I[i] - (
                        (Gn[j][0] - Pn[2][0]) * m[j] * dGn[j][1] - (Gn[j][1] - Pn[2][1]) * m[j] * dGn[j][0])

        for i in [0, 1, 2, 3]:
            j=4-1
            eq[1] = ((G[i][0] - P[4][0]) * m[i] * dG[i][1]) - ((G[i][1] - P[4][1]) * m[i] * dG[i][0]) + dq[i] * I[i] - (
                        (Gn[j][0] - Pn[1][0]) * m[j] * dGn[j][1] - (Gn[j][1] - Pn[1][1]) * m[j] * dGn[j][0])

        for i in [0, 1, 2, 3, 4]:
            j=4-1
            eq[0] = ((G[i][0] - P[5][0]) * m[i] * dG[i][1]) - ((G[i][1] - P[5][1]) * m[i] * dG[i][0]) + dq[i] * I[i] - (
                        (Gn[j][0] - Pn[0][0]) * m[j] * dGn[j][1] - (Gn[j][1] - Pn[0][1]) * m[j] * dGn[j][0])


        dqn5= eq[4] / self.i5
        dqn4= (eq[3] - dqn5*self.i5) / self.i4
        dqn3 = (eq[2] - dqn5*self.i5 - dqn4*self.i4) / self.i3
        dqn2 = (eq[1] - dqn5*self.i5 - dqn4*self.i4 - dqn3*self.i3) / self.i2
        dqn1 = (eq[0] - dqn5*self.i5 - dqn4*self.i4 - dqn3*self.i3 - dqn2*self.i2) / self.i1

        dqn=[dqn1, dqn2, dqn3, dqn4, dqn5]
        return [qn, dqn]






########################
# class nlp for optimization
########################


class nlp(five_link):
    def __init__(self, five_link):
        self.cost = self.get_cost(five_link.u, five_link.N, five_link.h)
        five_link.opti.minimize(self.cost)
        self.ceq = self.get_constraints(five_link)
        five_link.opti.subject_to(self.ceq)
        self.bounds = self.get_bounds(five_link)
        five_link.opti.subject_to(self.bounds)
        p_opts = {"expand": True}
        s_opts = {"max_iter": 1000}
        five_link.opti.solver("ipopt", p_opts, s_opts)
        self.initial_guess(five_link)


    def get_cost(self, u, N, h):
        result = 0
        for i in range(N - 1):
            for j in range(4):
                result += (h / 2) * (u[i][0][j] ** 2 + u[i + 1][0][j] ** 2)
        return result


    def get_constraints(self, five_link):
        ceq = []
        for i in range(five_link.N - 1):
            q1 = (five_link.state[i][0])
            q2 = (five_link.state[i + 1][0])
            dq1 = (five_link.state[i][1])
            dq2 = (five_link.state[i + 1][1])
            ddq1 = five_link.ddq[i]
            ddq2 = five_link.ddq[i + 1]
            ceq.extend(self.get_collocation(q1, q2, dq1, dq2, ddq1, ddq2, five_link.h))

        q0 = (five_link.state[0][0])
        dq0 = (five_link.state[0][1])
        qf = (five_link.state[-1][0])
        dqf = (five_link.state[-1][1])
        ceq.extend(self.get_boundary_constrainsts(q0, dq0, five_link.impactmap, five_link))
        #ceq.extend([(five_link.dp0[4][1] == 0), (five_link.dpN[4][1] == 0)])


        for i in range(five_link.N):
            ceq.extend([((five_link.pos[i][4][1]) >= 0)])
            ceq.extend([((five_link.pos[i][4][1]) <= 0.2)])

        ceq.extend([((five_link.pos[0][4][1]) == five_link.P0[1])])
        ceq.extend([five_link.pos[-1][4][1] == five_link.P0[1]])

        return ceq


    def get_collocation(self, q1, q2, dq1, dq2, ddq1, ddq2, h):
        c = []
        for i in range(4):
            c.extend([(((h / 2) * (ddq2[i] + ddq1[i])) - (dq2[i] - dq1[i]) == 0)])
        c.extend([(((h / 2) * (dq2 + dq1)) - (q2 - q1) == 0)])
        return c


    def get_boundary_constrainsts(self, state1, dstate1, impact, five_link):
        c = []
        # here impact
        for i in range(4): c.extend([(state1[i, 0] - impact[0][i] == 0), (dstate1[i, 0] - impact[1][i] == 0)
                                     , state1[i, 0] - five_link.initial[i]==0])

        return c



    def get_bounds(self, five_link):
        c = []
        f = 20
        for i in range(five_link.N):
            q = (five_link.state[i][0])
            dq = (five_link.state[i][1])
            u = (five_link.u[i][0])
            c.extend([five_link.opti.bounded(-np.pi, q[0], np.pi),
                    five_link.opti.bounded(-np.pi, q[1], np.pi),
                    five_link.opti.bounded(-np.pi, q[2], np.pi),
                    five_link.opti.bounded(-np.pi, q[3], np.pi),
                    five_link.opti.bounded(-np.pi, q[4], np.pi),
                    five_link.opti.bounded(-f * np.pi, dq[0], f * np.pi),
                    five_link.opti.bounded(-f * np.pi, dq[1], f * np.pi),
                    five_link.opti.bounded(-f * np.pi, dq[2], f * np.pi),
                    five_link.opti.bounded(-f * np.pi, dq[3], f * np.pi),
                    five_link.opti.bounded(-f * np.pi, dq[4], f * np.pi),
                    five_link.opti.bounded(-five_link.max_tau, u[0], five_link.max_tau),
                    five_link.opti.bounded(-five_link.max_tau, u[1], five_link.max_tau),
                    five_link.opti.bounded(-five_link.max_tau, u[2], five_link.max_tau),
                    five_link.opti.bounded(-five_link.max_tau, u[3], five_link.max_tau)])
        return c



    def initial_guess(self, five_link):
        for i in range (5):
            for j in range(five_link.N):
                five_link.opti.set_initial(five_link.state[j][0][i], five_link.qin[i][j])
                five_link.opti.set_initial(five_link.state[j][1][i], five_link.dqin[i][j])
                if i<4:
                    five_link.opti.set_initial(five_link.u[j][0][i], five_link.uin[i][j])




#########################
# main code starts from below
#########################

N=100
T=0.1
k=10 #steps
start=[-0.3, 0.7, 0, -0.5, -0.3]#[-0.3, 0.7, 0, -0.5, -0.3]#[-0.35, 0.65, 0, -0.55, -0.35]
stop=start[::-1]

# Initial guess
qin = np.zeros((5, N))
dqin = np.zeros((5, N))
uin = np.zeros((4, N))

qing=np.zeros((5, k*N))
dqing=np.zeros((5, k*N))
uing=np.zeros((4, k*N))

for i in range(5):
    for j in range(N):
        qin[i, j] = (start[i] + (j / (N - 1)) * (stop[i] - start[i]))
        dqin[i, j] = (stop[i] - start[i])/T

        qing[i, j] = (start[i] + (j / (N - 1)) * (stop[i] - start[i]))
        dqing[i, j] = (stop[i] - start[i]) / T



q = []
dq = []
u = []
pos = []

p=[0,0]
p01=[]
for t in range(k):
    pas=five_link(start, stop, qin, dqin, uin, N, T)
    solv=nlp(pas)

    sol1=pas.opti.solve()

    print(sol1.value(pas.u[50][0][0]))


    for j in range(5):
        tempq = []
        tempdq = []
        tempu = []
        temp = []
        for i in range(N):
            tempq.append(sol1.value(pas.state[i][0][j]))
            tempdq.append(sol1.value(pas.state[i][1][j]))
            temp.append([sol1.value(pas.pos[i][j][0]), sol1.value(pas.pos[i][j][1])])
            if j < 4:
                tempu.append(sol1.value(pas.u[i][0][j]))
            if j<1:
                p01.append(p)


        q.append(tempq)
        pos.append(temp)
        dq.append(tempdq)
        u.append(tempu)

    for j in range(N):
        for i in range(5):
            qin[i, j]= sol1.value(pas.state[j][0][i])
            dqin[i, j]= sol1.value(pas.state[j][1][i])
            if t<(k-1):
                qing[i, (t+1)*N+j] = sol1.value(pas.state[j][0][i])
                dqing[i, (t+1)*N+j] = sol1.value(pas.state[j][1][i])
            if i<4:
                uin[i, j]= sol1.value(pas.u[j][0][i])
                if t<k-1:
                    uing[i, (t+1)*N+j] = sol1.value(pas.u[j][0][i])

    # print(len(pos[4]))
    # print((t+1)*N)

    p = [(t+1)*pos[4][-1][0], 0]
    #print(p)



time = np.arange(0, k*T, pas.h)




fig = plt.figure()
camera = Camera(fig)

p11=[]
p21=[]
p31=[]
p41=[]
p51=[]

q11=[]
q21=[]
q31=[]
q41=[]
q51=[]

dq11=[]
dq21=[]
dq31=[]
dq41=[]
dq51=[]

u11=[]
u21=[]
u31=[]
u41=[]

for t in range(k):
    for i in range(N):
        p11.append([pos[5*t+0][i][0], pos[5*t+0][i][1]])
        p21.append([pos[5*t+1][i][0], pos[5*t+1][i][1]])
        p31.append([pos[5*t+2][i][0], pos[5*t+2][i][1]])
        p41.append([pos[5*t+3][i][0], pos[5*t+3][i][1]])
        p51.append([pos[5*t+4][i][0], pos[5*t+4][i][1]])

        q11.append(q[5 * t + 0][i])
        q21.append(q[5 * t + 1][i])
        q31.append(q[5 * t + 2][i])
        q41.append(q[5 * t + 3][i])
        q51.append(q[5 * t + 4][i])

        dq11.append(dq[5 * t + 0][i])
        dq21.append(dq[5 * t + 1][i])
        dq31.append(dq[5 * t + 2][i])
        dq41.append(dq[5 * t + 3][i])
        dq51.append(dq[5 * t + 4][i])

        u11.append(u[5 * t + 0][i])
        u21.append(u[5 * t + 1][i])
        u31.append(u[5 * t + 2][i])
        u41.append(u[5 * t + 3][i])



for i in range(k*N):
    p0=[p01[i][0], p01[i][1]]
    p1 = [p11[i][0], p11[i][1]]
    p2 = [p21[i][0], p21[i][1]]
    p3 = [p31[i][0], p31[i][1]]
    p4 = [p41[i][0], p41[i][1]]
    p5 = [p51[i][0], p51[i][1]]

    plt.axes(xlim=(-1, 10), ylim=(-2, 2))

    plt.plot([p0[0], p1[0]+p0[0]], [p0[1], p1[1]+p0[1]], 'r', [p1[0]+p0[0], p2[0]+p0[0]], [p1[1]+p0[1], p2[1]+p0[1]], 'b',
             [p2[0]+p0[0], p3[0]+p0[0]], [p2[1]+p0[1], p3[1]+p0[1]], 'c', [p2[0]+p0[0], p4[0]+p0[0]], [p2[1]+p0[1], p4[1]+p0[1]], 'b',
             [p4[0]+p0[0], p5[0]+p0[0]], [p4[1]+p0[1], p5[1]+p0[1]], 'r')

    plt.plot([-1, 10], [0, 0], 'g')

    camera.snap()
animation = camera.animate(interval=60)

plt.show()
plt.close()

qing1=np.zeros((5, k*N))
dqing1=np.zeros((5, k*N))
uing1=np.zeros((4, k*N))

qing2=np.zeros((5, k*N))
dqing2=np.zeros((5, k*N))
uing2=np.zeros((4, k*N))

q=np.zeros((5, k*N))
dq=np.zeros((5, k*N))
u=np.zeros((4, k*N))
for t in range(k):
    for i in range(N):
        for j in range(5):
            if (t%2==0):
                qing1[j][t*N+i]=qing[j][t*N+i]
                dqing1[j][t * N + i] = dqing[j][t * N + i]


            else:
                qing1[4-j][t * N + i] = qing[j][t*N+i]
                dqing1[4-j][t * N + i] = dqing[j][t * N + i]

            qing2[j][t * N + i] = qing[j][t * N + i]
            dqing2[j][t * N + i] = dqing[j][t * N + i]

            if j<4:
                if (t % 2 == 0):
                    uing1[j][t * N + i] = uing[j][t * N + i]
                else:
                    uing1[3-j][t * N + i] = uing[j][t * N + i]
                uing2[j][t * N + i] = uing[j][t * N + i]


for t in range(k):
    for i in range(N):
        if (t % 2 == 0):
            q[0][t*N+i]=q11[t*N+i]
            q[1][t * N + i] = q21[t * N + i]
            q[2][t * N + i] = q31[t * N + i]
            q[3][t * N + i] = q41[t * N + i]
            q[4][t * N + i] = q51[t * N + i]

            dq[0][t * N + i] = dq11[t * N + i]
            dq[1][t * N + i] = dq21[t * N + i]
            dq[2][t * N + i] = dq31[t * N + i]
            dq[3][t * N + i] = dq41[t * N + i]
            dq[4][t * N + i] = dq51[t * N + i]

            u[0][t * N + i] = u11[t * N + i]
            u[1][t * N + i] = u21[t * N + i]
            u[2][t * N + i] = u31[t * N + i]
            u[3][t * N + i] = u41[t * N + i]

        else:
            q[0][t * N + i] = q51[t * N + i]
            q[1][t * N + i] = q41[t * N + i]
            q[2][t * N + i] = q31[t * N + i]
            q[3][t * N + i] = q21[t * N + i]
            q[4][t * N + i] = q11[t * N + i]

            dq[0][t * N + i] = dq41[t * N + i]
            dq[1][t * N + i] = dq31[t * N + i]
            dq[2][t * N + i] = dq31[t * N + i]
            dq[3][t * N + i] = dq21[t * N + i]
            dq[4][t * N + i] = dq11[t * N + i]

            u[0][t * N + i] = u41[t * N + i]
            u[1][t * N + i] = u31[t * N + i]
            u[2][t * N + i] = u21[t * N + i]
            u[3][t * N + i] = u11[t * N + i]



name = ['q', 'dq', 'u']

plt.subplot(322)
plt.title('Optimised Solution')
plt.plot(time, q[0], 'r', time, q[1], 'g', time, q[2], 'b',
         time, q[3], 'y', time, q[4], 'c')

plt.subplot(321)
plt.title('Initial Guess')


plt.plot(time, qing1[0][:], 'r', time, qing1[1][:], 'g', time, qing1[2][:], 'b',
         time, qing1[3][:], 'y', time, qing1[4][:], 'c')
plt.ylabel(name[0])

plt.subplot(324)
plt.plot(time, dq[0], 'r', time, dq[1], 'g', time, dq[2], 'b',
         time, dq[3], 'y', time, dq[4], 'c')

plt.subplot(323)

plt.plot(time, dqing1[0][:], 'r', time, dqing1[1][:], 'g', time, dqing1[2][:], 'b',
         time, dqing1[3][:], 'y', time, dqing1[4][:], 'c')
plt.ylabel(name[1])

plt.subplot(326)
plt.plot(time, u[0], 'g', time, u[1], 'b', time, u[2], 'y',
         time, u[3], 'c')

plt.subplot(325)

plt.plot(time, uing1[0][:], 'g', time, uing1[1][:], 'b', time, uing1[2][:], 'y',
         time, uing1[3][:], 'c')
plt.ylabel(name[2])

plt.suptitle('Five-Link')
plt.show()
# print(u)
# print(max(u))
# print(max(max(u.all())))
# print(max(max(max(u.all()))))
# print(max(max(max(max(u.all())))))



#####################

##############
# regular graph

name = ['q', 'dq', 'u']

plt.subplot(322)
plt.title('Optimised Solution')
plt.plot(time, q11, 'r', time, q21, 'g', time, q31, 'b',
         time, q41, 'y', time, q51, 'c')

plt.subplot(321)
plt.title('Initial Guess')


plt.plot(time, qing2[0][:], 'r', time, qing2[1][:], 'g', time, qing2[2][:], 'b',
         time, qing2[3][:], 'y', time, qing2[4][:], 'c')
plt.ylabel(name[0])

plt.subplot(324)
plt.plot(time, dq11, 'r', time, dq21, 'g', time, dq31, 'b',
         time, dq41, 'y', time, dq51, 'c')

plt.subplot(323)

plt.plot(time, dqing2[0][:], 'r', time, dqing2[1][:], 'g', time, dqing2[2][:], 'b',
         time, dqing2[3][:], 'y', time, dqing2[4][:], 'c')
plt.ylabel(name[1])

plt.subplot(326)
plt.plot(time, u11, 'g', time, u21, 'b', time, u31, 'y',
         time, u41, 'c')

plt.subplot(325)

plt.plot(time, uing2[0][:], 'g', time, uing2[1][:], 'b', time, uing2[2][:], 'y',
         time, uing2[3][:], 'c')
plt.ylabel(name[2])

plt.suptitle('Five-Link')
plt.show()









