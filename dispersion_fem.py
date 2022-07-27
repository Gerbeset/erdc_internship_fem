import numpy as np
import sympy as syp
import matplotlib.pyplot as plt
import math as mth
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

x,t = syp.symbols("x,t")

u_0 = syp.sin(2*syp.pi*x)

N = 5 #Number of dofs

#Linear transport wave speed
a = 1
K = 1 #Dispersive constant

#Space domain [b,c]; assume [0,1]
b = 0
c = 1

h = (c-b)/(N-1) #Mesh size
T = .5         #Final time

#CFL: use linear transport CFL

CFL = 1
tau = CFL * (h / (2 * a)) #CFL computation

I = mth.floor(T / tau)
print('The number of times steps is: I=', I)
tau = T / I
print('tau=', tau)

nodes = np.zeros(N)

nodes[0] = b  # Generates nodes
for i in range(1, N):
    nodes[i] = nodes[i - 1] + h

m_i = h
c_iim1 = -1/2
c_iip1 = 1/2
a_iim1 = -1/h
a_iip1 = -1/h
a_ii = 2/h

#Matrix construction
A = np.zeros((2*N,2*N))

# First row:
A[0, 0] = m_i
A[0, 1] = - tau * K * a_ii
A[0, 3] = - tau * K * a_iip1
A[0, 2 * N - 3] = - tau * K * a_iim1

# Second row:
A[1, 1] = 1

# Second to last row
A[2 * N - 2, 0] = 1
A[2 * N - 2, 2 * N - 2] = -1

# Last row
A[2 * N - 1, 1] = 1
A[2 * N - 1, 2 * N - 1] = -1

# Rest
for i in range(2, 2 * N - 2):
    if i % 2 == 0:
        A[i, i] = m_i
        A[i, i - 1] = - tau * K * a_iim1
        A[i, i + 1] = - tau * K * a_ii
        A[i, i + 3] = - tau * K * a_iip1
    if i % 2 == 1:
        A[i, i - 1] = - a_ii
        A[i, i - 2] = c_iip1
        A[i, i - 3] = -a_iim1
        A[i, i + 1] = -a_iip1
        A[i, i + 2] = c_iim1

A_sparse = csc_matrix(A, dtype=float)

A_inv = np.linalg.inv(A)

#################   Implicit update subroutine ################
def im_up(u_cur):
    nn = len(u_cur)
    u_update = np.zeros(nn)
    b = np.zeros(2*nn)
    x_result = np.zeros(2*nn)
    for i in range(0, nn-1):
        b[2*i] = u_cur[i]*m_i
    x_result = np.dot(A_inv,b)
    for i in range(0,nn):
        u_update[i] = x_result[2*i]
    return u_update
#End of implicit update subroutine

U_0 = np.zeros(N)  # Creates initial vector
for i in range(0, N):
    U_0[i] = u_0.subs(x, nodes[i])  # Create initial condition vector

#Creates vectors for method
U_current = np.zeros(N)
U_current = U_0
U_up = np.zeros(N)
U_Final = np.zeros(N)
time = 0

######### METHOD ###########
for i in range(0, I):
    U_up = U_current
    U_current = im_up(U_up)
    time = time + tau
print(time)
U_Final = U_current

#Generates figures
plt.figure()
plt.subplot(1,2,1)
plt.plot(nodes, U_0, color ='blue')
plt.title('Initial Condition')

plt.subplot(1,2,2)
plt.plot(nodes,U_Final, color = "black")
plt.title('Final State')

plt.show()