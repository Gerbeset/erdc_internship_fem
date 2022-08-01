import numpy as np
import sympy as syp
import matplotlib.pyplot as plt
import math as mth
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

x,t = syp.symbols("x,t")

u_0 = syp.sin(2*syp.pi*x)

u_exact = syp.sin(2*syp.pi*(x + ((2*syp.pi)**2)*t)) #Exact solution is correct

N = 120 #Number of dofs

#Linear transport wave speed
a = (2*syp.pi)**3
K = 1 #Dispersive constant

#Space domain [b,c]; assume [0,1]
b = 0
c = 1

h = (c-b)/(N-1) #Mesh size
T = 1/((2*syp.pi)**2)        #Final time

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
c_iim1 = -1 / 2
c_iip1 = 1 / 2
a_iim1 = -1 / h
a_iip1 = -1 / h
a_ii = 2 / h

# Matrix construction
A = np.zeros((2 * N, 2 * N))

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

def para_up(u_cur):
    nn = len(u_cur)
    u_update = np.zeros(nn)
    b = np.zeros(2*nn)
    x_result = np.zeros(2*nn)
    for i in range(0, nn-1):
        b[2*i] = u_cur[i]*m_i
    x_result = spsolve(A_sparse, b)
    for i in range(0,nn):
        u_update[i] = x_result[2*i]
    return u_update

U_current = np.zeros(N)

U_0 = np.zeros(N)  # Create initial vector
for i in range(0, N):
    U_0[i] = u_0.subs(x, nodes[i])  # Create initial condition vector

U_up = np.zeros(N)

################## 1-step test #########################
U_current = U_0
#U_Final = np.zeros(N)

U_1 = np.zeros(N)
U_1 = para_up(U_current)

#Error Analysis
u_exact_1 = u_exact.subs(t, tau)

U_exact_1 = np.zeros(N)

for i in range(0, N):
    U_exact_1[i] = u_exact_1.subs(x, nodes[i])

# L1 Error for single step
Error_1 = 0
for i in range(0, N):
    Error_1 = Error_1 + abs(U_1[i] - U_exact_1[i])
L1_norm = 0
for i in range(0, N):
    L1_norm = L1_norm + abs(U_exact_1[i])

print('The 1-step error is:', Error_1 / L1_norm)

################ METHOD #########################
U_final = np.zeros(N)
time = 0
for n in range(0,I):
    U_up = para_up(U_current)
    U_current = U_up
    time = time + tau

print("The final time is:", time)
U_final = U_current

u_exact_t = u_exact.subs(t, time)

U_t = np.zeros(N)

for i in range(0,N):
    U_t[i] = u_exact_t.subs(x,nodes[i])

#print(U_final)

L1_norm = 0
L1_error =0
for i in range(0,N):
    L1_norm = L1_norm + abs(U_t[i])*m_i
    L1_error = L1_error + abs(U_t[i] - U_final[i])*m_i

print('The L1 error is:', L1_error/L1_norm)

#Creates figures
plt.figure()
plt.subplot(1,2,1)
plt.plot(nodes, U_0, color ='blue')
plt.title('Initial Condition')

plt.subplot(1,2,2)
plt.plot(nodes,U_final, color = "black")
plt.title('Final State')

syp.plot(u_exact_t, (x,0,1))

plt.show()





