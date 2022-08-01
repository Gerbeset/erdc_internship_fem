import numpy as np
import sympy as syp
import matplotlib.pyplot as plt
import math as mth

# Problem: dt u + a dx u = 0; u(x,0) = sin(2pi x) on [0,1] x [0,1] with periodic boundary conditions

x, t = syp.symbols('x, t')

u_0 = syp.sin(2 * syp.pi * x)  # Exact Initial condition

a = 1  # Wave speed

u_exact = syp.sin(2 * syp.pi * (x - a * t))  # Exact solution

b = 0  # Space Domain [b,c]; assume [0,1]
c = 1

N = 80  # Number of Dofs
h = (c - b) / (N - 1)  # Mesh size (Uniform mesh)
T = 0.5  # Final Time

#Generates time stepping
CFL = 1
tau = CFL * (h / (2 * a))  # CFL Computation
I = mth.floor(T / tau)
print('The number of times steps is: I=', I)
tau = T / I
print('tau=', tau)
nodes = np.zeros(N)

#Generates nodes
nodes[0] = b
for i in range(1, N):
    nodes[i] = nodes[i - 1] + h

U_0 = np.zeros(N)  # Create initial vector
for i in range(0, N):
    U_0[i] = u_0.subs(x, nodes[i])  # Create initial condition vector

#Coefficients
c_10=-1/2
m_i = h
c_iim1 = -1/2
c_iip1 = 1/2

#Creates vectors for method
U_current = np.zeros(N)
U_up1 = np.zeros(N)
U_up2 = np.zeros(N)
U_up3 = np.zeros(N)
F_1 = np.zeros(N)
F_2 = np.zeros(N)
F_3 = np.zeros(N)

##################################   Flux    ################################
def Flux(u_cur):
           nn = len(u_cur)
           u_update = np.zeros(nn)
           u_update[0] =  - a * (1 / m_i) * (u_cur[nn - 2] * c_10 + u_cur[1] * c_iip1) + (1/m_i)*(a/2)*(u_cur[N-2] - u_cur[0]+ u_cur[1] - u_cur[0])
           for i in range(1, nn - 1):
               u_update[i] =  - a * (1 / m_i) * (u_cur[i - 1] * c_iim1 + u_cur[i + 1] * c_iip1)+ (1/m_i)*(a/2)*(u_cur[i-1] - u_cur[i] + u_cur[i+1] - u_cur[i])
           u_update[nn - 1] = u_update[0]
           return u_update
#End of subroutine-Euler_Update

##################################   METHOD    ################################
U_current = U_0
time = 0
for j in range(0,I):
    U_up1=U_current
    F_1 = Flux(U_up1)
    U_up2 = U_up1+(tau/3)*F_1
    F_2 = Flux(U_up2)
    U_up3 = U_up1 + (2*tau/3)*F_2
    F_3 = Flux(U_up3)
    U_current = U_up1 + (tau/4)*F_1 + (3*tau/4)*F_3
    #U_current = U_up1 + tau*F_1
    time = time + tau

print("The final time is t=", time)
U_Final = U_current

#Creates figures
plt.figure()
plt.subplot(1,2,1)
plt.plot(nodes, U_0, color ='blue')
plt.title('Initial Condition')

plt.subplot(1,2,2)
plt.plot(nodes,U_Final, color = "black")
plt.title('Final State')

###########################  ERROR ANALYSIS  ####################################

#Error analysis at t = T:
u_exact_T = u_exact.subs(t,T)
U_Exact_T = np.zeros(N)

for i in range(0, N):
    U_Exact_T[i] = u_exact_T.subs(x, nodes[i])

#L1 Error
L1_error = 0
L1_norm = 0

for i in range(0, N):
    L1_error = L1_error + m_i * abs(U_Exact_T[i] - U_Final[i])
    L1_norm = L1_norm + m_i * abs(U_Exact_T[i])
print('The L1 error is:', L1_error/L1_norm)

#L2 Error
L2_error = 0
L2_norm =0
for i in range(0, N):
    L2_error = L2_error + m_i * (U_Exact_T[i] - U_Final[i]) ** 2
    L2_norm = L2_norm + m_i * (U_Exact_T[i]) ** 2

print('The L2 error is:', syp.sqrt(L2_error/L2_norm))

#Linf Error
Linf_error =0
Linf_norm=0
Linf_list = np.zeros(N)
Linf_norm_list = np.zeros(N)

for i in range(0,N):
    Linf_list[i] = abs(U_Exact_T[i] - U_Final[i])
    Linf_norm_list[i] = abs(U_Exact_T[i])

Linf_norm = max(Linf_norm_list)
Linf_error = max(Linf_list)

print('The Linf error is:', Linf_error/Linf_norm)

#plt.show()