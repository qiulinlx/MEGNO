import numpy as np
import matplotlib.pyplot as plt
from Maps4 import *
from Variationalequations import *
import scipy.integrate as integrate
from scipy.integrate import solve_ivp


#MEGNO

E=1/24
x=0
y= 0.01 #np.random.uniform(-0.05, 0.05)
py= 0.012 #np.random.uniform(-0.05, 0.05)
px= np.sqrt(2*E-y**2+2/3*y**3-py**2)

tau=0.02

v1=0.01
v2=0.01
v3=0.01
v4=0.01
meg=[]
for i in range (30000):

    #Step 2 Evolve Deviation and Orbit
    t= [tau*(i-1), i*tau]
    q=np.array([x,y,px,py])
    state=  np.array([v1,v2,v3,v4, x,y])

    Dev= solve_ivp( Henonvar ,t, state, method="DOP853" ) 
    Traj =solve_ivp(henonmotion, t, q, method="DOP853") 
    
    dx=Dev.y[0]
    dy=Dev.y[1]
    dpx=Dev.y[2]
    dpy=Dev.y[3]

    v1=dx[-1]
    v2=dy[-1]
    v3=dpx[-1]
    v4=dpy[-1]

    x=Traj.y[0]
    y=Traj.y[1]
    px=Traj.y[2]
    py=Traj.y[3]

    x=x[-1]
    y=y[-1]
    px=px[-1]
    py=py[-1]

    #Step 3
    Dmat= np.array([[0,0, 1, 0], [0,0,0,1], [-1-2*y, -2*x, 0, 0], [-2*x, -1+2*y, 0, 0]])

    Dv= np.dot(Dmat, [v1, v2,v3,v4])

    #Step 4 Euclidean Norm 

    ew= np.sqrt(v1**2+v2**2+v3**2+v4**2)

    #Step 5
    v1=v1/ew
    v2=v2/ew
    v3=v3/ew
    v4=v4/ew

    ww=[v1, v2, v3, v4]
    m= np.dot(Dv,ww )

    #Step 6
    me= m*i*tau/ ew
    meg.append(me)

mmm=[]
mmm.append(2*meg[1])
for k in range (2,len(meg)):
    Y=2*np.mean(meg[1:k])
    #print(Y)
    mmm.append(Y)

plt.plot(mmm)
meeg=[]
for i in range (2,len(mmm)):
    yd=np.mean(mmm[1:i])
    meeg.append(yd)

plt.plot(meeg)

plt.show() 
