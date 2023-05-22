import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from matplotlib import cm   

E=1/8

n=5000
tau=0.1

def henon_heiles(t, z):
    q1, q2, p1, p2 = z
    dq1 = p1
    dq2 = p2
    dp1 = -q1 - 2*q1*q2
    dp2 = -q2 - q1**2 + q2**2
    return [dq1, dq2, dp1, dp2]


tau=0.1
c1=0.5*(1-1/np.sqrt(3))
c2= np.sqrt(3)/3
d1=0.5
cc=(2-np.sqrt(3))/24

lylis=[]
ylis=[]
pylis=[]
loglis=[]
lbar=[]

def eA(c, x, y, px, py, dx, dy, dpx, dpy):
    x=x+tau*c*px
    y=y+tau*c*py
    px=px
    py=py
    dx=dx+dpx*tau*c
    dy=dy+dpy*tau*c
    dpx=dpx
    dpy=dpy
    return (x, y, px, py, dx, dy, dpx, dpy)

def eB(d, x, y, px, py, dx, dy, dpx, dpy):
    x=x
    y=y
    px=px-tau*d*x*(1+2*y)
    py=py+tau*d*(y**2-x**2-y)
    dx=dx 
    dy=dy
    dpx=dpx-tau*d*((1+2*y)*dx+2*x*dy)
    dpy= dpy+d*tau*(-2*x*dx+(-1+2*y)*dy)
    return (x,y,px,py, dx, dy, dpx, dpy)

def eC(cc, x,y, px, py, dx, dy, dpx, dpy):
    x=x
    y=y
    px=px-2*x*(1+2*x**2+6*y+2*y**2)*cc*tau
    py=py-2*(y-3*y**2+2*y**3+3*x**2+2*x**2*y)*cc*tau
    dx=dx
    dy=dy
    dpx=dpx-2*((1+6*x**2+2*y**2+6*y)*dx+2*x*(3+2*y)*dy)*cc*tau
    dpy=dpy-2*(2*x*(3+2*y)*dx+(1+2*x**2+6*y**2-6*y)*dy)*cc*tau 
    return (x,y,px,py, dx, dy, dpx, dpy)

for i in range(5000):

    x=0
    y= np.random.uniform(-0.35, 0.55)
    py= np.random.uniform(-0.2, 0.2)
    px= np.sqrt(2*E-y**2+2/3*y**3-py**2)


    v1=0.01
    v2=0.01
    v3=0.01
    v4=0.01

    dx=v1
    dy=v2
    dpx=v3
    dpy=v4
    alpha=[]

    tt=[]
    xl=[]
    xl.append(x)
    yl=[]
    yl.append(y)
    pyl=[]
    pyl.append(py)
    pxl=[]
    pxl.append(px)
    dxl=[]
    dxl.append(v1)
    dyl=[]
    dyl.append(v2)
    dpxl=[]
    dpxl.append(v3)
    dpyl=[]
    dpyl.append(v4) 


    for j in range(n):

        z=eC(cc, x, y, px, py, dx, dy, dpx, dpy)
        x=z[0]
        y=z[1]
        px=z[2]
        py=z[3]
        dx=z[4]
        dy=z[5]
        dpx=z[6]
        dpy=z[7]



        z=eA(c1,x,y, px,py, dx, dy, dpx, dpy)
        x=z[0]
        y=z[1]
        px=z[2]
        py=z[3]
        dx=z[4]
        dy=z[5]
        dpx=z[6]
        dpy=z[7]
        

        z=eB(d1, x,y,px,py, dx, dy, dpx, dpy)
        x=z[0]
        y=z[1]
        px=z[2]
        py=z[3]
        dx=z[4]
        dy=z[5]
        dpx=z[6]
        dpy=z[7]
        

        z=eA(c2, x,y,px,py, dx, dy, dpx, dpy)
        x=z[0]
        y=z[1]
        px=z[2]
        py=z[3]
        dx=z[4]
        dy=z[5]
        dpx=z[6]
        dpy=z[7]
        

        z= eB(d1, x,y,px,py, dx, dy, dpx, dpy)
        x=z[0]
        y=z[1]
        px=z[2]
        py=z[3]
        dx=z[4]
        dy=z[5]
        dpx=z[6]
        dpy=z[7]
        

        z=eA(c1,x,y,px,py, dx, dy, dpx, dpy)
        x=z[0]
        y=z[1]
        px=z[2]
        py=z[3]
        dx=z[4]
        dy=z[5]
        dpx=z[6]
        dpy=z[7]
        
        z=eC(cc, x, y, px, py, dx, dy, dpx, dpy)
        x=z[0]
        y=z[1]
        px=z[2]
        py=z[3]
        dx=z[4]
        dy=z[5]
        dpx=z[6]
        dpy=z[7]
        
        a=np.sqrt(dx**2+dy**2+dpx**2+dpy**2)
        alpha.append(a)

        dx=dx/a
        dy=dy/a
        dpx=dpx/a
        dpy=dpy/a

        xl.append(x)
        yl.append(y)
        pxl.append(px)
        pyl.append(py)
        dxl.append(dx)
        dyl.append(dy)
        dpxl.append(dpx)
        dpyl.append(dpy)
        tt.append(j*tau)

    alpha=alpha[1:]
    lalpha=np.log(alpha)

    l=sum(lalpha)/(tt[-1])
    loglis.append(np.log10(l))
    poinx=[]
    poiny=[]
    for k in range(1,len(pyl)-1):
            if (xl[k]>0) and (xl[k-1]<0):
                ti= tt[k-1]
                tend= tt[k]
                teval = np.linspace(ti, tend, 50)
                t = (ti, tend)
                z0= [xl[k-1], yl[k-1], pxl[k-1], pyl[k-1]]
                sol1 = solve_ivp(henon_heiles, t, z0, t_eval=teval, rtol= 10**(-12), method='DOP853' ) #Up to here is correct
                q11=sol1.y[0]
                q21=sol1.y[1]
                p11=sol1.y[2]
                p21=sol1.y[3]
                for i in range(len(p11)):
                    if (( 0 < p11[i] )) and ( q11[i-1] < 0) and (q11[i]>0):
                        y= q21[i]
                        dy= p21[i]
                        poinx.append(y)
                        poiny.append(dy)
                        ylis.append(y)
                        pylis.append(dy)
    for z in range(len(poinx)):
        lylis.append((l))  
             
plt.scatter(ylis, pylis, c=lylis, cmap='PuBuGn', marker='.', s=4)
plt.colorbar()
plt.xlabel('y')
plt.ylabel('py')
plt.title('Global Henon Heiles PSS where x=0 ')
plt.savefig('Globalhenon.png', dpi=500)
#print(lylis)
#wee=np.arange(len(loglis))
#plt.plot(wee,loglis)
plt.clf()
plt.hist(loglis, bins=20, color='lightblue')
plt.title("Log(mLCE) of Global Henon-Heiles PSS")
plt.xlabel("Log(mLCE)")
plt.ylabel("Frequency")
plt.savefig('Globalhenonhist.png', dpi=500)

