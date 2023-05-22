import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from matplotlib import cm
from sklearn.metrics import auc
from scipy.integrate import odeint, solve_ivp

#MEGNO
time=5000
E=1/8

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

def eA(c, x, y, px, py, dx, dy, dpx, dpy, tau):
    x=x+tau*c*px
    y=y+tau*c*py
    px=px
    py=py
    dx=dx+dpx*tau*c
    dy=dy+dpy*tau*c
    dpx=dpx
    dpy=dpy
    return (x, y, px, py, dx, dy, dpx, dpy)

def eB(d, x, y, px, py, dx, dy, dpx, dpy, tau):
    x=x
    y=y
    px=px-tau*d*x*(1+2*y)
    py=py+tau*d*(y**2-x**2-y)
    dx=dx 
    dy=dy
    dpx=dpx-tau*d*((1+2*y)*dx+2*x*dy)
    dpy= dpy+d*tau*(-2*x*dx+(-1+2*y)*dy)
    return (x,y,px,py, dx, dy, dpx, dpy)

def eC(cc, x,y, px, py, dx, dy, dpx, dpy, tau):
    x=x
    y=y
    px=px-2*x*(1+2*x**2+6*y+2*y**2)*cc*tau
    py=py-2*(y-3*y**2+2*y**3+3*x**2+2*x**2*y)*cc*tau
    dx=dx
    dy=dy
    dpx=dpx-2*((1+6*x**2+2*y**2+6*y)*dx+2*x*(3+2*y)*dy)*cc*tau
    dpy=dpy-2*(2*x*(3+2*y)*dx+(1+2*x**2+6*y**2-6*y)*dy)*cc*tau 
    return (x,y,px,py, dx, dy, dpx, dpy)

def Henon(x,w):
    t, y, px, py = w
    dt = 1/px
    dydt=py/px
    dpxdt = (-x-2*x*y)/px
    dpydt = (-y-x**2+y**2)/px    
    return [dt, dydt, dpxdt, dpydt]


loglis=[]
ylis=[]
pylis=[]
mlis=[]
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

        xx=[]
        yy=[]
        dxx=[]
        dyy=[]

        tt=[]
        megno=[]

        for i in range (time):
            #Evolve the orbit----------------------
            z=eC(cc, x, y, px, py, dx, dy, dpx, dpy , tau)
            x=z[0]
            y=z[1]
            px=z[2]
            py=z[3]
            dx=z[4]
            dy=z[5]
            dpx=z[6]
            dpy=z[7]

            z=eA(c1,x,y, px,py, dx, dy, dpx, dpy, tau)
            x=z[0]
            y=z[1]
            px=z[2]
            py=z[3]
            dx=z[4]
            dy=z[5]
            dpx=z[6]
            dpy=z[7]
            

            z=eB(d1, x,y,px,py, dx, dy, dpx, dpy, tau)
            x=z[0]
            y=z[1]
            px=z[2]
            py=z[3]
            dx=z[4]
            dy=z[5]
            dpx=z[6]
            dpy=z[7]
            

            z=eA(c2, x,y,px,py, dx, dy, dpx, dpy, tau)
            x=z[0]
            y=z[1]
            px=z[2]
            py=z[3]
            dx=z[4]
            dy=z[5]
            dpx=z[6]
            dpy=z[7]
            

            z= eB(d1, x,y,px,py, dx, dy, dpx, dpy, tau)
            x=z[0]
            y=z[1]
            px=z[2]
            py=z[3]
            dx=z[4]
            dy=z[5]
            dpx=z[6]
            dpy=z[7]
            

            z=eA(c1,x,y,px,py, dx, dy, dpx, dpy, tau)
            x=z[0]
            y=z[1]
            px=z[2]
            py=z[3]
            dx=z[4]
            dy=z[5]
            dpx=z[6]
            dpy=z[7]
            
            z=eC(cc, x, y, px, py, dx, dy, dpx, dpy, tau)
            x=z[0]
            y=z[1]
            px=z[2]
            py=z[3]
            dx=z[4]
            dy=z[5]
            dpx=z[6]
            dpy=z[7]
            #plt.plot(py,y, 'b.', markersize=0.1)
            #Compute Delta
            delta= np.sqrt(dx**2+dy**2+dpx**2+dpy**2)

            xx.append(x)
            yy.append(y)
            dxx.append(px)
            dyy.append(py)

            #Compute deltad
            Dmat= np.array([[0,0, 1, 0], [0,0,0,1], [-1-2*y, -2*x, 0, 0], [-2*x, -1+2*y, 0, 0]])
            wdot= np.matmul(Dmat, [dx,dy, dpx,dpy])

            ww= [dx/delta, dy/delta, dpx/delta, dpy/delta]
            deltad= np.dot(wdot, ww)

            #Compute function to be averaged
            f= deltad*i*tau/delta
            megno.append(f)

            #Renormalise
            dx=dx/delta
            dy=dy/delta
            dpx=dpx/delta
            dpy=dpy/delta
            #Time
            tt.append(i*tau)
        p=[]
        for i in range(2,len(tt)):
            a= auc(tt[:i], megno[:i])
            sol= 2*a/tt[i]
            p.append(sol)
        mt=tt[2:]
        #print(dyy)
        avmeg=[]

        aa= auc(mt, p)
        ydash=(aa/mt[-1])
        loglis.append(np.log10(ydash))

        poinx=[]
        poiny=[]
        for k in range(1,len(dyy)-1):
            if (xx[k]>0) and (xx[k-1]<0):
                ti= tt[k-1]
                tend= tt[k]
                teval = np.linspace(ti, tend, 50)
                t = (ti, tend)
                z0= [xx[k-1], yy[k-1], dxx[k-1], dyy[k-1]]
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
            mlis.append((ydash))      
             
plt.scatter(ylis, pylis, c=mlis, cmap='PuBuGn', marker='.', s=4)
plt.colorbar()
plt.xlabel('y')
plt.ylabel('py')
plt.title('Global Henon Heiles PSS where x=0 ')
plt.savefig('GlobalhenonMEGNO.png', dpi=500)
#print(lylis)
#wee=np.arange(len(loglis))
#plt.plot(wee,loglis)
plt.clf()
plt.hist(loglis, bins=20, color='lightblue')
plt.title("Log(Ydash) of Global Henon-Heiles PSS")
plt.xlabel("Log(Ydash)")
plt.ylabel("Frequency")
plt.savefig('Globalhenonhist.png', dpi=500)

