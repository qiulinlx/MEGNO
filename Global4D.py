
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

def standard4d(K, B, x1,x2,x3,x4):
    x4n=x4+(K/(2*np.pi))*(np.sin(2*np.pi*x3))-(B/(2*np.pi))*np.sin((2*np.pi*(x1-x3))) % (1)
    x3n= x3+x4+(K/(2*np.pi))*(np.sin(2*np.pi*x3))-(B/(2*np.pi))*np.sin(2*np.pi*(x1-x3))  % (1)
    x2n=x2+(K/(2*np.pi))*(np.sin(2*np.pi*x1))-(B/(2*np.pi)) *np.sin(2*np.pi*(x3-x1)) % (1)
    x1n=x1+x2+(K/(2*np.pi))*(np.sin(2*np.pi*x1))-(B/(2*np.pi))*np.sin(2*np.pi*(x3-x1)) % (1)
    return x1n,x2n,x3n, x4n

t=2000
K=0.8
B= 0.1

dx1=0.01
dx2=0.01
dx3=0.01
dx4=0.01

ox=[]
ol=[]
oy=[]
logl=[]

for k in range(10000):

    x1= np.random.uniform(0, 1)
    x3=np.random.uniform(0, 1)  
    x2=0
    x4=0

    alpha=[]
    x11=[]
    x22=[]
    x33=[]
    x44=[]


    for i in range(t):
        x1n,x2n, x3n, x4n=standard4d(K, B, x1,x2,x3,x4)
        x1=x1n %1
        x2=x2n %1
        x3=x3n %1
        x4=x4n %1

        x11.append(x1)
        x22.append(x2)
        x33.append(x3)
        x44.append(x4)

        Mat4=np.array([[1+K*np.cos(2*np.pi*x1)+B*np.cos(2*np.pi*(x3-x1)), 1, -B*np.cos(2*np.pi*(x3-x1)), 0], 
                    [K*np.cos(2*np.pi*x1)+ B*np.cos(2*np.pi*(x1-x3)) , 1, -B*np.cos(2*np.pi*(x3-x1)), 0],
                    [-B*np.cos(2*np.pi*(x1-x3)), 0, 1+K*np.cos(2*np.pi*x3)+B*np.cos(2*np.pi*(x1-x3)), 1],
                    [-B*np.cos(2*np.pi*(x1-x3)), 0, K*np.cos(2*np.pi*x3)+B*np.cos(2*np.pi*(x1-x3)), 1]])

        dx1, dx2, dx3, dx4 =np.dot(Mat4, np.array([dx1,dx2, dx3, dx4]))

        #print(dp, dq)

        a=np.sqrt(dx1**2+dx2**2+dx3**2+dx4**2)
        alpha.append(a)
        dx1=dx1/a
        dx2=dx2/a #Renormalization 
        dx3=dx3/a
        dx4=dx4/a

    alpha=alpha[1:]
    lalpha=np.log(alpha)
    #print(x33)
    l=sum (lalpha)/(t+1)
    logl.append(np.log10(l))
    length=[]

    for k in range(len(x11)):
        if (np.abs(x22[k])<0.01) and (np.abs(x44[k])<0.01):
            ox.append(x11[k])
            oy.append(x33[k])
            length.append(x11[k])
            #print(x33[k], x44[k])'''
    #ox.append(x22)
    #oy.append(x44)    

    for i in range(len(length)):
        ol.append(l)  


#print(length) 

plt.scatter(ox,oy,c=ol, cmap='summer', marker='.', s=2)
plt.colorbar()    
plt.xlabel('y')
plt.ylabel('py')
plt.title('Global 4D Map PSS where x2=0 and x4=0 ')
#plt.show()
plt.savefig('Global4D.png', dpi=500)
plt.clf()
plt.hist(logl, bins=20, color='green')
plt.title("Log(mLCE) of Global 4D Standard Map")
plt.xlabel("Log(mLCE)")
plt.ylabel("Frequency")
plt.savefig('Global4Dhist.png', dpi=500)