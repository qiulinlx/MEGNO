import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

t=2000 #HPC Ready
e=0.8
K=e/2*np.pi

def chirikov(p, q, K):
    """Applies Chirikov's standard map once with parameter e"""
    
    p_prime = (p + (K)*np.sin(q)) % (2*np.pi)
    q_prime = (p + q + (K)*np.sin(q)) % (2*np.pi)
    
    return p_prime, q_prime
ox=[]
oy=[]
ol=[]
logl=[]

for k in range(5000):
        x1=np.random.uniform(0, 2*np.pi)
        x2=np.random.uniform(0, 2*np.pi)
        alpha=[]
        x11=[]
        x22=[]
        dp=0.01
        dq=0.01
        for i in range(t):
            x1n,x2n=chirikov(x1,x2,K)
            x1=x1n 
            x2=x2n 
            #print(x1)

            x11.append(x1)
            x22.append(x2)  

            Mat2=np.array([[1, K*np.cos(x2)],[1, 1+K*np.cos(x2)]]) 

            dp,dq=np.dot(Mat2, np.array([dp,dq]))

            a1=np.sqrt(dp**2+dq**2)
            alpha.append(a1)
            #print(a1)
            dp=dp/a1
            dq=dq/a1 #Renormalization """

        alpha=alpha[1:]
        lalpha=np.log(alpha)

        l=sum (lalpha)/(i+1)
        logl.append(np.log10(l))
        ll=[]
        for i in range(len(x11)):
            ll.append(l)
        ox.append(x11)
        oy.append(x22)
        ol.append(ll)  

plt.scatter(ox,oy,c=ol, cmap='autumn', marker='.', s=2) 
plt.colorbar()
plt.xlabel('y')
plt.ylabel('py')
plt.title('Global 2D Map PSS where x=0 ')
plt.savefig('Global2D.png', dpi=500)

plt.clf()
plt.hist(logl, bins=20, color='orange')
plt.title("Log(mLCE) of Global 2D Standard Map")
plt.xlabel("Log(mLCE)")
plt.ylabel("Frequency")
plt.savefig('Global2Dhist.png', dpi=500)


