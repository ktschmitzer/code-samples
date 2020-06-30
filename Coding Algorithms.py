#!/usr/bin/env python
# coding: utf-8

# # Coding Algorithms 

# In[ ]:


# Forward euler algorithm for differential equations

def fwdeuler(a,b,n,alpha,f): 
    h = (b-a)/n # creating step size
    t = a # starting input with initial condition
    u = alpha # starting input with initial condition
    
    for i in range (1,n+1):
        u = u + (h*f(t,u)) # updating condition u_i = u_i+1 using the forward euler formula
        t = a + (i*h) # updating time step
    return (t,u)


# In[ ]:


# Trapezoid method for differential equations

def trapezoid(a,b,n,alpha,f, tol = .0001, M = 30): 
    h = (b-a)/n # creating step size
    t = a # starting input with initial condition
    u = alpha # starting input with initial condition
    
    for i in range (1,n+1):
        k_1 = u + (h/2)*f(t,u) # trapezoid formula
        u_0 = k_1 
        j = 1
        flag = 0 
        while (flag == 0):
            u = u_0 - (u_0 - (h/2)*f(t+h,u_0)-k_1)/(1 - (h/2)*f(t+h,u_0)) # updating condition u_i = u_i+1 using Newton's method
            if (abs(u - u_0 < tol):
                flag = 1
            else:
                j += 1
                u_0 = u
                if j > M:
                    print("maximum iterations exceeded")
        t = a + (i*h) # updating time step
    return(t,u)


# In[ ]:


# Leapfrog method for differential equations

def leapfrog(a,b,n,alpha,f): 
    h = (b-a)/n # creating step size
    t = a # starting input with initial condition
    u = alpha # starting input with initial condition
    u1 = u + (h*f(t,u)) # use forward euler method to get U_1

    for i in range(1, n+1):
        unext = u + 2*h*f(t+h, u1) # updating condition u_i = u_i+1
        t = a + (i*h) # updating time step
        u = u1
        u1 = unext
    return (t,u)


# In[ ]:


# ADI Method for boundary value problems 

def ADI(d, t, sigma, timesteps, spacesteps, f): # spacesteps and timesteps are the number of iterations through time and space this method will implement, not the size of the time or space step
    k = t/timesteps
    h = d/spacesteps
    
    initialcond = np.zeros((spacesteps + 1, spacesteps + 1)) # initializing a matrix for the initial conditions of the grid
    
    x = np.linspace(0, d, spacesteps + 1) # x values in the [0,1] grid
    y = np.linspace(0, d, spacesteps + 1) # y values in the [0,1] grid

    for i in range (1, spacesteps):
        for j in range(1, spacesteps): # leaving out the 0 and spacesteps + 1 spots because the boundary condition is 0
            initialcond[i,j] = f(x[i], y[j]) # filling in the initial conditions of the grid
            
    solution = np.zeros((timesteps + 1, spacesteps + 1, spacesteps + 1)) # initializing the t,x,y solution matrix
    
    for i in range (0, spacesteps + 1):
        for j in range (0, spacesteps + 1):
            solution[0,i,j] = initialcond[i][j] # adding the initial grid conditions into the solution matrix for t = 0       
    
    A = np.zeros((spacesteps - 1, spacesteps - 1)) # creating the A matrix
    
    UstarMat = np.zeros((spacesteps + 1, spacesteps + 1)) # initializing the matrix that will hold the half time-step updates
    Unplus1Mat = np.zeros((spacesteps + 1,spacesteps + 1)) # initializing the matric that will hold the Un+1 updates
    
    alpha = sigma*(k/(2*h**2))
    
    for i in range (0, spacesteps - 1):
        A[i][i] = 1 + 2*alpha # filling in the diagonal
        
        if i == spacesteps - 2:
            break
            
        A[i][i+1] = -alpha  # filling in the above diagonal
        A[i+1][i] = -alpha # filling in the below diagonal
    #print(A)
    
    for i in range (1, timesteps + 1): # iterating though all the time values 
        
        for j in range (1, spacesteps):
            Ustar = [] # creating a vector that will hold the Ustar values before plugging them into the matrix
            
            for k in range (2, spacesteps + 1):
                Ustar.append(alpha*initialcond[k-2,j] + (1 - (2 * alpha))*initialcond[k-1,j] + alpha*initialcond[k,j])
            UstarMat[1:spacesteps,j] = np.linalg.solve(A,Ustar) # solving for Ustar
            
        for m in range (1, spacesteps):
            Unplus1 = [] # creating a vector that will hold the Un+1 values before plugging them into the matrix 
            
            for k in range (2, spacesteps + 1):
                Unplus1.append(alpha*UstarMat[m,k-2] + (1 - (2 * alpha))*UstarMat[m,k-1] + alpha*UstarMat[m,k])
            Unplus1Mat[m,1:spacesteps] = np.linalg.solve(A,Unplus1) # solving for Unplus1
            
            #print(Unplus1Mat[m,1:-1])    
            
            solution[i,m] = Unplus1Mat[m] # inputting the Unplus1 values into the solution matrix
            initialcond = solution[i,:,:] # updating the initial conditions
            
    return solution 


# In[ ]:


# Runge Kutta 4 Scheme

def rungekutta4(a,b,n,x,f,usolution):
    k = (b - a)/n
    t = a
    e = abs(usolution(t) - x)
    earray = []
    #print(0, t, x, e)
    
    for i in range(1, n+1):
        F1 = k*f(t,x)
        F2 = k*f(t + k/2,x + F1/2)
        F3 = k*f(t + k/2, x + F2/2)
        F4 = k*f(t + k, x + F3)
        x = x + (F1 + 2*F2 + 2*F3 + F4)/6
        t += k
        e = abs(usolution(t) - x)
        earray.append(e)
        #print(i ,t, x, e)
    
    return(k,max(earray))  


# In[ ]:


# Runge Kutta 2 Scheme

def rungekutta2(a,b,n,x,f,usolution):
    k = (b - a)/n
    t = a
    e = abs(usolution(t) - x)
    earray = []
    #print(0, t, x, e)
    
    for i in range(1, n+1):
        F1 = f(t,x)
        F2 = f(t + k,x + k*F1)
        x = x + k*(F1 + F2)/2
        t += k
        e = abs(usolution(t) - x)
        earray.append(e)
        #print(i ,t, x, e)
    
    return(k,max(earray))  


# In[ ]:


# Forward Time, Forward Space Wave Equation Implementation

def FTFS(a, b, T, timesteps, spacesteps, f):
    k = T/timesteps
    h = (b - a)/spacesteps
    
    x = np.arange(a, b + h, h)
    
    A = np.zeros((spacesteps + 1, spacesteps + 1))
    
    for i in range(0, spacesteps + 1):
        A[i][i] = 1 + k/h # filling in the diagonal
        
        if i == spacesteps:
            break
            
        A[i][i + 1] = -k/h  # filling in the above diagonal
    
    for i in range (0, spacesteps + 1):
        x[i] = f(x[i])
    
    solution = np.zeros((timesteps + 1, spacesteps + 1))
    
    for i in range (0 , timesteps + 1):
        if i == 0:
            solution[i,:] = x
        else:
            Unplus1 = A.dot(x)
            solution[i,0:199] = Unplus1[0:199] # dimensions change based on number of timesteps and spacesteps
            solution[i,200] = Unplus1[199]  
            x = solution[i,:]
    return solution


# In[ ]:


# Upwind Scheme Wave Equation Implementation

def upwind(a, b, T, timesteps, spacesteps, f):
    k = T/timesteps
    h = (b - a)/spacesteps
    
    x = np.arange(a + h, b + h, h)
    
    A = np.zeros((spacesteps, spacesteps))
    
    for i in range(0, spacesteps):
        A[i][i] = 1 - k/h # filling in the diagonal
        
        if i == spacesteps - 1:
            break
            
        A[i][i - 1] = k/h  # filling in the lower diagonal
    
    for i in range (0, spacesteps):
        x[i] = f(x[i])
    
    solution = np.zeros((timesteps + 1, spacesteps ))
    
    for i in range (0 , timesteps + 1):
        if i == 0:
            solution[i,:] = x
        else:
            
            Unplus1 = A.dot(x)
            x = Unplus1
            solution[i,:] = Unplus1
    #print(solution.shape)
    return (np.hstack((np.zeros([41,1]), solution))) # dimensions change based on number of timesteps and spacesteps

