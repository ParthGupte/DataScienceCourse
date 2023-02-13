#Parth Gupte
#20201008

from scipy.stats import linregress
import numpy as np
import math as mt


class H0:
    def __init__(self):
        self.y = 0
    
    def fit(self,data):
        c = 0
        n = 0
        for x,y in data:
            n += 1
            c += y
        self.y = c/n
    
    def predict(self,x):
        return self.y

class H1:
    def __init__(self):
        self.a = 0
        self.b = 0
    def fit(self,data):
        xL = []
        yL = []
        for x,y in data:
            xL.append(x)
            yL.append(y)
        result = linregress(xL,yL)
        self.b = result.intercept
        self.a = result.slope
    
    def predict(self,x):
        return self.a*x+self.b

#draw data

X = np.random.uniform(0,2*mt.pi,(500,))

#we store the randomly drawn data so that it can be used for both H0 and H1 and we get a standard result
DataSamples = []
for i in range(50):
    xsample = np.random.choice(X,2)
    ysample = np.sin(xsample)
    DataSamples.append([(xsample[0],ysample[0]),(xsample[1],ysample[1])])

#H0 

#Fitting Models on samples

H0Models = []

for sample in DataSamples:
    hsample = H0()
    hsample.fit(sample)
    H0Models.append(hsample)

#Computing the Avg Model

h0avg = H0()
n = 0
for h in H0Models:
    n += 1
    h0avg.y += h.y
h0avg.y /= n

#Variance
VarH0 = 0
i = 0
for sample in DataSamples:
    h = H0Models[i]
    for point in sample:
        x0 = point[0]
        VarH0 += (h.predict(x0)-h0avg.predict(x0))**2
    i += 1

VarH0 /= 50*2
print("The Variance for H0 = ",VarH0)

#Bias^2
BiasH0 = 0

for sample in DataSamples:
    for point in sample:
        y0 = point[1] #We need not take the avg y since the data has no noise
        x0 = point[0]
        BiasH0 += (y0-h0avg.predict(x0))**2

BiasH0 /= 100
print("The Bias for H0 = ", BiasH0)

#Since the data has no noise we can compute the generalised test error by simply adding these 2 quantities

MSEH0 = BiasH0 + VarH0
print("The generalised test error or the MSE test error for H0 = ",MSEH0)


#H1

#Fitting Models on samples

H1Models = []

for sample in DataSamples:
    hsample = H1()
    hsample.fit(sample)
    H1Models.append(hsample)

#Computing the Avg Model

h1avg = H1()
n = 0
for h in H1Models:
    n += 1
    h1avg.a += h.a
    h1avg.b += h.b
h1avg.a /= n
h1avg.b /= n

#Variance
VarH1 = 0
i = 0
for sample in DataSamples:
    h = H1Models[i]
    for point in sample:
        x0 = point[0]
        VarH1 += (h.predict(x0)-h1avg.predict(x0))**2
    i += 1

VarH1 /= 50*2
print("The Variance for H1 = ",VarH1)

#Bias^2
BiasH1 = 0

for sample in DataSamples:
    for point in sample:
        y0 = point[1] #We need not take the avg y since the data has no noise
        x0 = point[0]
        BiasH1 += (y0-h1avg.predict(x0))**2

BiasH1 /= 100
print("The Bias for H1 = ", BiasH1)

#Since the data has no noise we can compute the generalised test error by simply adding these 2 quantities

MSEH1 = BiasH1 + VarH1
print("The generalised test error or the MSE test error for H1 = ",MSEH1)

if MSEH1 < MSEH0:
    print("Model H1 is a better fit for the given data.")
else:
    print("Model H0 is a better fit for the data.")

if BiasH1 < BiasH0:
    print("H1 has lesser Bias.")
else:
    print("H0 has lesser Bias.")

if VarH1 < VarH0:
    print("H1 has lower Variance")
else:
    print("H0 has lower variance.")



    



    