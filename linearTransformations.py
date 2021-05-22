import numpy as np

x = np.array([[1,2,3,4,5],
	[6,7,8,9,10]])

a = np.array([[1],[2]])

miu = x.mean(axis = 1)

print(x)

print(miu)

miu = np.expand_dims(miu,axis = 1)


eY2 = np.dot(a.transpose(),miu)

y = np.dot(a.transpose(),x)

print("y :", y)
eY1 = y.mean()

print("mean of y = aTx :",eY1,
	"\nexpectation of y direclty acquired by aTµ (where µ = mean(x)):" ,eY2)

covX = np.cov(x)

varY1 = np.var(y,axis=1,ddof=1)

varY2 = np.dot(np.dot(a.transpose(),covX),a)

print("variance of y = aTx :",varY1,
	"\nvariance of y direclty acquired by a⊤Σa (where Σ = cov(x)):" ,varY2)

z =  np.array([[1,2,3,4,5],
	[6,7,8,9,10]])

z1 = np.array([[1,2,3,4,5]])

z2 = np.array([6,7,8,9,10])

miu1 = z1.mean()

miu2 = z2.mean()

sigma11 = np.var(z1, ddof = 1)

sigma12 = np.cov(z).item((0,1))

sigma21 = np.cov(z).item((1,0))

sigma22 = np.var(z2,ddof =1)

print(np.size(z1),np.size(sigma11),np.size(sigma21))

print(sigma21)

def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density
p_z2_z1 = normal_dist(z2,miu2+sigma21*(sigma11**-1)*(z1-miu1),sigma22-sigma21*(sigma11**-1)*sigma12)

print("p(z2|z1) = ",p_z2_z1)

print(sigma22-sigma21*(sigma11**-1)*sigma12)