from numba import jit
import numpy as np
import math
from scipy.optimize import fsolve
from matplotlib import pyplot as plt
import csv
import time
from scipy import integrate
from scipy import interpolate
import gc
import tracemalloc
import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore")

@jit
def myGradient(x):
	ret = np.zeros(len(x))
	ret[:len(x)-1] = x[1:] - x[:len(x)-1]
	ret[len(x)] = ret[len(x)-1]
	return ret

@jit
def dispersionTB(mu,k_x,k_y,a,t1,t2):
	return -mu - 2.*t1*(np.cos(k_x*a)+np.cos(k_y*a)) - 4.*t2*np.cos(k_x*a)*np.cos(k_y*a)

def dispersionCircular(mu,k_x,k_y):
	return (k_x**2. + k_y**2.) - mu
@jit
def deltaFunc_swave(k_x,k_y):
	return np.ones(k_x.shape)
@jit
def deltaFunc_dxy(k_x,k_y):
	return np.sin(k_x)*np.sin(k_y)
@jit
def deltaFunc_dx2y2(k_x,k_y):
	return (np.cos(k_x)-np.cos(k_y)) #d_{x^2-y^2}order parameter

@jit
def deltaFunc_SCNem(kx,ky,r):
	return (1. + r*(np.cos(kx) - np.cos(ky)))/np.sqrt(1. + (r**2.))

@jit
def PhiFunc_Cartesian(kx,ky):
	return np.cos(kx) - np.cos(ky)

def PhiFunc(theta,r):
	return np.sqrt(2.)*np.cos(2.*theta)

def eigenvalues2D(m):
	ret1 = -np.sqrt(m[0][0]**2. + m[0][1]**2.)
	ret2 = -ret1
	return ret1, ret2

def fermi_deriv(E,T):
	energy = np.array(E)
	return 1/(2.*np.cosh(E/(2.*T)))**2.

def PhiSelfConsistency(mu,disp,Phi,PhiFunc,T,Tn,t1,t2,num_k):
	kx = np.linspace(-np.pi,np.pi,num_k)
	ky = np.linspace(-np.pi,np.pi,num_k)
	dk = 2.*np.pi/(num_k)
	Kx,Ky = np.meshgrid(kx,ky)
	
	xi_k = disp(mu,Kx,Ky,1.,t1,t2)
	
	f_k = PhiFunc(Kx,Ky)
	
	return np.sum(f_k*(np.tanh((xi_k + Phi*f_k)/(2.*T)) - 1.))*dk*dk/(2.*np.pi)**2. - Phi*np.sum((f_k**2.)/(2*Tn*np.cosh(xi_k/(2.*Tn))**2.))*dk*dk/(2.*np.pi)**2.

def pureNemPhi(T,disp,PhiFunc,mu,Tn,t1,t2,num_k):
	
	Phi = np.zeros(T.shape)
	for i in range(0,len(T)):
		if i < 4:
			Phi[i] = np.abs(fsolve(lambda x: PhiSelfConsistency(mu,disp,x,PhiFunc,T[i],Tn,t1,t2,num_k),1.))
		else:
			Phi[i] = np.abs(fsolve(lambda x: PhiSelfConsistency(mu,disp,x,PhiFunc,T[i],Tn,t1,t2,num_k),Phi[i-1]))
		
	return Phi

def deltaMixedSelfConsistency(mu,disp,delta,deltaFunc,Phi,PhiFunc,T,Tc,t1,t2,num_k,r,m):
	kx = np.linspace(-np.pi,np.pi,num_k)
	ky = np.linspace(-np.pi,np.pi,num_k)
	dk = 2.*np.pi/(num_k)
	Kx,Ky = np.meshgrid(kx,ky)
	
	xi_k = disp(mu,Kx,Ky,1.,t1,t2)
	
	f_k = PhiFunc(Kx,Ky)
	
	Y_k = deltaFunc(Kx,Ky,r)
	
	Ek = np.sqrt((xi_k + Phi*f_k)**2. + (delta*Y_k)**2.)
	
	ret = 0.
	
	for n in range(0,int(m/T)):
		ret = ret + np.sum((Y_k**2)*((2*T/(Ek**2. + (2.*np.pi*(float(n)+.5)*T)**2.)) - (2*Tc/(xi_k**2. + (2.*np.pi*(float(n)+.5)*Tc)**2.))))*dk*dk/(2.*np.pi)**2.
	
	return ret

def PhiMixedSelfConsistency(mu,disp,delta,deltaFunc,Phi,PhiFunc,T,Tn,t1,t2,num_k,r):
	kx = np.linspace(-np.pi,np.pi,num_k)
	ky = np.linspace(-np.pi,np.pi,num_k)
	dk = 2.*np.pi/(num_k)
	Kx,Ky = np.meshgrid(kx,ky)
	
	xi_k = disp(mu,Kx,Ky,1.,t1,t2)
	
	f_k = PhiFunc(Kx,Ky)
	
	Y_k = deltaFunc(Kx,Ky,r)
	
	Ek = np.sqrt((xi_k + Phi*f_k)**2. + (delta*Y_k)**2.)
	
	return np.sum(f_k*((xi_k + Phi*f_k)*np.tanh(Ek/(2.*T))/Ek - 1.))*dk*dk/(2.*np.pi)**2. - Phi*np.sum((f_k**2.)/(2*Tn*np.cosh(xi_k/(2.*Tn))**2.))*dk*dk/(2.*np.pi)**2.


def coupledIntegrals(mu,disp,delta,deltaFunc,Phi,PhiFunc,T,Tc,Tn,t1,t2,num_k,r):
	deltaResidual = deltaMixedSelfConsistency(mu,disp,delta,deltaFunc,Phi,PhiFunc,T,Tc,t1,t2,num_k,r,100)
	
	PhiResidual = PhiMixedSelfConsistency(mu,disp,delta,deltaFunc,Phi,PhiFunc,T,Tn,t1,t2,num_k,r)
	return [deltaResidual,PhiResidual]

def deltaPhi(T,disp,deltaFunc,PhiFunc,mu,Tn,Tc,t1,t2,num_k,r):
	
	n = len(T[T < .01])
	T = T[T >= .01]
	
	delta = np.zeros(T.shape)
	Phi = np.zeros(T.shape)
	Phi0 = np.zeros(T.shape)
	
	zeroDeltaYet = False
	
	for i in range(0,len(T)):
		
		if zeroDeltaYet == False:
			if i < 3:
				params = fsolve(lambda x: coupledIntegrals(mu,disp,x[0],deltaFunc,x[1],PhiFunc,T[i],Tc,Tn,t1,t2,num_k,r),[.5,1.])
				Phi0[i] = np.abs(fsolve(lambda x: PhiSelfConsistency(mu,disp,x[0],PhiFunc,T[i],Tn,t1,t2,num_k),1.))
			else:
				params = fsolve(lambda x: coupledIntegrals(mu,disp,x[0],deltaFunc,x[1],PhiFunc,T[i],Tc,Tn,t1,t2,num_k,r),[1.5*delta[i-1],Phi0[i-1]])
				Phi0[i] = np.abs(fsolve(lambda x: PhiSelfConsistency(mu,disp,x[0],PhiFunc,T[i],Tn,t1,t2,num_k),Phi0[i-1]))

			delta[i] = np.abs(params[0])
			Phi[i] = np.abs(params[1])

			if np.abs(delta[i]) < 1e-2:
				zeroDeltaYet = True
				Phi[i] = Phi0[i]
				delta[i] = 0.
		else:
			Phi[i] = np.abs(fsolve(lambda x: PhiSelfConsistency(mu,disp,x[0],PhiFunc,T[i],Tn,t1,t2,num_k),Phi[i-1]))
			Phi0[i] = Phi[i]
	
	delta = np.concatenate((delta[0]*np.ones(n),delta))
	Phi = np.concatenate((Phi[0]*np.ones(n),Phi))
	
	return delta, Phi

def deltaSelfConsistency(mu,disp,delta,deltaFunc,T,Tc,t1,t2,num_k,m):

	kx = np.linspace(-np.pi,np.pi,num_k)
	ky = np.linspace(-np.pi,np.pi,num_k)
	dk = 2.*np.pi/(num_k)
	Kx,Ky = np.meshgrid(kx,ky)
	
	xi_k = disp(mu,Kx,Ky,1.,t1,t2)
	
	Y_k = deltaFunc(Kx,Ky)
	
	Ek = np.sqrt((xi_k)**2. + (delta*Y_k)**2.)
	
	ret = 0.
	
	for n in range(0,int(m/T)):
		ret = ret + np.sum((Y_k**2)*((2*T/(Ek**2. + (2.*np.pi*(float(n)+.5)*T)**2.)) - (2*Tc/(xi_k**2. + (2.*np.pi*(float(n)+.5)*Tc)**2.))))*dk*dk/(2.*np.pi)**2.
	
	return ret

def deltaT(T,disp,deltaFunc,mu,Tc,t1,t2,num_k):
	m = 150
	n = len(T[T < .01])
	T = T[T >= .01]
	
	delta = np.zeros(T.shape)
	
	for i in range(0,len(T)):
		
		if i < 3:
			delta[i] = fsolve(lambda x: deltaSelfConsistency(mu,disp,x,deltaFunc,T[i],Tc,t1,t2,num_k,m),[1.7*Tc])[0]
		else:
			delta[i] = fsolve(lambda x: deltaSelfConsistency(mu,disp,x,deltaFunc,T[i],Tc,t1,t2,num_k,m),[delta[i-1]])[0]
	
	if n != 0:
		delta = np.concatenate((delta[0]*np.ones(n),delta))
	
	return delta

@jit
def velocityTB(kx,ky,a,t1,t2):
	
	vel_x = a*(2.*t1*np.sin(kx*a)+4.*t2*np.sin(kx*a)*np.cos(ky*a))
	vel_y = a*(2.*t1*np.sin(ky*a)+4.*t2*np.cos(kx*a)*np.sin(ky*a))
	
	return vel_x, vel_y

@jit
def velocityTBNem(kx,ky,a,t1,t2,Phi):
	vel_x,vel_y = velocityTB(kx,ky,a,t1,t2)
	vel_x = vel_x - Phi*a*np.sin(kx*a)
	vel_y = vel_y + Phi*a*np.sin(ky*a)
	return vel_x,vel_y

def velocityCircular(kx,ky):
	vel_x = 2.*kx
	vel_y = 2.*ky
	
	return vel_x, vel_y

@jit	
def ghFunc_TB_Cartesian(x,mu,deltaFunc,Phi,PhiFunc,r,N_kx,t1,t2):
	omega = 1e-4
	
	nLists = 1
	zeros = []
	
	if Phi != 0.:
		a = 4.*r*t2
		b1 = 4.*t2 + 4.*r*t1 + 4.*x*t2*np.sqrt(1.+r**2.)
		b2 = 4.*t2 + 4.*r*t1 - 4.*x*t2*np.sqrt(1.+r**2.)
		c1 = 2.*t1 + Phi + r*mu + 2.*x*t1*np.sqrt(1.+r**2.) + x*Phi*np.sqrt(1.+r**2.)
		c2 = 2.*t1 + Phi + r*mu - 2.*x*t1*np.sqrt(1.+r**2.) - x*Phi*np.sqrt(1.+r**2.)
		
		if r > 0.:
			k1_arg = (-b1 + np.sqrt(b1**2. - 4.*a*c1))/(2.*a)
			k2_arg = (-b2 + np.sqrt(b2**2. - 4.*a*c2))/(2.*a)
			if np.abs(k1_arg) < 1.:
				nLists = nLists + 1
				zeros.append(np.arccos(k1_arg))
			if np.abs(k2_arg) < 1.:
				nLists = nLists + 1
				zeros.append(np.arccos(k2_arg))
		else:
			k1_arg = (-b1 - np.sqrt(b1**2. - 4.*a*c1))/(2.*a)
			k2_arg = (-b2 - np.sqrt(b2**2. - 4.*a*c2))/(2.*a)
			if np.abs(k1_arg) < 1.:
				nLists = nLists + 1
				zeros.append(np.arccos(k1_arg))
			if np.abs(k2_arg) < 1.:
				nLists = nLists + 1
				zeros.append(np.arccos(k2_arg))
		
		zeros = np.array(zeros)
		zeros = zeros[np.argsort(zeros)]
		
		if nLists == 1:
			kx = np.linspace(1e-8,np.pi-1e-8,N_kx)
			ky = np.arccos(-(mu + 2.*t1*np.cos(kx) - Phi*np.cos(kx))/(2.*t1 + 4.*t2*np.cos(kx) + Phi))
		
			dkx = myGradient(kx)
			dky = myGradient(ky)
			dl = np.sqrt(dkx**2. + dky**2.)
			
			xi = dispersionTB(mu,kx,ky,1.,t1,t2) + Phi*PhiFunc(kx,ky)
			
			kx[np.abs(xi) > 1e-4] = -100
			ky[np.abs(xi) > 1e-4] = -100
			
			dl = dl[kx != -100]
			kx = kx[kx != -100]
			ky = ky[ky != -100]
			
			vF_x,vF_y = velocityTBNem(kx,ky,1.,t1,t2,Phi)
			vF = np.sqrt(vF_x**2. + vF_y**2.)
			
			Yk = deltaFunc_SCNem(kx,ky,r)
			
			N0 = np.sum(4.*(dl/vF))
			g = np.sum(4.*(dl/vF)*(x/np.sqrt((x + omega*1j)**2. - Yk**2.)))/N0
			h = np.sum(4.*(dl/vF)*(Yk/np.sqrt((x + omega*1j)**2. - Yk**2.)))/N0
			
		elif nLists == 2:
			kx1 = np.linspace(1e-8,zeros[0]-1e-8,N_kx)
			kx2 = np.linspace(zeros[0]+1e-8,np.pi-1e-8,N_kx)
			ky1 = np.arccos(-(mu + 2.*t1*np.cos(kx1) - Phi*np.cos(kx1))/(2.*t1 + 4.*t2*np.cos(kx1) + Phi))
			ky2 = np.arccos(-(mu + 2.*t1*np.cos(kx2) - Phi*np.cos(kx2))/(2.*t1 + 4.*t2*np.cos(kx2) + Phi))
			xi1 = dispersionTB(mu,kx1,ky1,1.,t1,t2) + Phi*PhiFunc(kx1,ky1)
			xi2 = dispersionTB(mu,kx2,ky2,1.,t1,t2) + Phi*PhiFunc(kx2,ky2)
			
			dkx1 = myGradient(kx1)
			dky1 = myGradient(ky1)
			dl1 = np.sqrt(dkx1**2. + dky1**2.)
			dkx2 = myGradient(kx2)
			dky2 = myGradient(ky2)
			dl2 = np.sqrt(dkx2**2. + dky2**2.)
			
			kx1[np.abs(xi1) > 1e-4] = -100
			ky1[np.abs(xi1) > 1e-4] = -100
			
			kx2[np.abs(xi2) > 1e-4] = -100
			ky2[np.abs(xi2) > 1e-4] = -100
			
			dl1 = dl1[kx1 != -100]
			kx1 = kx1[kx1 != -100]
			ky1 = ky1[ky1 != -100]
			dl2 = dl2[kx2 != -100]
			kx2 = kx2[kx2 != -100]
			ky2 = ky2[ky2 != -100]
			
			vF_x1,vF_y1 = velocityTBNem(kx1,ky1,1.,t1,t2,Phi)
			vF1 = np.sqrt(vF_x1**2. + vF_y1**2.)
			vF_x2,vF_y2 = velocityTBNem(kx2,ky2,1.,t1,t2,Phi)
			vF2 = np.sqrt(vF_x2**2. + vF_y2**2.)
			
			Yk1 = deltaFunc_SCNem(kx1,ky1,r)
			Yk2 = deltaFunc_SCNem(kx2,ky2,r)
			
			N0 = np.sum(4.*(dl1/vF1)) + np.sum(4.*(dl2/vF2))
			g = -1j*np.sum(4.*(dl1/vF1)*(x/np.sqrt(Yk1**2. - (x + omega*1j)**2.)))/N0 + np.sum(4.*(dl2/vF2)*(x/np.sqrt((x + omega*1j)**2. - Yk2**2.)))/N0
			h = -1j*np.sum(4.*(dl1/vF1)*(Yk1/np.sqrt(Yk1**2. - (x + omega*1j)**2.)))/N0 + np.sum(4.*(dl2/vF2)*(Yk2/np.sqrt((x + omega*1j)**2. - Yk2**2.)))/N0
			
		elif nLists == 3:
			kx1 = np.linspace(1e-8,zeros[0]-1e-8,N_kx)
			kx2 = np.linspace(zeros[0]+1e-8,zeros[1]-1e-8,N_kx)
			kx3 = np.linspace(zeros[1]+1e-8,np.pi-1e-8,N_kx)
			ky1 = np.arccos(-(mu + 2.*t1*np.cos(kx1) - Phi*np.cos(kx1))/(2.*t1 + 4.*t2*np.cos(kx1) + Phi))
			ky2 = np.arccos(-(mu + 2.*t1*np.cos(kx2) - Phi*np.cos(kx2))/(2.*t1 + 4.*t2*np.cos(kx2) + Phi))
			ky3 = np.arccos(-(mu + 2.*t1*np.cos(kx3) - Phi*np.cos(kx3))/(2.*t1 + 4.*t2*np.cos(kx3) + Phi))
			
			xi1 = dispersionTB(mu,kx1,ky1,1.,t1,t2) + Phi*PhiFunc(kx1,ky1)
			xi2 = dispersionTB(mu,kx2,ky2,1.,t1,t2) + Phi*PhiFunc(kx2,ky2)
			xi3 = dispersionTB(mu,kx3,ky3,1.,t1,t2) + Phi*PhiFunc(kx3,ky3)
			
			dkx1 = myGradient(kx1)
			dky1 = myGradient(ky1)
			dl1 = np.sqrt(dkx1**2. + dky1**2.)
			dkx2 = myGradient(kx2)
			dky2 = myGradient(ky2)
			dl2 = np.sqrt(dkx2**2. + dky2**2.)
			dkx3 = myGradient(kx3)
			dky3 = myGradient(ky3)
			dl3 = np.sqrt(dkx3**2. + dky3**2.)
			
			kx1[np.abs(xi1) > 1e-4] = -100
			ky1[np.abs(xi1) > 1e-4] = -100
			
			kx2[np.abs(xi2) > 1e-4] = -100
			ky2[np.abs(xi2) > 1e-4] = -100
			
			kx3[np.abs(xi3) > 1e-4] = -100
			ky3[np.abs(xi3) > 1e-4] = -100

			dkx1 = dkx1[kx1 != -100]
			dky1 = dky1[kx1 != -100]
			dl1 = dl1[kx1 != -100]
			kx1 = kx1[kx1 != -100]
			ky1 = ky1[ky1 != -100]
			dl2 = dl2[kx2 != -100]
			dkx2 = dkx2[kx2 != -100]
			dky2 = dky2[kx2 != -100]
			kx2 = kx2[kx2 != -100]
			ky2 = ky2[ky2 != -100]
			dl3 = dl3[kx3 != -100]
			dkx3 = dkx3[kx3 != -100]
			dky3 = dky3[kx3 != -100]
			kx3 = kx3[kx3 != -100]
			ky3 = ky3[ky3 != -100]
			
			vF_x1,vF_y1 = velocityTBNem(kx1,ky1,1.,t1,t2,Phi)
			vF1 = np.sqrt(vF_x1**2. + vF_y1**2.)
			vF_x2,vF_y2 = velocityTBNem(kx2,ky2,1.,t1,t2,Phi)
			vF2 = np.sqrt(vF_x2**2. + vF_y2**2.)
			vF_x3,vF_y3 = velocityTBNem(kx3,ky3,1.,t1,t2,Phi)
			vF3 = np.sqrt(vF_x3**2. + vF_y3**2.)
			
			Yk1 = deltaFunc_SCNem(kx1,ky1,r)
			Yk2 = deltaFunc_SCNem(kx2,ky2,r)
			Yk3 = deltaFunc_SCNem(kx3,ky3,r)
			
			N0 = np.sum(4.*(dl1/vF1)) + np.sum(4.*(dl2/vF2)) + np.sum(4.*(dl3/vF3))
			g = -1j*np.sum(4.*(dl1/vF1)*(x/np.sqrt(Yk1**2. - (x + omega*1j)**2.)))/N0 + np.sum(4.*(dl2/vF2)*(x/np.sqrt((x + omega*1j)**2. - Yk2**2.)))/N0 - 1j*np.sum(4.*(dl3/vF3)*(x/np.sqrt(Yk3**2. - (x + omega*1j)**2.)))/N0
			h = -1j*np.sum(4.*(dl1/vF1)*(Yk1/np.sqrt(Yk1**2. - (x + omega*1j)**2.)))/N0 + np.sum(4.*(dl2/vF2)*(Yk2/np.sqrt((x + omega*1j)**2. - Yk2**2.)))/N0 - 1j*np.sum(4.*(dl3/vF3)*(Yk3/np.sqrt(Yk3**2. - (x + omega*1j)**2.)))/N0
	else:
		a = 4*t2
		b1 = 4*t1 + 4*x*t2
		b2 = 4*t1 - 4*x*t2
		c1 = mu + 2*x*t1
		c2 = mu - 2*x*t1
		
		kx1_arg = (-b1+np.sqrt(b1**2.-4*a*c1))/(2*a)
		kx2_arg = (-b2+np.sqrt(b2**2.-4*a*c2))/(2*a)

		if np.arccos(kx1_arg) < np.arccos(-(mu+2*t1)/(2*t1+4*t2)):
			nLists = nLists + 1
			zeros.append(np.arccos(kx1_arg))
		if np.arccos(kx2_arg) < np.arccos(-(mu+2*t1)/(2*t1+4*t2)):
			nLists = nLists + 1
			zeros.append(np.arccos(kx2_arg))
		
		zeros = np.array(zeros)
		zeros = zeros[np.argsort(zeros)]

		if nLists == 1:
			kx = np.linspace(1e-8,np.arccos(-(mu+2*t1)/(2*t1+4*t2))-1e-8,N_kx)
			ky = np.arccos(-(mu+2*t1*np.cos(kx))/(2*t1+4*t2*np.cos(kx)))
			
			dkx = myGradient(kx)
			dky = myGradient(ky)
			dl = np.sqrt(dkx**2. + dky**2.)
			
			vF_x,vF_y = velocityTB(kx,ky,1.,t1,t2)
			vF = np.sqrt(vF_x**2. + vF_y**2.)
			
			Yk = deltaFunc(kx,ky)
			
			N0 = np.sum(4.*(dl/vF))
			g = np.sum(4.*(dl/vF)*(x/np.sqrt((x + omega*1j)**2. - Yk**2.)))/N0
			h = np.sum(4.*(dl/vF)*(Yk/np.sqrt((x + omega*1j)**2. - Yk**2.)))/N0
			
		elif nLists == 2:
			kx1 = np.linspace(1e-8,zeros[0]-1e-8,N_kx)
			kx2 = np.linspace(zeros[0]+1e-8,np.arccos(-(mu+2*t1)/(2*t1+4*t2))-1e-8,N_kx)
			ky1 = np.arccos(-(mu+2*t1*np.cos(kx1))/(2*t1+4*t2*np.cos(kx1)))
			ky2 = np.arccos(-(mu+2*t1*np.cos(kx2))/(2*t1+4*t2*np.cos(kx2)))
			
			dkx1 = myGradient(kx1)
			dky1 = myGradient(ky1)
			dl1 = np.sqrt(dkx1**2. + dky1**2.)
			dkx2 = myGradient(kx2)
			dky2 = myGradient(ky2)
			dl2 = np.sqrt(dkx2**2. + dky2**2.)
			
			vF_x1,vF_y1 = velocityTB(kx1,ky1,1.,t1,t2)
			vF1 = np.sqrt(vF_x1**2. + vF_y1**2.)
			vF_x2,vF_y2 = velocityTB(kx2,ky2,1.,t1,t2)
			vF2 = np.sqrt(vF_x2**2. + vF_y2**2.)
			
			Yk1 = deltaFunc(kx1,ky1)
			Yk2 = deltaFunc(kx2,ky2)
			
			N0 = np.sum(4.*(dl1/vF1)) + np.sum(4.*(dl2/vF2))
			g = -1j*np.sum(4.*(dl1/vF1)*(x/np.sqrt(Yk1**2. - (x + omega*1j)**2.)))/N0 + np.sum(4.*(dl2/vF2)*(x/np.sqrt((x + omega*1j)**2. - Yk2**2.)))/N0
			h = -1j*np.sum(4.*(dl1/vF1)*(Yk1/np.sqrt(Yk1**2. - (x + omega*1j)**2.)))/N0 + np.sum(4.*(dl2/vF2)*(Yk2/np.sqrt((x + omega*1j)**2. - Yk2**2.)))/N0
			
		elif nLists == 3:
			kx1 = np.linspace(1e-8,zeros[0]-1e-8,N_kx)
			kx2 = np.linspace(zeros[0]+1e-8,zeros[1]-1e-8,N_kx)
			kx3 = np.linspace(zeros[1]+1e-8,np.arccos(-(mu+2*t1)/(2*t1+4*t2))-1e-8,N_kx)
			ky1 = np.arccos(-(mu+2*t1*np.cos(kx1))/(2*t1+4*t2*np.cos(kx1)))
			ky2 = np.arccos(-(mu+2*t1*np.cos(kx2))/(2*t1+4*t2*np.cos(kx2)))
			ky3 = np.arccos(-(mu+2*t1*np.cos(kx3))/(2*t1+4*t2*np.cos(kx3)))
			
			dkx1 = myGradient(kx1)
			dky1 = myGradient(ky1)
			dl1 = np.sqrt(dkx1**2. + dky1**2.)
			dkx2 = myGradient(kx2)
			dky2 = myGradient(ky2)
			dl2 = np.sqrt(dkx2**2. + dky2**2.)
			dkx3 = myGradient(kx3)
			dky3 = myGradient(ky3)
			dl3 = np.sqrt(dkx3**2. + dky3**2.)
			
			vF_x1,vF_y1 = velocityTBNem(kx1,ky1,1.,t1,t2,Phi)
			vF1 = np.sqrt(vF_x1**2. + vF_y1**2.)
			vF_x2,vF_y2 = velocityTBNem(kx2,ky2,1.,t1,t2,Phi)
			vF2 = np.sqrt(vF_x2**2. + vF_y2**2.)
			vF_x3,vF_y3 = velocityTBNem(kx3,ky3,1.,t1,t2,Phi)
			vF3 = np.sqrt(vF_x3**2. + vF_y3**2.)
			
			Yk1 = deltaFunc(kx1,ky1)
			Yk2 = deltaFunc(kx2,ky2)
			Yk3 = deltaFunc(kx3,ky3)
			
			N0 = np.sum(4.*(dl1/vF1)) + np.sum(4.*(dl2/vF2)) + np.sum(4.*(dl3/vF3))
			g = -1j*np.sum(4.*(dl1/vF1)*(x/np.sqrt(Yk1**2. - (x + omega*1j)**2.)))/N0 + np.sum(4.*(dl2/vF2)*(x/np.sqrt((x + omega*1j)**2. - Yk2**2.)))/N0 - 1j*np.sum(4.*(dl3/vF3)*(x/np.sqrt(Yk3**2. - (x + omega*1j)**2.)))/N0
			h = -1j*np.sum(4.*(dl1/vF1)*(Yk1/np.sqrt(Yk1**2. - (x + omega*1j)**2.)))/N0 + np.sum(4.*(dl2/vF2)*(Yk2/np.sqrt((x + omega*1j)**2. - Yk2**2.)))/N0 - 1j*np.sum(4.*(dl3/vF3)*(Yk3/np.sqrt(Yk3**2. - (x + omega*1j)**2.)))/N0	

	return g,h

def N_func(FS_geom,N_phi,mu,t1,t2,Xi,Phi,PhiFunc):
	phi = np.linspace(np.pi-1e-8,1e-8,N_phi)
	kF = np.zeros(N_phi)
	kx = np.zeros(N_phi)
	ky = np.zeros(N_phi)
	
	for i in range(0,N_phi):
		if FS_geom == 'closed':
			kF[i] = fsolve(lambda k: (dispersionTB(mu,k*np.cos(phi[i]),k*np.sin(phi[i]),1.,t1,t2) - Xi + Phi*PhiFunc(k*np.cos(phi[i]),k*np.sin(phi[i])))**2,2.)
			kx[i] = kF[i]*np.cos(phi[i])
			ky[i] = kF[i]*np.sin(phi[i])
			
			xi = dispersionTB(mu,kx[i],ky[i],1.,t1,t2) + Phi*PhiFunc(kx[i],ky[i])
			if np.abs(xi - Xi) > 1e-4:
				kx[i] = -100
				ky[i] = -100
						
		elif FS_geom == 'open':
			if phi[i] <= np.pi/2.:
				kF[i] = fsolve(lambda k: (dispersionTB(mu,np.pi-k*np.cos(np.pi/2.-phi[i]),np.pi-k*np.sin(np.pi/2.-phi[i]),1.,t1,t2) - Xi + Phi*PhiFunc(np.pi-k*np.cos(np.pi/2.-phi[i]),np.pi-k*np.sin(np.pi/2.-phi[i])))**2,2.)
				kx[i] = np.pi-kF[i]*np.cos(np.pi/2.-phi[i])
				ky[i] = np.pi-kF[i]*np.sin(np.pi/2.-phi[i])
				xi = dispersionTB(mu,kx[i],ky[i],1.,t1,t2) + Phi*PhiFunc(kx[i],ky[i])
				if np.abs(xi - Xi) > 1e-4:
					kx[i] = -100
					ky[i] = -100
			elif phi[i] > np.pi/2.:
				kF[i] = fsolve(lambda k: (dispersionTB(mu,-np.pi-k*np.cos(3.*np.pi/2.-phi[i]),np.pi-k*np.sin(3.*np.pi/2.-phi[i]),1.,t1,t2) - Xi + Phi*PhiFunc(-np.pi-k*np.cos(3.*np.pi/2.-phi[i]),np.pi-k*np.sin(3.*np.pi/2.-phi[i])))**2,2.)
				kx[i] = -np.pi-kF[i]*np.cos(3.*np.pi/2.-phi[i])
				ky[i] = np.pi-kF[i]*np.sin(3.*np.pi/2.-phi[i])
				xi = dispersionTB(mu,kx[i],ky[i],1.,t1,t2) + Phi*PhiFunc(kx[i],ky[i])
				if np.abs(xi - Xi) > 1e-4:
					kx[i] = -100
					ky[i] = -100
	
	dphi = phi[0] - phi[1]
	dl = np.abs(kF)*dphi/np.cos(dphi)
	
	dl = dl[kx != -100]
	kx = kx[kx != -100]
	ky = ky[ky != -100]
	
	if Phi > 0.:
		vx,vy = velocityTBNem(kx,ky,1.,t1,t2,Phi)
	else:
		vx,vy = velocityTB(kx,ky,1.,t1,t2)
	v = np.sqrt(vx**2. + vy**2.)
	
	N = np.sum(dl/v)
	return N

def FS_grid(FS_shape,FS_geom,SC_order,mu,delta,deltaFunc,Phi,PhiFunc,r0,r1,T,N_xi,N_phi,t1,t2,E_c):
	
	xi_array = np.linspace(-E_c,E_c,N_xi)
	phi = np.linspace(np.pi-1e-8,0.+1e-8,N_phi)
	phi_grid,xi_grid = np.meshgrid(phi,xi_array)
	if SC_order == 'pureSC_swave' or SC_order == 'pureSC_dx2y2':
		tildeN = np.ones(xi_grid.shape)
		N0 = 1.
	else:
		tildeN = np.zeros(xi_grid.shape)
		for i in range(0,N_xi):
			tildeN[i] = N_func(FS_geom,2000,mu,t1,t2,xi_array[i],Phi,PhiFunc)*np.ones(N_phi)
		N0 = N_func(FS_geom,2000,mu,t1,t2,0.,Phi,PhiFunc)
	
	kF = np.zeros(xi_grid.shape)
	kx = np.zeros(xi_grid.shape)
	ky = np.zeros(xi_grid.shape)
	
	for i in range(0,N_xi):
		for j in range(0,N_phi):
			if FS_shape == 'TB':
				if FS_geom == 'closed':
						kF[i][j] = fsolve(lambda k: (dispersionTB(mu,k*np.cos(phi_grid[i][j]),k*np.sin(phi_grid[i][j]),1.,t1,t2) - xi_grid[i][j] + Phi*PhiFunc(k*np.cos(phi_grid[i][j]),k*np.sin(phi_grid[i][j])))**2,2.)
						kx[i][j] = kF[i][j]*np.cos(phi_grid[i][j])
						ky[i][j] = kF[i][j]*np.sin(phi_grid[i][j])
						
						xi = dispersionTB(mu,kx[i][j],ky[i][j],1.,t1,t2) + Phi*PhiFunc(kx[i][j],ky[i][j])
						if np.abs(xi - xi_grid[i][j]) > 1e-4:
							kx[i][j] = -100
							ky[i][j] = -100
						
				elif FS_geom == 'open':
						if phi_grid[i][j] <= np.pi/2.:
							kF[i][j] = fsolve(lambda k: (dispersionTB(mu,np.pi-k*np.cos(np.pi/2.-phi_grid[i][j]),np.pi-k*np.sin(np.pi/2.-phi_grid[i][j]),1.,t1,t2) - xi_grid[i][j] + Phi*PhiFunc(np.pi-k*np.cos(np.pi/2.-phi_grid[i][j]),np.pi-k*np.sin(np.pi/2.-phi_grid[i][j])))**2,2.)
							kx[i][j] = np.pi-kF[i][j]*np.cos(np.pi/2.-phi_grid[i][j])
							ky[i][j] = np.pi-kF[i][j]*np.sin(np.pi/2.-phi_grid[i][j])
							xi = dispersionTB(mu,kx[i][j],ky[i][j],1.,t1,t2) + Phi*PhiFunc(kx[i][j],ky[i][j])
							if np.abs(xi - xi_grid[i][j]) > 1e-4:
								kx[i][j] = -100
								ky[i][j] = -100

						
						elif phi_grid[i][j] > np.pi/2.:
							kF[i][j] = fsolve(lambda k: (dispersionTB(mu,-np.pi-k*np.cos(3.*np.pi/2.-phi_grid[i][j]),np.pi-k*np.sin(3.*np.pi/2.-phi_grid[i][j]),1.,t1,t2) - xi_grid[i][j] + Phi*PhiFunc(-np.pi-k*np.cos(3.*np.pi/2.-phi_grid[i][j]),np.pi-k*np.sin(3.*np.pi/2.-phi_grid[i][j])))**2,2.)
							kx[i][j] = -np.pi-kF[i][j]*np.cos(3.*np.pi/2.-phi_grid[i][j])
							ky[i][j] = np.pi-kF[i][j]*np.sin(3.*np.pi/2.-phi_grid[i][j])
							xi = dispersionTB(mu,kx[i][j],ky[i][j],1.,t1,t2) + Phi*PhiFunc(kx[i][j],ky[i][j])
							if np.abs(xi - xi_grid[i][j]) > 1e-4:
								kx[i][j] = -100
								ky[i][j] = -100

			elif FS_shape == 'Circular':
				kF[i][j] = fsolve(lambda k: (dispersionCircular(np.pi**2.,k*np.cos(phi_grid[i][j]),k*np.sin(phi_grid[i][j])) - xi_grid[i][j] + Phi*PhiFunc(phi_grid[i][j],0.))**2,3.)
				kx[i][j] = kF[i][j]*np.cos(phi_grid[i][j])
				ky[i][j] = kF[i][j]*np.sin(phi_grid[i][j])

	kFermi_array = kF[int(N_xi/2)]
	phi_array = phi_grid[int(N_xi/2)]
	
	dphi = phi[0] - phi[1]

	if FS_shape == 'TB':
		vel_x_n,vel_y_n = velocityTBNem(kx,ky,1.,t1,t2,Phi)
	elif FS_shape == 'Circular':
		vel_x_n,vel_y_n = velocityCircular(kx,ky)
	
	dl = np.abs(kF)*dphi/np.cos(dphi)
	
	xi_k = dispersionTB(mu,kx,ky,1.,t1,t2) + Phi*PhiFunc(kx,ky)

	if SC_order == 'pureSC_dx2y2':
		delta_k = delta*deltaFunc_dx2y2(kx,ky)
	elif SC_order == 'pureSC_swave':
		delta_k = delta*deltaFunc_swave(kx,ky)
	else:
		delta_k = delta*deltaFunc_SCNem(kx,ky,r1)

	E_k = np.zeros(kx.shape)
	
	for i in range(0,N_xi):
		for j in range(0,N_phi):
			minus, plus = eigenvalues2D(np.array([[xi_k[i][j],-delta_k[i][j]],[-delta_k[i][j],-xi_k[i][j]]]))
			E_k[i][j] = plus
	E_k = np.array(E_k)

	g = np.zeros(kx.shape)
	h = np.zeros(kx.shape)
	a = np.zeros(kx.shape)
	b = np.zeros(kx.shape)
	c = np.zeros(kx.shape)

	if delta != 0.:
		g = g + 0j
		h = h + 0j
		
		for i in range(0,N_xi):
			for j in range(0,N_phi):
				x = E_k[i][j]/delta + 1e-3
				if Phi != 0.:
					g[i][j], h[i][j] = ghFunc_TB_Cartesian(x,mu,deltaFunc,Phi,PhiFunc,r1,1000,t1,t2)
				else:
					g[i][j], h[i][j] = ghFunc_TB_Cartesian(x,mu,deltaFunc,Phi,PhiFunc,r1,10000,t1,t2)
	
	else:
		g = np.ones(kx.shape)
		h = np.zeros(kx.shape)
		
	if SC_order == 'pureSC_dx2y2':
		h = np.zeros(kx.shape)
		
	a = np.abs(g * g.conj()) + np.abs(h * h.conj())
	b = np.abs(g * g.conj()) - np.abs(h * h.conj())
	c = (g * h.conj()).real
	denom = (np.abs(g.real**2. - g.imag**2.) - np.abs(h.real**2. - h.imag**2.))**2.
	
	mask = denom.real != 0.
	a[mask] = a[mask].real/denom[mask].real
	a[~mask] = 0.
	b[mask] = b[mask].real/denom[mask].real
	b[~mask] = 0.
	c[mask] = c[mask]/denom[mask].real
	c[~mask] = 0.
	
	a[np.isnan(a)] = 0.
	b[np.isnan(b)] = 0.
	c[np.isnan(c)] = 0.
	
	if delta != 0.:
		err = 1e-8
		vel_x = vel_x_n*np.abs(xi_k)/E_k
		vel_y = vel_y_n*np.abs(xi_k)/E_k
		
		vel_x[np.abs(delta_k) < err] = vel_x_n[np.abs(delta_k) < err]
		vel_y[np.abs(delta_k) < err] = vel_y_n[np.abs(delta_k) < err]
		
		vel_x[np.isnan(vel_x)] = vel_x_n[np.isnan(vel_x)]
		vel_y[np.isnan(vel_y)] = vel_y_n[np.isnan(vel_y)]
	else:
		vel_x = vel_x_n
		vel_y = vel_y_n
	
	vF = np.sqrt(vel_x_n**2. + vel_y_n**2.)

	phi = phi_grid[ky != -100].flatten()
	xi = xi_k[ky != -100].flatten()
	Delta = delta_k[ky != -100].flatten()
	E = E_k[ky != -100].flatten()
	dl = dl[ky != -100].flatten()
	vx = vel_x[ky != -100].flatten()
	vy = vel_y[ky != -100].flatten()
	vF = vF[ky != -100].flatten()
	g = g[ky != -100].flatten()
	h = h[ky != -100].flatten()
	a = a[ky != -100].flatten()
	b = b[ky != -100].flatten()
	c = c[ky != -100].flatten()
	tildeN = tildeN[ky != -100].flatten()
	kx = kx[ky != -100].flatten()
	ky = ky[ky != -100].flatten()
	
	if delta > 0.:
		fermi_deriv_FBZ = fermi_deriv(E,T)
	else:
		fermi_deriv_FBZ = fermi_deriv(xi,T)
	
	return kx, ky, phi, xi, Delta, E, vx, vy, vF, dl, fermi_deriv_FBZ, g, h, a, b, c, tildeN, N0
	
def SC_scattering_analytic(SC_order,kx,ky,r,E_k,delta,g,h,a,c,tildeN,N0):
	Tau_Born = np.zeros(kx.shape)
	Tau_Unitary = np.zeros(kx.shape)
	if SC_order == 'SCNem' or SC_order == 'pureNem':
		if delta == 0.:
			Tau_Born = 1./tildeN
			Tau_Unitary = (N0**2.)/tildeN
		else:
			x = E_k/delta
			Yk = deltaFunc_SCNem(kx,ky,r)
			
			Tau_Born[np.abs(g.real - Yk*h.real/x) > 1e-8] = 1./((tildeN)*(g.real - Yk*h.real/x))[np.abs(g.real - Yk*h.real/x) > 1e-8]
			Tau_Unitary[np.abs(a*(g.real + Yk*h.real/x) + 2.*c*(Yk*g.real/x + h.real)) > 1e-8] = (N0**2.)/((tildeN)*(a*(g.real + Yk*h.real/x) + 2.*c*(Yk*g.real/x + h.real)))[np.abs(a*(g.real + Yk*h.real/x) + 2.*c*(Yk*g.real/x + h.real)) > 1e-8]
	elif SC_order == 'pureSC_dx2y2':
		if delta == 0.:
			Tau_Born = np.ones(kx.shape)
			Tau_Unitary = np.ones(kx.shape)
		else:
			Tau_Born = 1./(g.real)
			Tau_Unitary = (g.real**2. + g.imag**2)/(g.real)
	elif SC_order == 'pureSC_swave':
		if delta == 0.:
			Tau_Born = np.ones(kx.shape)
			Tau_Unitary = np.ones(kx.shape)
		else:
			x = E_k/delta
			Yk = deltaFunc_swave(kx,ky)
			
			Tau_Born[(1-(delta**2.)/(E_k**2.)) > 1e-3] = 1./(g.real*(1-(delta**2.)/(E_k**2.)))[(1-(delta**2.)/(E_k**2.)) > 1e-3]
			
			Tau_Unitary[np.abs(a*(g.real + Yk*h.real/x) + 2.*c*(Yk*g.real/x + h.real)) > 1e-8] = 1./((a*(g.real + Yk*h.real/x) + 2.*c*(Yk*g.real/x + h.real)))[np.abs(a*(g.real + Yk*h.real/x) + 2.*c*(Yk*g.real/x + h.real)) > 1e-8]
	return Tau_Born,Tau_Unitary

def SC_conductivity(FS_shape,FS_geom,SC_order,mu,delta,deltaFunc,Phi,PhiFunc,r0,r1,N_xi,N_phi,E_cutoff,f,t1,t2,T,plot):
	
	kx,ky,phi,xi_k,delta_k,E_k,vx,vy,v,dl,fermi_deriv_FBZ,g,h,a,b,c,tildeN,N0 = FS_grid(FS_shape,FS_geom,SC_order,mu,delta,deltaFunc,Phi,PhiFunc,r0,r1,T,N_xi,N_phi,t1,t2,E_cutoff)
	badDeltaMask = np.abs(xi_k) < E_cutoff - 2.*f*(T+1)/2.
	
	Tau_SC_Born,Tau_SC_Unitary = SC_scattering_analytic(SC_order,kx,ky,r1,E_k,delta,g,h,a,c,tildeN,N0)
	
	if plot:
		
		gFig = plt.figure(figsize=(24,18))
		plt.scatter(E_k/delta,1./(g.real*(1-(delta**2.)/(E_k**2.))))
		
		plt.savefig('./picsHeatmap/g_' + str(N_phi) + '_' + str(int(1000*T)) + '.png')
		gFig.clear()
		plt.close(gFig)
		del gFig
		
		sort = np.argsort(Tau_SC_Born[badDeltaMask])
		TauFig = plt.figure(figsize=(24,18))
		plt.scatter(kx[badDeltaMask][sort],ky[badDeltaMask][sort],c = Tau_SC_Born[badDeltaMask][sort],cmap = 'inferno')
		plt.colorbar()
		plt.xlim([-1.2*max(ky.flatten()),1.2*max(ky.flatten())])
		plt.ylim([0.,1.2*max(ky.flatten())])
		plt.title(r'$\tau_k^{Born}$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_N$' + ')')
		plt.savefig('./picsHeatmap/Tau_Born_' + str(N_phi) + '_' + str(int(1000*T)) + '.png')
		TauFig.clear()
		plt.close(TauFig)
		del TauFig
		
		sort = np.argsort(Tau_SC_Unitary[badDeltaMask])
		TauFig = plt.figure(figsize=(24,18))
		plt.scatter(kx[badDeltaMask][sort],ky[badDeltaMask][sort],c = Tau_SC_Unitary[badDeltaMask][sort],cmap = 'inferno')
		plt.colorbar()
		plt.xlim([-1.2*max(ky.flatten()),1.2*max(ky.flatten())])
		plt.ylim([0.,1.2*max(ky.flatten())])
		plt.title(r'$\tau_k^{Unitary}$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_N$' + ')')
		plt.savefig('./picsHeatmap/Tau_Unitary_' + str(N_phi) + '_' + str(int(1000*T)) + '.png')
		TauFig.clear()
		plt.close(TauFig)
		del TauFig
		
		EFig = plt.figure(figsize=(24,18))
		plt.scatter(kx[badDeltaMask],ky[badDeltaMask],c = E_k[badDeltaMask]**2.,cmap = 'inferno')
		plt.colorbar()
		plt.xlim([-1.2*max(ky.flatten()),1.2*max(ky.flatten())])
		plt.ylim([0.,1.2*max(ky.flatten())])
		plt.title(r'$E_k^2$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_N$' +')')
		plt.savefig('./picsHeatmap/E2_' + str(N_phi) + '_' + str(int(1000*T)) + '.png')
		EFig.clear()
		plt.close(EFig)
		del EFig
        
		vxFig = plt.figure(figsize=(24,18))
		plt.scatter(kx[badDeltaMask],ky[badDeltaMask],c = vx[badDeltaMask],cmap = 'inferno')
		plt.colorbar()
		plt.xlim([-1.2*max(ky.flatten()),1.2*max(ky.flatten())])
		plt.ylim([0.,1.2*max(ky.flatten())])
		plt.title(r'$v_x$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_N$' +')')
		plt.savefig('./picsHeatmap/vx_' + str(N_phi) + '_' + str(int(1000*T)) + '.png')
		vxFig.clear()
		plt.close(vxFig)
		del vxFig
		
		vyFig = plt.figure(figsize=(24,18))
		plt.scatter(kx[badDeltaMask],ky[badDeltaMask],c = vy[badDeltaMask],cmap = 'inferno')
		plt.colorbar()
		plt.xlim([-1.2*max(ky.flatten()),1.2*max(ky.flatten())])
		plt.ylim([0.,1.2*max(ky.flatten())])
		plt.title(r'$v_y$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_N$' +')')
		plt.savefig('./picsHeatmap/vy_' + str(N_phi) + '_' + str(int(1000*T)) + '.png')
		vyFig.clear()
		plt.close(vyFig)
		del vyFig
		
		sort = np.argsort(fermi_deriv_FBZ[badDeltaMask])
		fermiDerivFig = plt.figure(figsize=(24,18))
		plt.scatter(kx[badDeltaMask][sort],ky[badDeltaMask][sort],c = fermi_deriv_FBZ[badDeltaMask][sort], cmap = 'inferno')
		plt.colorbar()
		plt.xlim([-1.2*max(ky.flatten()),1.2*max(ky.flatten())])
		plt.ylim([0.,1.2*max(ky.flatten())])
		plt.title(r'$\frac{\partial f}{\partial \epsilon}$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_N$' +')')
		plt.savefig('./picsHeatmap/fermiDeriv_' + str(N_phi) + '_' + str(int(1000*T)) + '.png')
		fermiDerivFig.clear()
		plt.close(fermiDerivFig)
		del fermiDerivFig
		
		plt.close("all")
		
	kxx_Born = np.sum((vx[badDeltaMask]**2.)*(E_k[badDeltaMask]**2.)*Tau_SC_Born[badDeltaMask]*fermi_deriv_FBZ[badDeltaMask]*dl[badDeltaMask]/v[badDeltaMask])
	kxx_Born = np.sum((vx[badDeltaMask]**2.)*(E_k[badDeltaMask]**2.)*Tau_SC_Born[badDeltaMask]*fermi_deriv_FBZ[badDeltaMask]*dl[badDeltaMask]/v[badDeltaMask])
	kyy_Born = np.sum((vy[badDeltaMask]**2.)*(E_k[badDeltaMask]**2.)*Tau_SC_Born[badDeltaMask]*fermi_deriv_FBZ[badDeltaMask]*dl[badDeltaMask]/v[badDeltaMask])
	kxy_Born = np.sum((vx[badDeltaMask]*vy[badDeltaMask])*(E_k[badDeltaMask]**2.)*Tau_SC_Born[badDeltaMask]*fermi_deriv_FBZ[badDeltaMask]*dl[badDeltaMask]/v[badDeltaMask])
	kxx_Unitary = np.sum((vx[badDeltaMask]**2.)*(E_k[badDeltaMask]**2.)*Tau_SC_Unitary[badDeltaMask]*fermi_deriv_FBZ[badDeltaMask]*dl[badDeltaMask]/v[badDeltaMask])
	kyy_Unitary = np.sum((vy[badDeltaMask]**2.)*(E_k[badDeltaMask]**2.)*Tau_SC_Unitary[badDeltaMask]*fermi_deriv_FBZ[badDeltaMask]*dl[badDeltaMask]/v[badDeltaMask])
	kxy_Unitary = np.sum((vx[badDeltaMask]*vy[badDeltaMask])*(E_k[badDeltaMask]**2.)*Tau_SC_Unitary[badDeltaMask]*fermi_deriv_FBZ[badDeltaMask]*dl[badDeltaMask]/v[badDeltaMask])
	return kxx_Born,kyy_Born,kxy_Born,kxx_Unitary,kyy_Unitary,kxy_Unitary

def n_conductivity(FS_shape,FS_geom,mu,N_xi,N_phi,E_cutoff,f,t1,t2,T):
	
	kx,ky,phi,xi_k,delta_k,E_k,vx,vy,v,dl,fermi_deriv_FBZ,g,h,a,b,c,tildeN,N0 = FS_grid(FS_shape,FS_geom,'pureSC_dx2y2',mu,0.,deltaFunc_dx2y2,0.,PhiFunc,0.,0.,T,N_xi,N_phi,t1,t2,E_cutoff)
	badDeltaMask = np.abs(xi_k) < E_cutoff - 2.*f*(T+1)/2.

	Tau_n_Born,Tau_n_Unitary = SC_scattering_analytic('pureSC_dx2y2',kx,ky,0.,xi_k,0.,g,h,a,c,tildeN,N0)
	
	vx = np.array(vx)
	xi_k = np.array(xi_k)
	dl = np.array(dl)

	sort = np.argsort(Tau_n_Born[badDeltaMask])
	TauFig = plt.figure(figsize=(24,18))
	plt.scatter(kx[badDeltaMask][sort],ky[badDeltaMask][sort],c = Tau_n_Born[badDeltaMask][sort],cmap = 'inferno')
	plt.colorbar()
	plt.xlim([-1.2*max(ky.flatten()),1.2*max(ky.flatten())])
	plt.ylim([0.,1.2*max(ky.flatten())])
	plt.title(r'$\tau_n^{Born}$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_N$' + ')')
	plt.savefig('./picsHeatmap/Tau_n_Born_' + str(N_phi) + '_' + str(int(1000*T)) + '.png')
	TauFig.clear()
	plt.close(TauFig)
	del TauFig
	
	sort = np.argsort(Tau_n_Unitary[badDeltaMask])
	TauFig = plt.figure(figsize=(24,18))
	plt.scatter(kx[badDeltaMask][sort],ky[badDeltaMask][sort],c = Tau_n_Unitary[badDeltaMask][sort],cmap = 'inferno')
	plt.colorbar()
	plt.xlim([-1.2*max(ky.flatten()),1.2*max(ky.flatten())])
	plt.ylim([0.,1.2*max(ky.flatten())])
	plt.title(r'$\tau_n^{Unitary}$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_N$' + ')')
	plt.savefig('./picsHeatmap/Tau_n_Unitary_' + str(N_phi) + '_' + str(int(1000*T)) + '.png')
	TauFig.clear()
	plt.close(TauFig)
	del TauFig
	
	EFig = plt.figure(figsize=(24,18))
	plt.scatter(kx[badDeltaMask],ky[badDeltaMask],c = xi_k[badDeltaMask]**2.,cmap = 'inferno')
	plt.colorbar()
	plt.xlim([-1.2*max(ky.flatten()),1.2*max(ky.flatten())])
	plt.ylim([0.,1.2*max(ky.flatten())])
	plt.title(r'$\xi_k^2$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_N$' +')')
	plt.savefig('./picsHeatmap/xi2_' + str(N_phi) + '_' + str(int(1000*T)) + '.png')
	EFig.clear()
	plt.close(EFig)
	del EFig
       
	vxFig = plt.figure(figsize=(24,18))
	plt.scatter(kx[badDeltaMask],ky[badDeltaMask],c = vx[badDeltaMask],cmap = 'inferno')
	plt.colorbar()
	plt.xlim([-1.2*max(ky.flatten()),1.2*max(ky.flatten())])
	plt.ylim([0.,1.2*max(ky.flatten())])
	plt.title(r'$v_{nx}$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_N$' +')')
	plt.savefig('./picsHeatmap/vnx_' + str(N_phi) + '_' + str(int(1000*T)) + '.png')
	vxFig.clear()
	plt.close(vxFig)
	del vxFig
	
	vyFig = plt.figure(figsize=(24,18))
	plt.scatter(kx[badDeltaMask],ky[badDeltaMask],c = vy[badDeltaMask],cmap = 'inferno')
	plt.colorbar()
	plt.xlim([-1.2*max(ky.flatten()),1.2*max(ky.flatten())])
	plt.ylim([0.,1.2*max(ky.flatten())])
	plt.title(r'$v_{ny}$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_N$' +')')
	plt.savefig('./picsHeatmap/vny_' + str(N_phi) + '_' + str(int(1000*T)) + '.png')
	vyFig.clear()
	plt.close(vyFig)
	del vyFig
	
	sort = np.argsort(fermi_deriv_FBZ[badDeltaMask])
	fermiDerivFig = plt.figure(figsize=(24,18))
	plt.scatter(kx[badDeltaMask][sort],ky[badDeltaMask][sort],c = fermi_deriv_FBZ[badDeltaMask][sort], cmap = 'inferno')
	plt.colorbar()
	plt.xlim([-1.2*max(ky.flatten()),1.2*max(ky.flatten())])
	plt.ylim([0.,1.2*max(ky.flatten())])
	plt.title(r'$\frac{\partial f}{\partial \epsilon}$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_N$' +')')
	plt.savefig('./picsHeatmap/fermiDeriv_n_' + str(N_phi) + '_' + str(int(1000*T)) + '.png')
	fermiDerivFig.clear()
	plt.close(fermiDerivFig)
	del fermiDerivFig
	
	plt.close("all")
	
	return np.sum((vx[badDeltaMask]**2.)*(xi_k[badDeltaMask]**2.)*Tau_n_Born[badDeltaMask]*fermi_deriv_FBZ[badDeltaMask]*dl[badDeltaMask]/v[badDeltaMask]), np.sum((vx[badDeltaMask]**2.)*(xi_k[badDeltaMask]**2.)*Tau_n_Unitary[badDeltaMask]*fermi_deriv_FBZ[badDeltaMask]*dl[badDeltaMask]/v[badDeltaMask])

def thermalConductivity(args):
	T,FS_shape,FS_geom,whichNorm,SC_order,mu,deltaFunc,delta,PhiFunc,Phi,r0,r1,N_xi,N_phi,E_cutoff,f,t1,t2 = args
	kxx_b,kyy_b,kxy_b,kxx_u,kyy_u,kxy_u = SC_conductivity(FS_shape,FS_geom,SC_order,mu,delta,deltaFunc,Phi,PhiFunc,r0,r1,N_xi,N_phi,(T+.05)*E_cutoff,f,t1,t2,T,True)
	if whichNorm == 'Normal':
		k_n_b,k_n_u = n_conductivity(FS_shape,FS_geom,mu,N_xi,N_phi,(T+.05)*E_cutoff,f,t1,t2,T)
		kxx_n_b = k_n_b
		kyy_n_b = k_n_b
		kxx_n_u = k_n_u
		kyy_n_u = k_n_u
	elif whichNorm == 'Nematic':
		kxx_n_b,kyy_n_b,kxy_n_b,kxx_n_u,kyy_n_u,kxy_n_u = SC_conductivity(FS_shape,FS_geom,'pureNem',mu,0.,deltaFunc,Phi,PhiFunc,r0,r1,N_xi,N_phi,(T+.05)*E_cutoff,f,t1,t2,T,False)
	print(T,delta,Phi,kxx_n_b,kyy_n_b,kxx_n_u,kyy_n_u,kxx_b/kxx_n_b,kyy_b/kyy_n_b,kxy_b/kxx_n_b,kxx_u/kxx_n_u,kyy_u/kyy_n_u,kxy_u/kxx_n_u)
	return kxx_n_b,kyy_n_b,kxx_n_u,kyy_n_u,kxx_b/kxx_n_b,kyy_b/kyy_n_b,kxy_b/kxx_n_b,kxx_u/kxx_n_u,kyy_u/kyy_n_u,kxy_u/kxx_n_u
	

def main(FS_shape,FS_geom,whichNorm,SC_order,mu,deltaFunc,PhiFunc,r0,r1,num_T,N_xi,N_phi,E_cutoff,f,t1,t2):
	pool = mp.Pool(processes=8)
	
	kxx_Born = []
	kyy_Born = []
	kxy_Born = []
	kxx_Unitary = []
	kyy_Unitary = []
	kxy_Unitary = []
	kxx_n_Born = []
	kyy_n_Born = []
	kxx_n_Unitary = []
	kyy_n_Unitary = []
	#T = np.linspace(.001,.999,num_T)
	T = np.linspace(.01,1.,num_T)
	#T = np.linspace(.4,.6,num_T)
	if SC_order == 'pureNem':
		Phi = pureNemPhi(T,dispersionTB,PhiFunc,mu,1.,t1,t2,500)
		delta = np.zeros(T.shape)
	elif SC_order == 'SCNem':
		delta,Phi = deltaPhi(T,dispersionTB,deltaFunc_SCNem,PhiFunc,mu,1.,.4,t1,t2,500,r1)
	elif SC_order == 'pureSC_dx2y2':
		delta = deltaT(T,dispersionTB,deltaFunc_dx2y2,mu,1,t1,t2,500)
		Phi = np.zeros(T.shape)
	elif SC_order == 'pureSC_swave':
		delta = deltaT(T,dispersionTB,deltaFunc_swave,mu,1,t1,t2,500)
		Phi = np.zeros(T.shape)
	args = [(T[i],FS_shape,FS_geom,whichNorm,SC_order,mu,deltaFunc,delta[i],PhiFunc,Phi[i],r0,r1,N_xi,N_phi,E_cutoff,f,t1,t2) for i in range(0,num_T)]
	results = pool.map(thermalConductivity,args)
	print(' ')
	print('Results in proper order: ')
	print(' ')
	for i in range(0,len(results)):
		kxx_n_Born.append(results[i][0])
		kyy_n_Born.append(results[i][1])
		kxx_n_Unitary.append(results[i][2])
		kyy_n_Unitary.append(results[i][3])
		kxx_Born.append(results[i][4])
		kyy_Born.append(results[i][5])
		kxy_Born.append(results[i][6])
		kxx_Unitary.append(results[i][7])
		kyy_Unitary.append(results[i][8])
		kxy_Unitary.append(results[i][9])
		print(T[i],delta[i],Phi[i],kxx_n_Born[i],kyy_n_Born[i],kxx_n_Unitary[i],kyy_n_Unitary[i],kxx_Born[i],kyy_Born[i],kxy_Born[i],kxx_Unitary[i],kyy_Unitary[i],kxy_Unitary[i])


t0 = time.time()
main('TB','closed','Normal','pureSC_dx2y2',-4.8*10,deltaFunc_dx2y2,PhiFunc_Cartesian,1.,.6,100,401,600,5.,.05,6.*10,-1*10)
t1 = time.time()
print('time taken: ', t1-t0, ' seconds')
