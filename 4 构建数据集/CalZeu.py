import numpy as np 

def Zeu(chla=None, Rrs440=None, Rrs490=None, Rrs555=None, option=None,tE='0.5%'):
	if option == "Chl":
		return Zeu_simple(chla=chla,option=option)
		

	elif option == "IOP":
		shape = Rrs440.shape
		Rrs440,Rrs490,Rrs555 = Rrs440.flatten(),Rrs490.flatten(),Rrs555.flatten()
		z1 = np.array([Zeu_simple(Rrs440=Rrs440[i],Rrs490=Rrs490[i],Rrs555=Rrs555[i],option=option,tE=tE)  if ~np.isnan(Rrs440[i]+Rrs490[i]+Rrs555[i]) else np.nan for i in range(len(Rrs440))])
		return z1.reshape(shape)


def Zeu_simple(chla=None, Rrs440=None, Rrs490=None, Rrs555=None, option=None,tE=None):
	if option == "Chl":
		z1 = 34.0*(chla)**(-0.39)
		return z1
		

	elif option == "IOP":
		theta_a = 30/180*np.pi
		
		rrs440 = Rrs440/(0.52+1.7*Rrs440)
		rrs490 = Rrs490/(0.52+1.7*Rrs490)
		rrs555 = Rrs555/(0.52+1.7*Rrs555)

		# need rrs490 rrs555
		# give u490 u555
		g0, g1 = 0.0895, 0.1247
		u490 = (-g0+(g0**2+4*g1*rrs490)**0.5)/(2*g1)
		u555 = (-g0+(g0**2+4*g1*rrs555)**0.5)/(2*g1)

		# need rrs440 rrs555
		# give a555
		rho = np.log(rrs440/rrs555)
		a440i = np.exp(-2.0-1.4*rho+0.2*rho**2)
		a555 = 0.0596+0.2*(a440i-0.01)


		# need u555 u490 a555 rrs440 rrs555
		# give bb490 a490
		bbw555 = 0.000855824153404683
		bbp555 = (u555*a555)/(1-u555) - bbw555
		Y = 2.2*(1-1.2*np.exp(-0.9*(rrs440/rrs555)))
		bbw490 = 0.00147526187356561
		bbp490 = bbp555*(555/490)**(Y)
		bb490 = bbw490+bbp490

		a490 = (1-u490)*(bbw490+bbp490)/u490


		# need a490 bb490 theta_a
		# give K1 K2
		chi0,chi1,chi2 = -0.057, 0.482, 4.221
		zeta0,zeta1,zeta2 = 0.183, 0.702, -2.567
		alpha0,alpha1,alpha2 = 0.090,1.465,-0.667		
		K1 = (chi0+chi1*a490**0.5+chi2*bb490)*(1+alpha0*np.sin(theta_a))
		K2 = (zeta0+zeta1*a490+zeta2*bb490)*(alpha1+alpha2*np.cos(theta_a))


		# need K1 K2
		# give y1 y2 y3
		if tE == '0.5%':
			tauE = 5.2983
		elif tE == '1%':
			tauE = 4.605
# 		tauE = 5.2983 # z_1%: 4.605, z_0.5% 5.2983
		y1 = (K1**2-K2**2-2*tauE*K1)/K1**2
		y2 = (tauE**2-2*tauE*K1)/K1**2
		y3 = tauE**2/K1**2


		# 
		out3 = np.roots([1,y1,y2,y3])
		z = [out.real for out in out3 if np.isreal(out)]

		z = np.sort(z)
		# print(f"Rrs440:{Rrs440:.6f}, Rrs490:{Rrs490:.6f}, Rrs555:{Rrs555:.6f}, error z1 num: {len(z)}, z: {z}")
		if len(z)!=3 or z[0]>=0:
			print(f"Rrs440:{Rrs440:.6f}, Rrs490:{Rrs490:.6f}, Rrs555:{Rrs555:.6f}, error z1 num: {len(z)}, z: {z}")
			z1 = np.nan 
		else:
			z1 = z[1]

		return z1

if __name__ == '__main__':
	import numpy as np
	print(np.sin(40), np.sin(30/180*np.pi))
	print(np.log(np.e))

