import numpy as np  
def Zeu(chla=None, Rrs440=None, Rrs490=None, Rrs555=None, option=None):
	if option == "Chl":
		z1 = 34.0*((chla)**(-0.39))

	# elif option == "IOP":
	# 	rrs440 = Rrs440/(0.52+1.7*Rrs440)
	# 	rrs490 = Rrs490/(0.52+1.7*Rrs490)
	# 	rrs555 = Rrs555/(0.52+1.7*Rrs555)

	# 	# g0, g1 = 0.0949, 0.0794
	# 	g0, g1 = 0.0895, 0.1247
	# 	u490 = (-g0+(g0**2+4*g1*rrs490)**2)
	# 	u555 = (-g0+(g0**2+4*g1*rrs555)**2)

	# 	rho = np.log(rrs440/rrs555)
	# 	a440 = exp(-2.0-1.4*rho+0.2*rho**2)
	# 	a555 = 0.0596+0.2*(a440-0.001)

	# 	bbw555 = 0.000855824153404683
	# 	bbp555 = (u555*a555)/(1-u555) - bbw555
		
	# 	bbp490 = bbp555(555/490)**(Y)

	# 	bbw490 = 0.00147526187356561
	# 	a490 = (1-u490)*(bbw490+bbp490)/u490
	# 	bb490 = bbw490+bbp490


	return z1

def strictcase1water(Rrs412,Rrs443,Rrs490,Rrs560):
	# strict case1
    RR12 = Rrs412/Rrs443
    RR53 = Rrs560/Rrs490
    RR12_CS1 = 0.9351+0.113/RR53-0.0217/(RR53**2)+0.003/(RR53**3)
    Rrs555_CS1 = 0.0006+0.0027*RR53-0.0004*RR53**2-0.0002*RR53**3
    gamma,nu = 0.1,0.5
#     strict_case1 = ((1-gamma)*RR12_CS1<=RR12)&(RR12<=(1+gamma)*RR12_CS1)
    strict_case1 = ((1-nu)*Rrs555_CS1<=Rrs560)&(Rrs560<=(1+nu)*Rrs555_CS1)
    return strict_case1
