import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import rebound
from matplotlib.ticker import FormatStrFormatter
import time
from operator import attrgetter
import scipy.optimize as op
from scipy import stats
from pyevolve import Util
from pyevolve import GTree
from pyevolve import GSimpleGA
from pyevolve import Consts
from pyevolve import G1DList
from pyevolve import GAllele
from pyevolve import Crossovers
from pyevolve import Initializators
from pyevolve import Mutators
from pyevolve import Scaling
from pyevolve import Selectors
from pyevolve import DBAdapters

plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.borderpad'] = 0.5
plt.rcParams['legend.labelspacing'] = 0.1
plt.rcParams['legend.handletextpad'] = 0.1
plt.rcParams['font.family'] = 'stixgeneral'
plt.rcParams['font.size'] = 18
mpl.rcParams['legend.numpoints'] = 1
plt.rc('lines', linewidth=1.0)
colors = ['4D4D4D','5DA5DA', 'FAA43A', '60BD68', 'F17CB0','B2912F','B276B2','DECF3F','F15854']
#  (blue)
# (orange)
# (green)
# (pink)
#  (brown)
# (purple)
#  (yellow)
# (red)
# ']
mpl.rcParams['axes.color_cycle'] = colors


class RVPlanet:

    """Class creating planets, used to add planets to RVSystem"""
    
    def __init__(self, per=365.25, mass=1, M=0, e=0, pomega=0, i=90.,Omega=0):
        M_J = 9.5458e-4
        
        self.per = per
        self.mass = mass*M_J
        self.M = M
        self.e = e
        self.pomega = pomega
        self.i = i
        self.Omega = Omega
        self.l = (M + pomega)%360.


class RVSystem(RVPlanet):

    """Main class for RV Simulations"""
    
    def __init__(self,mstar=1.0):
        
        self.mstar = mstar #Mass of central star
        self.planets = [] #Array containing RVPlanets class
        self.RV_data = [] #Array of RV velocities, assumed to be of the form: JD, RV, error
        self.offsets = [] #Array of constant velocity offsets for each data set
        self.path_to_data = "" #Optional prefix that points to location of datasets
    
    
    def sort_data(self):    
        JDs = []
        vels = []
        errs = []

        for i,fname in enumerate(self.RV_files):
            tmp_arr = np.loadtxt(self.path_to_data + fname)
            JDs = np.concatenate((JDs,tmp_arr[:,0]))
            vels = np.concatenate((vels,(tmp_arr[:,1]-self.offsets[i])))
            errs = np.concatenate((errs,tmp_arr[:,2]))

        #There might be a better way to do this -- these commands sort the data by time so that we can integrate
        #up to each time
        sort_arr = [JDs,vels,errs]
        sort_arr = np.transpose(sort_arr)
        
        self.RV_data = sort_arr[np.argsort(sort_arr[:,0])]
        
        
        
        
    def add_planet(self,per=365.25, mass=1, M=0, e=0, pomega=0, i=90.,Omega=0):
        """Add planet to RV simulation. Angles are in degrees, planet mass is in Jupiter masses"""
        self.planets.append(RVPlanet(per,mass,M,e,pomega,i,Omega))
        
    def semi_maj(self):

        """List semi-major axes of all planets in simulation"""

        G = 6.674e-8
        JD_sec = 86400.0
        msun_g = 1.989e33
        AU = 1.496e13
            
        for i,planet in enumerate(self.planets):
            r = (G * (self.mstar)*msun_g * (planet.per*JD_sec)**2/(4.*np.pi**2))**(1./3.)
            r_AU = r/AU
            print "a_%i = %.3f AU" %(i,r_AU)

    def plot_RV(self,epoch=2450000,data=1,pnts_per_period=100.,save=0,filename=None):

        """Make a plot of the RV time series with data and integrated curve"""

        JDs = []
        vels = []
        errs = []

        for i,fname in enumerate(self.RV_data): #Read in RV data
            tmp_arr = np.loadtxt(self.path_to_data + fname)
            JDs.append(tmp_arr[:,0])
            vels.append(tmp_arr[:,1]-self.offsets[i])
            errs.append(tmp_arr[:,2])

        #Intialize Rebound simulation
        deg2rad = np.pi/180.
        sim = rebound.Simulation()
        sim.units = ('day', 'AU', 'Msun')
        sim.t = epoch #Epoch is the starting time of simulation
        sim.add(m=self.mstar,hash='star')


        min_per = np.inf
        max_per = 0

        for planet in self.planets: #Add planets in self.planets to Rebound simulation
            sim.add(m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
            min_per = min(min_per,planet.per) #Minimum period
            max_per = max(max_per,planet.per)

        
        JD_max = max(np.amax(JDs[i]) for i in range(len(self.RV_data)))
        JD_min = min(np.amin(JDs[i]) for i in range(len(self.RV_data)))


        Noutputs = int((JD_max-JD_min)/min_per*pnts_per_period)
        
        sim.move_to_com()
        ps = sim.particles

        times = np.linspace(JD_min, JD_max, Noutputs)

        if not(data):
            Noutputs = 1000
            times = np.linspace(0,10*max_per,Noutputs)
        AU_day_to_m_s = 1.731456e6 #Conversion factor from Rebound units to m/s

        rad_vels = np.zeros(Noutputs)


        for i,t in enumerate(times): #Perform integration
            sim.integrate(t)
            rad_vels[i] = -ps['star'].vz * AU_day_to_m_s
            # print i

        fig = plt.figure(1,figsize=(11,6)) #Plot RV

        plt.plot(times,rad_vels)

        if data:
                for i in range(len(self.RV_data)):
                    plt.errorbar(JDs[i],vels[i],yerr = errs[i],fmt='o')

        plt.xlabel("Time [JD]")
        plt.ylabel("RV [m/s]")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        if save == 1:
            plt.savefig(filename)
         
        
        plt.show()

       
    def calc_chi2(self,epoch=2450000,dt=0, star= None):

        """Calculate the chi^2 value of the RV time series for the planets currently in the system"""

        JDs = []
        vels = []
        errs = []

        for i,fname in enumerate(self.RV_data):
            tmp_arr = np.loadtxt(self.path_to_data + fname)
            JDs = np.concatenate((JDs,tmp_arr[:,0]))
            vels = np.concatenate((vels,(tmp_arr[:,1]-self.offsets[i])))
            errs = np.concatenate((errs,tmp_arr[:,2]))

        #There might be a better way to do this -- these commands sort the data by time so that we can integrate
        #up to each time
        sort_arr = [JDs,vels,errs]
        sort_arr = np.transpose(sort_arr)
        sort_arr = sort_arr[np.argsort(sort_arr[:,0])]
# add term for jitter here dependent on type as well as sub-type, use switch statement 
        jitter=np.zeros((len(errs),1))
        dev_prime_Gdwarf=3.5
        dev_prime_Atype=3.9
# this term can be updated on star type being analyzed, if you need to add another star type just add a switch statement below


        # here is switch for data type of star that needs to be added to chi-square calculation
        if star == 'G Dwarf':
            for x in (0,len(jitter)):
                jitter[i]=dev_prime_Gdwarf
        if star == 'A type':
            for x in (0,len(jitter)):
                jitter[i]=dev_prime_Atype
        
        deg2rad = np.pi/180.
        sim = rebound.Simulation()
        sim.units = ('day', 'AU', 'Msun')
        sim.t = epoch
        sim.add(m=self.mstar,hash='star')

        min_per = np.inf

        for planet in self.planets:
            sim.add(m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
            min_per = min(min_per,planet.per) #Minimum period

        sim.move_to_com()
        ps = sim.particles

        times = sort_arr[:,0] #Times to integrate to are just the times for each data point, no need to integrate
        #between data points

        if dt:
            sim.dt = min_per/dt

        AU_day_to_m_s = 1.731456e6
        rad_vels = np.zeros(len(times))

        for i,t in enumerate(times):
            sim.integrate(t)
            rad_vels[i] = -ps['star'].vz * AU_day_to_m_s

        chi_2 = 0

        for i,vel_theory in enumerate(rad_vels):
                chi_2 += (sort_arr[i,1]-vel_theory)**2/(sort_arr[i,2]**2+jitter[i]**2)
# denominator is error term, addend using jitter term as well, this is set above via switch.
        return np.asscalar(chi_2)
    
    def chi2_prob(self,chi=0,degrees=0):
        
        def uncertainty_chi2(degrees, chi):
            x=stats.chi2.cdf(x=chi,df=degrees)
            return x
        
        p=uncertainty_chi2(degrees,chi)
        return p
    
    def residuals(self,epoch=2450000,dt=0,star=None,save=0,filename=None):
        JDs = []
        vels = []
        errs = []

        for i,fname in enumerate(self.RV_data):
            tmp_arr = np.loadtxt(self.path_to_data + fname)
            JDs = np.concatenate((JDs,tmp_arr[:,0]))
            vels = np.concatenate((vels,(tmp_arr[:,1]-self.offsets[i])))
            errs = np.concatenate((errs,tmp_arr[:,2]))

        #There might be a better way to do this -- these commands sort the data by time so that we can integrate
        #up to each time
        sort_arr = [JDs,vels,errs]
        sort_arr = np.transpose(sort_arr)
        sort_arr = sort_arr[np.argsort(sort_arr[:,0])]
# add term for jitter here dependent on type as well as sub-type, use switch statement 
        jitter=np.zeros((len(errs),1))
        dev_prime_Gdwarf=3.5
        dev_prime_Atype=3.9
# this term can be updated on star type being analyzed, if you need to add another star type just add a switch statement below


        # here is switch for data type of star that needs to be added to chi-square calculation
        if star == 'G Dwarf':
            for x in (0,len(jitter)):
                jitter[i]=dev_prime_Gdwarf
        if star == 'A type':
            for x in (0,len(jitter)):
                jitter[i]=dev_prime_Atype
        
        deg2rad = np.pi/180.
        sim = rebound.Simulation()
        sim.units = ('day', 'AU', 'Msun')
        sim.t = epoch
        sim.add(m=self.mstar,hash='star')

        min_per = np.inf

        for planet in self.planets:
            sim.add(m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
            min_per = min(min_per,planet.per) #Minimum period

        sim.move_to_com()
        ps = sim.particles

        times = sort_arr[:,0] #Times to integrate to are just the times for each data point, no need to integrate
        #between data points

        if dt:
            sim.dt = min_per/dt

        AU_day_to_m_s = 1.731456e6
        rad_vels = np.zeros(len(times))

        for i,t in enumerate(times):
            sim.integrate(t)
            rad_vels[i] = -ps['star'].vz * AU_day_to_m_s
          
        residuals=np.zeros((len(errs),1))
        chi=np.zeros((len(errs),1))
        
        for i,vel_theory in enumerate(rad_vels):
            residuals[i]=(sort_arr[i,1]-vel_theory)/np.sqrt(np.abs(sort_arr[i,2]))
            
        for i,vel_theory in enumerate(rad_vels):
            chi[i] = (sort_arr[i,1]-vel_theory)**2/(sort_arr[i,2]**2+jitter[i]**2)   
        n_bins=40
        #gauss=np.linspace(stats.norm.ppf(0.01),stats.norm.ppf(0.99), len(residuals))
        #hist_data = np.vstack([residuals, gauss]).T
        #plt.hist(gauss, n_bins, alpha=0.7, label='Gaussian')
        plt.hist(residuals,n_bins,label='residuals')
        plt.xlabel('Residuals')
        if save == 1:
            plt.savefig(filename)
        plt.show()
        plt.hist(chi,n_bins,label='chi distribution')
        plt.xlabel('Distribution')
        plt.show()
        
            
        
        Ksm=stats.kstest(residuals,'norm')
        return Ksm
        
        
        pyplot.show()
    def log_like(self,epoch=2450000):


        """Calculate the log likelihood for MCMC"""

        JDs = []
        vels = []
        errs = []

        for i,fname in enumerate(self.RV_data):
            tmp_arr = np.loadtxt(self.path_to_data + fname)
            JDs = np.concatenate((JDs,tmp_arr[:,0]))
            vels = np.concatenate((vels,(tmp_arr[:,1]-self.offsets[i])))
            errs = np.concatenate((errs,tmp_arr[:,2]))

        #There might be a better way to do this -- these commands sort the data by time so that we can integrate
        #up to each time
        sort_arr = [JDs,vels,errs]
        sort_arr = np.transpose(sort_arr)
        sort_arr = sort_arr[np.argsort(sort_arr[:,0])]

        deg2rad = np.pi/180.
        sim = rebound.Simulation()
        sim.units = ('day', 'AU', 'Msun')
        sim.t = epoch
        sim.add(m=self.mstar,hash='star')

        for planet in self.planets:
            sim.add(m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)

        sim.move_to_com()
        ps = sim.particles

        times = sort_arr[:,0] #Times to integrate to are just the times for each data point, no need to integrate
        #between data points

        AU_day_to_m_s = 1.731456e6
        rad_vels = np.zeros(len(times))

        for i,t in enumerate(times):
            sim.integrate(t)
            rad_vels[i] = -ps['star'].vz * AU_day_to_m_s

        return -0.5*np.sum((sort_arr[:,1]-rad_vels)**2/sort_arr[:,2]**2 + np.log(2*np.pi*sort_arr[:,2]**2))

    def rem_planet(self,i=0):
        del self.planets[i]

    def clear_planets(self):
        self.planets = []

    
    def orbit_evolution(self,periods=1e4,pnts_per_period=5,outputs_per_period=1,verbose=0,integrator='whfast',safe=1,
                   timing=0,save_output=0,plot=0,energy_err=0,log=1):


        deg2rad = np.pi/180.
        sim = rebound.Simulation()
        sim.integrator = integrator
        exact = 1
        if integrator != 'ias15':
            exact = 0
        sim.units = ('day', 'AU', 'Msun')
        # sim.t = epoch #Epoch is the starting time of simulation
        sim.add(m=self.mstar,hash='star')

        min_per = np.inf
        max_per = 0

        for planet in self.planets: #Add planets in self.planets to Rebound simulation
            sim.add(m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
            min_per = min(min_per,planet.per) #Minimum period
            max_per = max(max_per,planet.per) #Maximum period

        t_max = max_per*periods
        Noutputs = int(t_max/min_per*outputs_per_period)
        times = np.linspace(0,t_max, Noutputs)

        sim.move_to_com()
        sim.dt = min_per/pnts_per_period
        ps = sim.particles[1:]

        if plot:
            semi_major_arr = np.zeros((len(ps),Noutputs))

        # print Noutputs

        if timing:
            start_time = time.time()

        if energy_err:
            E0 = sim.calculate_energy()

        # if not(safe):
        #     sim.ri_whfast.safe_mode = 0

        a0 = [planet.a for planet in ps]

        stable = 1
        planet_stab = 0

        for i,t in enumerate(times): #Perform integration
            sim.integrate(t,exact_finish_time = exact)
            for k,planet in enumerate(ps):
                if (np.abs((a0[k]-planet.a)/a0[k])>1) or planet.a < 0.1:
                    stable = 0
                    planet_stab = k
                if plot:
                    semi_major_arr[k,i] = planet.a
            if verbose and (i % (Noutputs/10) == 0):
                 # print "%3i %%" %(float(i+1)/float(Noutputs)*100.)
                print "%2i %%" %(100*i/Noutputs)

            if stable == 0:
                break

        if timing:
            print "Integration took %.5f seconds" %(time.time() - start_time)

        if energy_err:
            Ef = sim.calculate_energy()
            print "Energy Error is %.3f%% " %(np.abs((Ef-E0)/E0*100))


        if plot:
            plt.figure(1,figsize=(11,6))

            if log:
                for i in range(len(ps)):
                    plt.semilogx(times/365.25,semi_major_arr[i])

            else:
                for i in range(len(ps)):
                    plt.plot(times/365.25,semi_major_arr[i])

            plt.xlabel("Time [Years]")
            plt.ylabel("a [AU]")

           

        return (time.time() - start_time)
    
    
    
    def orbit_stab(self,periods=1e4,pnts_per_period=5,outputs_per_period=1,verbose=0,integrator='whfast',safe=1,
                   timing=0,save_output=0,plot=0,energy_err=0,save=0,filename=None):


        deg2rad = np.pi/180.
        sim = rebound.Simulation()
        sim.integrator = integrator
        exact = 1
        if integrator != 'ias15':
            exact = 0
        sim.units = ('day', 'AU', 'Msun')
        # sim.t = epoch #Epoch is the starting time of simulation
        sim.add(m=self.mstar,hash='star')

      
        min_per = min(self.planets,key=attrgetter('per')).per
        max_per = max(self.planets,key=attrgetter('per')).per

        for planet in self.planets: #Add planets in self.planets to Rebound simulation
            sim.add(m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
          
      
        
        t_max = max_per*periods
        Noutputs = int(t_max/min_per*outputs_per_period)
        times = np.linspace(0,t_max, Noutputs)

        sim.move_to_com()
        sim.dt = min_per/pnts_per_period
        ps = sim.particles[1:]

        if plot:
            semi_major_arr = np.zeros((len(ps),Noutputs))

        # print Noutputs

        if timing:
            start_time = time.time()

        if energy_err:
            E0 = sim.calculate_energy()

        # if not(safe):
        #     sim.ri_whfast.safe_mode = 0

        a0 = [planet.a for planet in ps]

        stable = 1

        for i,t in enumerate(times): #Perform integration
            sim.integrate(t,exact_finish_time = exact)
            for k,planet in enumerate(ps):
                if (np.abs((a0[k]-planet.a)/a0[k])>10) or planet.a < 0.1:
                    stable = 0
                if plot:
                    semi_major_arr[k,i] = planet.a
            if verbose and (i % (Noutputs/10) == 0):
                print ("%3i %%" %(float(i+1)/float(Noutputs)*100.))
                #print ("%2i %%" %(100*i/Noutputs))

            if stable == 0:
                break

        if timing:
            print ("Integration took %.5f seconds" %(time.time() - start_time))

        if energy_err:
            Ef = sim.calculate_energy()
            print( "Energy Error is %.3f%% " %(np.abs((Ef-E0)/E0*100)))


        if plot:
            plt.figure(1,figsize=(11,6))

            for i in range(len(ps)):
                plt.semilogx(times/365.25,semi_major_arr[i])

            plt.xlabel("Time [Years]")
            plt.ylabel("a [AU]")
            if save == 1:
                plt.savefig(filename)

        return stable

    def RMS_RV(self,epoch=2450000):
        JDs = []
        vels = []
        errs = []

        for i,fname in enumerate(self.RV_data):
            tmp_arr = np.loadtxt(self.path_to_data + fname)
            JDs = np.concatenate((JDs,tmp_arr[:,0]))
            vels = np.concatenate((vels,(tmp_arr[:,1]-self.offsets[i])))
            errs = np.concatenate((errs,tmp_arr[:,2]))

        #There might be a better way to do this -- these commands sort the data by time so that we can integrate
        #up to each time
        sort_arr = [JDs,vels,errs]
        sort_arr = np.transpose(sort_arr)
        sort_arr = sort_arr[np.argsort(sort_arr[:,0])]

        deg2rad = np.pi/180.
        sim = rebound.Simulation()
        sim.units = ('day', 'AU', 'Msun')
        sim.t = epoch
        sim.add(m=self.mstar,hash='star')

        for planet in self.planets:
            sim.add(m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)

        sim.move_to_com()
        ps = sim.particles

        planet_arr = ps[1:]

        a0 = [planet.a for planet in planet_arr]

        times = sort_arr[:,0] #Times to integrate to are just the times for each data point, no need to integrate
        #between data points

        AU_day_to_m_s = 1.731456e6
        rad_vels = np.zeros(len(times))

        for i,t in enumerate(times):
            sim.integrate(t)

            for j,planet in enumerate(planet_arr):
                if np.abs((planet.a - a0[i])/a0[i]) > 0.1:
                    return -np.inf

            rad_vels[i] = -ps['star'].vz * AU_day_to_m_s

        RMS_RV = 0
        for i,vel_theory in enumerate(rad_vels):
                RMS_RV += (sort_arr[i,1]-vel_theory)**2

        return np.sqrt(RMS_RV/len(times) )



    def stab_logprob(self,epoch=2450000,pnts_per_period=10,time=1e4):
        stable = self.orbit_stab(periods=time,pnts_per_period=pnts_per_period,outputs_per_period=1)
        if stable:
            return self.log_like(epoch=epoch)
        else:
            return -np.inf


    def plot_phi(self,p=2.,q=1.,pert_ind=0,test_ind=1,periods=1e2,pnts_per_period=100.,
                outputs_per_period=20.,verbose=0,log_t = 0, integrator='whfast',plot=1):

        deg2rad = np.pi/180.
        sim = rebound.Simulation()
        sim.integrator = integrator
        exact = 1
        if integrator != 'ias15':
            exact = 0
        sim.units = ('day', 'AU', 'Msun')
        # sim.t = epoch #Epoch is the starting time of simulation
        sim.add(m=self.mstar,hash='star')

        min_per = np.inf
        max_per = 0

        res_inds = [pert_ind,test_ind]

        per_max = 0
        per_min = np.inf

        for i,planet in enumerate(self.planets): #Add planets in self.planets to Rebound simulation
            sim.add(m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
            if i in res_inds:
                if planet.per > per_max:
                    outer = i
                    per_max = planet.per
                if planet.per < per_min:
                    inner = i
                    per_min = planet.per


            min_per = min(min_per,planet.per) #Minimum period
            max_per = max(max_per,planet.per)

        t_max = max_per*periods
        Noutputs = int(t_max/min_per*outputs_per_period)
        times = np.linspace(0,t_max, Noutputs)

        sim.move_to_com()
        sim.dt = min_per/pnts_per_period
        ps = sim.particles[1:]

        pert = ps[pert_ind]
        test = ps[test_ind]
        outer = ps[outer]
        inner = ps[inner]

        phi_arr = np.zeros(Noutputs)

        for i,t in enumerate(times): #Perform integration
            sim.integrate(t,exact_finish_time = exact)
            phi_arr[i] = ((p+q)*outer.l - p*inner.l - q*test.pomega)%(2*np.pi)

        angle_fixed = lambda phi: phi-2*np.pi if phi>np.pi else phi
        phi_arr = [angle_fixed(phi) for phi in phi_arr]

        if plot:
            plt.figure(1,figsize=(11,6))

            if log_t:
                plt.semilogx(times/365.25,phi_arr)
            else:
                plt.plot(times/365.25,phi_arr)

            plt.xlabel("Time [Years]")
            plt.ylabel(r"$\phi$ [deg]")

        return times, phi_arr

        # print inner,outer

    def save_params(self,fname):

        param_arr = np.zeros((len(self.planets),7))

        M_J = 9.5458e-4

        for i,planet in enumerate(self.planets):
            arr_tmp = [planet.per, planet.mass/M_J, planet.M, planet.e, planet.pomega, planet.i, planet.Omega]
            param_arr[i] = arr_tmp

        np.savetxt(fname,param_arr)
        np.savetxt(fname + "_offsets",self.offsets)

    def plot_planet_RV(self,epoch=2450000):

        """Make a plot of the RV time series for the star and planets"""

        #Intialize Rebound simulation
        deg2rad = np.pi/180.
        sim = rebound.Simulation()
        sim.units = ('day', 'AU', 'Msun')
        sim.t = epoch #Epoch is the starting time of simulation
        sim.add(m=self.mstar,hash='star')


        min_per = np.inf
        max_per = 0

        for planet in self.planets: #Add planets in self.planets to Rebound simulation
            sim.add(m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
            min_per = min(min_per,planet.per) #Minimum period
            max_per = max(max_per,planet.per)

        sim.move_to_com()
        ps = sim.particles

        Noutputs = 1000
        times = np.linspace(0,10*max_per,Noutputs)

        AU_day_to_m_s = 1.731456e6 #Conversion factor from Rebound units to m/s

        rad_vels = np.zeros((len(sim.particles),Noutputs))


        for i,t in enumerate(times): #Perform integration
            sim.integrate(t)
            rad_vels[0,i] = -ps['star'].vz * AU_day_to_m_s
            for j,plan in enumerate(ps[1:]):
                rad_vels[j+1,i] = plan.vz * AU_day_to_m_s * (self.planets[j].mass/self.mstar)
            # print i

        fig = plt.figure(1,figsize=(11,6)) #Plot RV

        plt.plot(times,rad_vels[0])

        for i in range(len(ps[1:])):
            plt.plot(times,rad_vels[i+1],linestyle='dashed')

        plt.xlabel("Time [JD]")
        plt.ylabel("RV [m/s]")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        plt.save('bigfig.png')

        plt.show()


    def plot_RV_kep(self,epoch=2450000,save=0):

        """Make a plot of the RV time series with data and integrated curve"""

        JDs = []
        vels = []
        errs = []

        for i,fname in enumerate(self.RV_data): #Read in RV data
            tmp_arr = np.loadtxt(self.path_to_data + fname)
            JDs.append(tmp_arr[:,0])
            vels.append(tmp_arr[:,1]-self.offsets[i])
            errs.append(tmp_arr[:,2])

        JDs = np.array(JDs) - epoch

        JD_max = max(np.amax(JDs[i]) for i in range(len(self.RV_data)))
        JD_min = min(np.amin(JDs[i]) for i in range(len(self.RV_data)))

        min_per = np.inf
        for planet in self.planets:
            min_per = min(min_per,planet.per) #Minimum period


        Noutputs = int((JD_max-JD_min)/min_per*100.)

        times = np.linspace(JD_min, JD_max, Noutputs)
        rad_vels = np.zeros(Noutputs)

        deg2rad = np.pi/180.
        t_p_arr = [-planet.M*deg2rad/2./np.pi*planet.per + planet.per for planet in self.planets]

        def kep_sol(t=0,t_p=0,e=0.1,per=100.):
            n = 2*np.pi/per
            M = n*(t-t_p)

            kep = lambda E: M - E + e*np.sin(E)

            E = op.fsolve(kep,M)

            return 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))%(2*np.pi)
        def RV_amp(m_star = 1.0, m_p = 9.5458e-4, omega = 0., i = np.pi/2., per = 365.25, f = 0., e = 0.0):
            omega = omega*deg2rad
            G = 6.67259e-11
            m_sun = 1.988435e30
            JD_sec = 86400.0

            m_star_mks = m_star*m_sun
            m_p_mks = m_p*m_sun

            per_mks = per*JD_sec
            n = 2.*np.pi/per_mks
            a = (G*(m_star_mks + m_p_mks)/n**2.)**(1./3.)

            return np.sqrt(G/(m_star_mks + m_p_mks)/a/(1-e**2.))*(m_p_mks*np.sin(i))*(np.cos(omega+f)+e*np.cos(omega))


        for i,t in enumerate(times):
            rv = 0
            for j,planet in enumerate(self.planets):
                f = kep_sol(t=t,t_p=t_p_arr[j],e=planet.e,per=planet.per)
                rv += RV_amp(m_star = self.mstar, m_p = planet.mass, per = planet.per, f = f, e = planet.e,
                             omega=planet.pomega)
            rad_vels[i] = rv


        fig = plt.figure(1,figsize=(11,6)) #Plot RV

        plt.plot(times + epoch,rad_vels)

        for i in range(len(self.RV_data)):
            plt.errorbar(JDs[i] + epoch,vels[i],yerr = errs[i],fmt='o')

        plt.xlabel("Time [JD]")
        plt.ylabel("RV [m/s]")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        plt.show()

        if save:
            fig.savefig('tst.pdf')
            print "Saved"

            
            
            
        
    
    
    
    def calc_chi2_kep(self,epoch=2450000):

        """Calculate the chi^2 value of the RV time series for the planets currently in the system"""

        JDs = []
        vels = []
        errs = []

        for i,fname in enumerate(self.RV_data):
            tmp_arr = np.loadtxt(self.path_to_data + fname)
            JDs = np.concatenate((JDs,tmp_arr[:,0]))
            vels = np.concatenate((vels,(tmp_arr[:,1]-self.offsets[i])))
            errs = np.concatenate((errs,tmp_arr[:,2]))

        #There might be a better way to do this -- these commands sort the data by time so that we can integrate
        #up to each time
        sort_arr = [JDs,vels,errs]
        sort_arr = np.transpose(sort_arr)
        sort_arr = sort_arr[np.argsort(sort_arr[:,0])]

        times = sort_arr[:,0]-epoch #Times to integrate to are just the times for each data point, no need to integrate
        #between data points
        rad_vels = np.zeros(len(times))

        min_per = np.inf
        for planet in self.planets:
            min_per = min(min_per,planet.per) #Minimum period

        deg2rad = np.pi/180.
        t_p_arr = [-planet.M*deg2rad/2./np.pi*planet.per + planet.per for planet in self.planets]

        def kep_sol(t=0,t_p=0,e=0.1,per=100.): #Solve Kepler's Equation for f eccentric anomaly, true anomaly
            n = 2*np.pi/per
            M = n*(t-t_p)

            kep = lambda E: M - E + e*np.sin(E)

            E = op.fsolve(kep,M)

            return 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))%(2*np.pi)

        def RV_amp(m_star = 1.0, m_p = 9.5458e-4, omega = 0., i = np.pi/2., per = 365.25, f = 0., e = 0.0):
            omega = omega*deg2rad
            G = 6.67259e-11
            m_sun = 1.988435e30
            JD_sec = 86400.0

            m_star_mks = m_star*m_sun
            m_p_mks = m_p*m_sun

            per_mks = per*JD_sec
            n = 2.*np.pi/per_mks
            a = (G*(m_star_mks + m_p_mks)/n**2.)**(1./3.)

            return np.sqrt(G/(m_star_mks + m_p_mks)/a/(1-e**2.))*(m_p_mks*np.sin(i))*(np.cos(omega+f)+e*np.cos(omega))


        for i,t in enumerate(times):
            rv = 0
            for j,planet in enumerate(self.planets):
                f = kep_sol(t=t,t_p=t_p_arr[j],e=planet.e,per=planet.per)
                rv += RV_amp(m_star = self.mstar, m_p = planet.mass, per = planet.per, f = f, e = planet.e,
                             omega=planet.pomega)
            rad_vels[i] = rv

        chi_2 = 0

        for i,vel_theory in enumerate(rad_vels):
                chi_2 += (sort_arr[i,1]-vel_theory)**2/sort_arr[i,2]**2

        return chi_2

    def genetic_search(self,bounds,length,func1,func2=None,func3=None,num_func=1,num_gen=50,crossover=0.90,mutation=0.25,pop_size=400,freq_stat=10,minmax='minimize',cores=0,scaling=None,database=0,file_name=None):
        alleles=GAllele.GAlleles()
        length=length
        def Grid_Constructor(a=bounds):
            alleles = GAllele.GAlleles()
            for i in range(0, length):
                alleles.add(GAllele.GAlleleRange(bounds[i][0], bounds[i][1], real=True))
            return alleles
        genome = G1DList.G1DList(length)
                # now I am initializing my chromosome

        genome.setParams(allele=Grid_Constructor())
#this sets my genome parameters specific to the batched 'data' or parameters that I want to use
        genome.initializator.set(Initializators.G1DListInitializatorAllele)
#this initializes my chromosome
        genome.mutator.set(Mutators.G1DListMutatorAllele)
        genome.crossover.set(Crossovers.G1DListCrossoverSinglePoint)
        if num_func==1:
            genome.evaluator.set(func1)
        elif num_func==2:
            genome.evaluator.set(func1)
            genome.evaluator.set(func2)
        elif num_func==3:
            genome.evaluator.set(func1)
            genome.evaluator.set(func2)
            genome.evaluator.set(func3)
        
#switch statement to initialize amount and type of evaluator functions
        


        ## genome side work here.. ## reference pe 0.6 as well as MLA master 
        ## can change both the crossover and mutator type problem-specific ##
        
        ga = GSimpleGA.GSimpleGA(genome)
        #starts  algorithm engine
        if database == 1:
            sqlite_adapter = DBAdapters.DBSQLite(identify=file_name)
            ga.setDBAdapter(sqlite_adapter)

        ga.setMinimax(Consts.minimaxType[minmax])
#sets type of operation we want the algorithm to work on 

        ga.setGenerations(num_gen)
#number of generations we want to seed

        ga.setCrossoverRate(crossover)
#sets the crossover reproduction rate of genes, manually set
        ga.setMutationRate(mutation)
#sets how frequently the genes should mutate, I am in favor of this being smaller as we pinpoint global maximization 
#beginning simulations should have a large mutation rate however

        ga.setPopulationSize(pop_size)
#this sets our initial population that we want to observe, akin to MCMC walkers.
        ga.setElitism
#this ensures that the best individual in the population reproduces, can be turned off once your 'zoomed' in
#scaling switch here
        stage_prop=ga.getPopulation()
        if scaling == 'Power':
            Scaling.PowerLawScaling(stage_prop)
        
 # only need Power scaling here, since the default scaling is linear. There's no documentation on their weights.           
            
#scaling here is very black-box, I don't like how they set up their weights. 
# I am more inclined to adjust it on the evaluation function side using constants
#multiplied into the score based on adjusted fitness, drawing from a distribution
        if cores == 0:
            ga.setMultiProcessing(False)
        else: 
            ga.setMultiProcessing(True)
#enables multiprocessing when evaluating candidates.

#starts up evolution, returns best candidate
        ga.evolve(freq_stats=freq_stat) 
        return ga.bestIndividual()

    def plot_planet_RV(self,epoch=2450000):

        """Make a plot of the RV time series for the star and planets"""

        #Intialize Rebound simulation
        deg2rad = np.pi/180.
        sim = rebound.Simulation()
        sim.units = ('day', 'AU', 'Msun')
        sim.t = epoch #Epoch is the starting time of simulation
        sim.add(m=self.mstar,hash='star')


        min_per = np.inf
        max_per = 0

        for planet in self.planets: #Add planets in self.planets to Rebound simulation
            sim.add(m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
            min_per = min(min_per,planet.per) #Minimum period
            max_per = max(max_per,planet.per)

        sim.move_to_com()
        ps = sim.particles

        Noutputs = 1000
        times = np.linspace(0,10*max_per,Noutputs)

        AU_day_to_m_s = 1.731456e6 #Conversion factor from Rebound units to m/s

        rad_vels = np.zeros((len(sim.particles),Noutputs))


        for i,t in enumerate(times): #Perform integration
            sim.integrate(t)
            rad_vels[0,i] = -ps['star'].vz * AU_day_to_m_s
            for j,plan in enumerate(ps[1:]):
                rad_vels[j+1,i] = plan.vz * AU_day_to_m_s * (self.planets[j].mass/self.mstar)
            # print i

        fig = plt.figure(1,figsize=(11,6)) #Plot RV

        plt.plot(times,rad_vels[0])

        for i in range(len(ps[1:])):
            plt.plot(times,rad_vels[i+1],linestyle='dashed')

        plt.xlabel("Time [JD]")
        plt.ylabel("RV [m/s]")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        plt.show()

        
        

# def like_wrap(params_opt,params_fixed,RVsys):
#
# #
# # def opt_params(RVsys,planet_num = 0,min_pars="p_m_e",replace_planet=1):
# #     if min_pars == "p_m_e":
# #         guesses = [RVsys.planets[planet_num].per,RVsys.planets[planet_num].mass,RVsys.planets[planet_num].e]
# #     elif min_pars == 'p_m'
# #         guesses = [RVsys.planets[planet_num].per,RVsys.planets[planet_num].mass,RVsys.planets[planet_num].e]