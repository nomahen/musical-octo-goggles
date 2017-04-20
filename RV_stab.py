import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import rebound
from matplotlib.ticker import FormatStrFormatter
import time

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
        self.l = M + pomega


class RVSystem(RVPlanet):

    """Main class for RV Simulations"""
    
    def __init__(self,mstar=1.0):
        
        self.mstar = mstar #Mass of central star
        self.planets = [] #Array containing RVPlanets class
        self.RV_data = [] #Array of RV velocities, assumed to be of the form: JD, RV, error
        self.offsets = [] #Array of constant velocity offsets for each data set
        self.path_to_data = "" #Optional prefix that points to location of datasets
    
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

    def plot_RV(self,epoch=2450000,save=0):

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
        for planet in self.planets: #Add planets in self.planets to Rebound simulation
            sim.add(m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
            min_per = min(min_per,planet.per) #Minimum period, used for plotting purposes
        
        JD_max = max(np.amax(JDs[i]) for i in range(len(self.RV_data)))
        JD_min = min(np.amin(JDs[i]) for i in range(len(self.RV_data)))
        
        
        Noutputs = int((JD_max-JD_min)/min_per*100.)
        
        sim.move_to_com()
        ps = sim.particles

        times = np.linspace(JD_min, JD_max, Noutputs)
        AU_day_to_m_s = 1.731456e6 #Conversion factor from Rebound units to m/s

        rad_vels = np.zeros(Noutputs)

        for i,t in enumerate(times): #Perform integration
            sim.integrate(t)
            rad_vels[i] = -ps['star'].vz * AU_day_to_m_s

        fig = plt.figure(1,figsize=(11,6)) #Plot RV

        plt.plot(times,rad_vels)

        for i in range(len(self.RV_data)):
            plt.errorbar(JDs[i],vels[i],yerr = errs[i],fmt='o')

        plt.xlabel("Time [JD]")
        plt.ylabel("RV [m/s]")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        plt.show()

        if save:
            fig.savefig('tst.pdf')
            print "Saved"

    def calc_chi2(self,epoch=2450000):

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

        chi_2 = 0

        for i,vel_theory in enumerate(rad_vels):
                chi_2 += (sort_arr[i,1]-vel_theory)**2/sort_arr[i,2]**2

        return chi_2

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

    def orbit_stab(self,periods=1e4,pnts_per_period=5,outputs_per_period=1,verbose=0,integrator='whfast',safe=1,
                   timing=0,save_output=0,plot=0,energy_err=0):


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
            max_per = max(max_per,planet.per)

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

            for i in range(len(ps)):
                plt.semilogx(times/365.25,semi_major_arr[i])

            plt.xlabel("Time [Years]")
            plt.ylabel("a [AU]")

            if not(stable):
                print "Planet %i went unstable" %planet_stab

        return stable

    def stab_logprob(self,epoch=2450000):
        stable = self.orbit_stab(periods=1e4,pnts_per_period=10,outputs_per_period=1)
        if stable:
            return self.log_like(epoch=epoch)
        else:
            return -np.inf




# def like_wrap(params_opt,params_fixed,RVsys):
#
# #
# # def opt_params(RVsys,planet_num = 0,min_pars="p_m_e",replace_planet=1):
# #     if min_pars == "p_m_e":
# #         guesses = [RVsys.planets[planet_num].per,RVsys.planets[planet_num].mass,RVsys.planets[planet_num].e]
# #     elif min_pars == 'p_m'
# #         guesses = [RVsys.planets[planet_num].per,RVsys.planets[planet_num].mass,RVsys.planets[planet_num].e]
