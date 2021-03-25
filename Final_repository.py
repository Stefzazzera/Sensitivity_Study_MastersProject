################################################################################################################
# This code stores all the functions and methods used in my Master's Project: 
# A sensitivity study of third-generation gravitational waves detectors using binary black hole encounters. 
# Required packages for the usage of this code: minke, lalsimulation.
################################################################################################################

import numpy as np
import minke.distribution
import minke.sources
import lalsimulation
import lal
import matplotlib.pyplot as plt
from minke import mdctools, distribution, sources, noise

nr_path = 'h_psi4/{}'

'''identify some well-behaved waveforms'''
waveforms_merge = {
    1: "h_m1_L0.9_l2m2_r300.dat",
    2: "h_m2_L0.87_l2m2_r300.dat",
    4: "h_m4_L0.5_l2m2_r300.dat",
    8: "h_m8_L0.35_l2m2_r280.dat",
    16: "h_m16_L0.18_l2m2_r300.dat"
}
waveforms_non_merge = {
    1: "h_m1_L1.5_l2m2_r300.dat",
    2: "h_m2_L1.2_l2m2_r400.dat",
    4: "h_m4_L0.8_l2m2_r400.dat",
    8: "h_m8_L0.45_l2m2_r200.dat",
    16: "h_m16_L0.25_l2m2_r240.dat"
}

sample_rate = 256
epoch = "1126259642.75"
total_mass = 100

'''load in the psd for the different telescopes'''
o1LSD=noise.PSD("o1.txt")
ETLSD=noise.PSD("et_D.txt")
ceLSD=noise.PSD("ce1.txt")

'''Plot the sensitivity curves of the interferometers'''
def plot_interferometers():
    f,ax=plt.subplots(1,1)
    plt.title("Noise curves for different interferometers")
    for i in ("ce1.txt","et_d.txt","o1.txt","aligo.txt"):
        sens = np.loadtxt(i)
        ax.semilogy(sens[:,0],sens[:,1],label=i.split('.')[0])
    ax.set_xlim(1e0,1e4)
    ax.set_xscale("log")
    ax.set_ybound(1e-25,1e-18)
    ax.set_xlabel(r"Frequency, Hz")
    ax.set_ylabel(r"Strain, 1/$\sqrt{Hz}$")
    ax.set_title("Sensitivity curves comparison")
    ax.legend()
    plt.savefig("Sensitivities.png")


def generate_for_detector(source, ifos, sample_rate, epoch, distance, total_mass, ra, dec, psi):
    '''
    Generate an injection for a given waveform.
    Parameters
    ----------
    source : ``minke.Source`` object
       The source which should generate the waveform.
    ifos : list
       A list of interferometer initialisms, e.g. ['L1', 'H1']
    sample_rate : int
       The sample rate in hertz for the generated signal.
    epoch : str
       The epoch (start time) for the injection.
       Note that this should be provided as a string to prevent overflows.
    distance : float
       The distance for the injection, in megaparsecs.
    total_mass : float
       The total mass for the injected signal.
    ra : float
        The right ascension of the source, in radians.
    dec : float
        The declination of the source, in radians.
    psi : float
        The polarisation angle of the source, in radians. '''

    nr_waveform = source.datafile
    data = {}
    data['data'] = {}
    data['times'] = {}
    data['meta'] = {}
    data['epoch'] = epoch
    data['meta']['ra'] = ra
    data['meta']['dec'] = dec
    data['meta']['psi'] = psi
    data['meta']['distance'] = distance
    data['meta']['total_mass'] = total_mass
    data['meta']['sample rate'] = sample_rate
    data['meta']['waveform'] = nr_waveform


    for ifo in ifos:
        det = lalsimulation.DetectorPrefixToLALDetector(ifo)
        hp, hx = source._generate(half=True, epoch=epoch, rate=sample_rate)[:2]
        h_tot = lalsimulation.SimDetectorStrainREAL8TimeSeries(hp, hx, ra, dec, psi, det)
        data['data'][ifo] = h_tot.data.data.tolist()
        data['times'][ifo] = np.linspace(0, len(h_tot.data.data)*h_tot.deltaT, len(h_tot.data.data)).tolist()

    return data

'''create the json files where to store the injections produced'''
import json
def jayz():
    for i, waveform in enumerate(waveforms):
        with open(f"{i}.json", "w") as f:
            json.dump(waveform, f)

'''some functions useful to plot images of the data'''
def graphs():
    f, axes = plt.subplots(len(glob("waveforms2/h*.dat")), 1, figsize=(7,20), sharex=True)
    for i, file in enumerate(glob("waveforms2/h*.dat")):
        ax = axes[i]
        waveform = np.genfromtxt(file)
        ax.plot(waveform[:,0], waveform[:,1],label=file.split('/')[1])
        ax.legend()
    plt.savefig("all h waveforms2 fixed2")

    f2, axes2 = plt.subplots(len(glob("waveforms/psi*.dat")), 1, figsize=(7,20), sharex=True)
    for i, file in enumerate(glob("waveforms/psi*.dat")):
        ax2 = axes2[i]
        waveform = np.genfromtxt(file)
        ax2.plot(waveform[:,0], waveform[:,1])
    plt.savefig("all psi waveforms")

'''create a function to sample a squared distribution of distances'''
def power(size):
    powerlaw.random_state.seed(1)
    a=3 #exponent of power law is a-1
    r=(powerlaw.rvs(a,size=size))*30
    '''plt.figure()
    plt.hist(r,bins=50, density=True, alpha=0.5)
    plt.savefig("hist.png")'''
    return r

'''sample the sky direction, for when we will start to randomise the injections'''
np.random.seed(1)
dec_vals = np.random.uniform(-(np.pi/2), (np.pi/2), 100)
np.random.seed(1)
ra_vals = np.random.uniform(0, 2*np.pi, 100)
np.random.seed(1)
psi_vals = np.random.uniform(0, 2*np.pi, 100)
locations=np.stack((ra_vals,dec_vals,psi_vals))

##############################
''' SNR CALCULATION'''
##############################
'''now initiate the functions required to calculate and compare the SNRs values'''
from scipy.interpolate import interp1d
from scipy import signal
from scipy import fftpack
from scipy.fftpack import fft,fftshift
import scipy.integrate as integrate
from glob import glob
'''this function will take in the Fourier transform of the data, the values of the psd and the frequency step
and will calculate the SNR of the waveform using the standard formula - the integral is performed as discrete sum'''
def calc_snr(data,psd,df,sr,N):
    sum=0
    for i in range(0,np.size(data)):
        v=(((abs(data[i])**2)/abs(psd[i]**2)))/(N*sr)
        sum=sum+v
    return np.sqrt(sum)

'''function used to compare the SNR calculated by Minke and that calculated with my algorithm for the injections'''
def compareSNR():
    i="h_psi4/"+waveforms_non_merge[1]
    '''next three lines identify the source and calculate the SNR with minke'''
    sourcex=minke.sources.Hyperbolic(datafile=i,total_mass=100,distance=1,time=1126259642.75,ra=4.83,dec=-0.02,psi=3.51); #1126630000
    sourcex.datafile=i
    CEsnr = cePSD.SNR(sourcex, ['H1']) #use H1 for CE and V1 for ET

    '''now generate the injection, and extract the data, sample rate and times'''
    wave = generate_for_detector(sourcex, ['H1'], sample_rate, epoch, distance=1, total_mass=100, ra=4.83,dec=-0.02,psi=3.51);
    data=wave['data']['H1']
    sr=wave['meta']['sample rate']
    t=wave['times']['H1']

    '''find the frequencies by using a freq fourier transform, and find the fourier transform of the data'''
    freq=np.fft.rfftfreq(np.size(t),1/sr)
    print("frequencies are",freq) #1-127
    Fourier_wave=np.fft.fft(data)
    print(Fourier_wave)
    Fourier_squared=Fourier_wave**2
    plt.plot(t,data)
    plt.savefig("time signal waveform 4.png")

    '''load in the PSD of the telescope we want to use and interpolate it'''
    sens = np.loadtxt("ce1.txt")
    ysens=sens[:,1]
    xsens=sens[:,0]
    sensdf=xsens[1]-xsens[0]
    f = interp1d(xsens, ysens,bounds_error=False, fill_value='extrapolate')

    '''need to make sure that the frequencies used are above 5Hz, which corresponds to above 5th number in list'''
    freq_corrected=[]
    for i in freq:
        if i>5:
            freq_corrected.append(i)
    S=[f(i) for i in freq_corrected]

    '''calculate the PSD Sh as the modulus of the fourier transform of the noise curve'''

    df=freq[1]-freq[0] #find the frequency step

    '''array of fourier transformed data has shape 180, PSD has shape (given by frequencies) of 91'''
    Fourier_corr=Fourier_wave[0:np.size(S)]

    '''calculate ENBW'''
    data_values=np.array(data)
    N=len(freq)

    print("df is",df)
    print(np.shape(Fourier_corr))
    print(np.shape(S))
    My_Snr=calc_snr(Fourier_corr,S,df,sr,N)#_corr
    print("mine is", My_Snr)
    print("Minke is", CEsnr[1])


'''Go through the distances and locations and find the SNR in each case'''
distances_NON_mergers=range(1,80,2)
distances_mergers=range(1,220,2)

'''Calculate the fraction of detections for a given set of waveforms (mergers or non-mergers)
   and plot it against distance with the subsequent function'''
def snr_fraction(set,i,distances,mass=100):
    file="waveforms/"+set[i]
    fraction=[]
    for distance in distances:
        snr_list_seen=0
        snr_list_tot=0
        for location in np.vstack(locations).T:
            sourcex=[]
            sourcex=minke.sources.Hyperbolic(datafile=file,total_mass=mass,distance=distance,time=1126259642.75,ra=location[0],dec=location[1],psi=location[2]); #1126630000
            sourcex.datafile=file
            cesnr = ceLSD.SNR(sourcex, ['V1'])
            print(cesnr[1][0])
            snr_list_tot+=1
            if float(cesnr[1][0]) > 8:
                snr_list_seen+=1
        fraction.append(snr_list_seen/snr_list_tot)
        print("fraction is",snr_list_seen/snr_list_tot)

    return fraction

def plot_snr_fraction(set,distances):
    fig, ax = plt.subplots()
    plt.xlabel("Distance, Mpc")
    plt.ylabel("Fraction of good snr")
    if set == waveforms_merge:
        plt.title("Fraction of snrs above 8 for all Mergers CE")
        for i in (1,2,4,8,16):
            fraction=snr_fraction(set,i,distances)
            ax.plot(distances,fraction,label='Waveform['+str(i)+']')
        ax.legend()
        plt.savefig("Selected fractions of mergers for CE.png")#"fraction of snrs above 8 per distance - NM"+str(i)+".png"
    else:
        plt.title("Fraction of snrs above 8 for all Non-Mergers CE")
        for i in (1,2,4,8,16):
            fraction=snr_fraction(set,i,distances)
            ax.plot(distances,fraction,label='Waveform['+str(i)+']')
        ax.legend()
        plt.savefig("Selected fractions of non-mergers for CE.png")#"fraction of snrs above 8 per distance - NM"+str(i)+".png"




'''Create a function that would window the non-merger waveforms in order to eliminate the offset at the tail'''
def fixdata():
    for file in glob("h_psi4/*.dat"):
        '''load in the data of a waveform'''
        file_data = np.loadtxt(file)
        t=file_data[:,0]
        y1=file_data[:,1]
        y2=file_data[:,2]
        dt=t[1]-t[0]
        t=np.append(t,t[-1]+(dt*np.array(range(1,151))))
        y1=np.append(y1, np.zeros(150))
        y2=np.append(y2, np.zeros(150))
        '''find the mean of the data, looking only at array points that are different than zero'''
        sum=0
        len=0
        for x in y1:
            if x != 0:
                sum=sum+x
                len+=1

        m=sum/len
        data_meaned=y1-m*np.ones(np.size(y1))

        '''find index where data crosses zero by looking at the change of sign -- +1 is necessary
        if we want to find the value after the crossing'''
        zero_crossings = (np.where(np.diff(np.sign(data_meaned)))[0])
        print("zeros are",zero_crossings)
        '''Apply the tukey window: identify if the file is of a merger or non merger by the amount of times the
        data crosses the mean (more than 7 and it will definitely be a merger, and won't require windowing, as
        there is no offset); one waveform has only 2 crossings, so needed to identify the particular case;
        the window is produced and applied to the data'''
        if np.size(zero_crossings) > 3 and np.size(zero_crossings) < 11:
            window=signal.tukey(np.size(t[:zero_crossings[2]]),alpha=0.05)
            while np.size(window) < np.size(t):
                window=np.append(window,0)
            y1_windowed=y1*window
            y2_windowed=y2*window
            print("used window 3-7")
        elif np.size(zero_crossings) < 3:
            window=signal.tukey(np.size(t[:zero_crossings[1]]),alpha=0.05)
            while np.size(window) < np.size(t):
                window=np.append(window,0)
            y1_windowed=y1*window
            y2_windowed=y2*window
            print("used window -3")
        else:
            y1_windowed=y1
            y2_windowed=y2
            print("not windowed")
        new_file_data = np.stack((t, y1_windowed, y2_windowed), axis=1)
        np.savetxt("waveforms/"+file.split('/')[1], new_file_data)
    return


'''define a quick command to produce the first SNR investigation for all populations'''
def do():
    plot_snr_fraction(waveforms_merge,distances_mergers)
    plot_snr_fraction(waveforms_non_merge,distances_NON_mergers)


'''Define a function that would go through all the waveforms and sort them into merger and non-mergers'''
def sort_populations(mergers,non_mergers):
    for file in glob("waveforms/h*.dat"):
        file_data = np.loadtxt(file)
        t=file_data[:,0]
        y1=file_data[:,1]
        y2=file_data[:,2]
        dt=t[1]-t[0]
        t=np.append(t,t[-1]+(dt*np.array(range(1,151))))
        y1=np.append(y1, np.zeros(150))
        y2=np.append(y2, np.zeros(150))
        '''find the mean of the data, looking only at array points that are different than zero'''
        sum=0
        len=0
        for x in y1:
            if x != 0:
                sum=sum+x
                len+=1
        m=sum/len
        data_meaned=y1-m*np.ones(np.size(y1))
        '''find index where data crosses zero by looking at the change of sign -- +1 is necessary
        if we want to find the value after the crossing'''
        zero_crossings = (np.where(np.diff(np.sign(data_meaned)))[0])
        print("zeros are",zero_crossings)
        '''Apply the tukey window: identify if the file is of a merger or non merger by the amount of times the
        data crosses the mean (more than 7 and it will definitely be a merger, and won't require windowing, as
        there is no offset); one waveform has only 2 crossings, so needed to identify the particular case;
        the window is produced and applied to the data'''
        if np.size(zero_crossings)  < 11:
            print("non_merger detected")
            non_mergers.append(file)
        else:
            print("merger detected")
            mergers.append(file)
    return mergers,non_mergers

#########################################
'''FIRST SNR INVESTIGATION'''
#########################################
'''Define a series of functions to for the first SNR investigation: using data coming from the tables produced earlier plot the
   fraction of detections of a population of BBH'''
def do_mergers(mergers):
    fig,ax =plt.subplots()
    plt.xlabel(r"Distance, Mpc")
    plt.ylabel(r"Fraction of Detections")
    plt.title("Fraction of snrs above 8 for all mergers")
    fraction_ce=snr_fraction_all_population(distances_mergers,mergers,"CE")
    fraction_et=snr_fraction_all_population(distances_mergers,mergers,"ET")
    fraction_ligo=snr_fraction_all_population(distances_mergers,mergers,"ligo")
    table_fractions_MM=np.stack((fraction_ce,fraction_et,fraction_ligo),axis=1)
    np.savetxt("TableFractionsMM.txt",table_fractions_MM)
    ax.plot(distances_mergers,fraction_ce,'r',label="Detections from CE")#,label='Merger'+merger
    ax.plot(distances_mergers,fraction_et,'b',label="Detections from ET")
    ax.plot(distances_mergers,fraction_ligo,'g',label="Detections from ALigo")
    ax.grid('both')
    ax.legend()
    plt.savefig("SNR INVESTIGATION 1 MM 220.png")

def do_non_mergers(non_mergers):
    fig,ax =plt.subplots()
    plt.xlabel(r"Distance, Mpc")
    plt.ylabel(r"Fraction of Detections")
    plt.title("Fraction of snrs above 8 for all non-mergers")
    fraction_ce=snr_fraction_all_population(distances_NON_mergers,non_mergers,"CE")
    fraction_et=snr_fraction_all_population(distances_NON_mergers,non_mergers,"ET")
    fraction_ligo=snr_fraction_all_population(distances_NON_mergers,non_mergers,"ligo")
    table_fractions_NM=np.stack((fraction_ce,fraction_et,fraction_ligo),axis=1)
    np.savetxt("TableFractionsNM.txt",table_fractions_NM)
    ax.plot(distances_NON_mergers,fraction_ce,'r',label="Detections from CE")#,label='Merger'+merger
    ax.plot(distances_NON_mergers,fraction_et,'b',label="Detections from ET")
    ax.plot(distances_NON_mergers,fraction_ligo,'g',label="Detections from LIGO o1")
    ax.grid('both')
    ax.legend()
    plt.savefig("SNR INVESTIGATION 1 NM.png")

def trial():
    fractionsNM=np.loadtxt("TableFractionsNM.txt")
    fractionsMM=np.loadtxt("TableFractionsMM.txt")

    fig1,ax1 =plt.subplots(dpi=300)
    ax1.set_xlabel(r"Distance, Mpc")
    ax1.set_ylabel(r"Fraction of Detections")
    ax1.set_title("Fraction of snrs above 8 for all Non-mergers")
    ax1.plot(distances_NON_mergers,fractionsNM[:,0],'r',label="Detections from CE")
    ax1.plot(distances_NON_mergers,fractionsNM[:,1],'b',label="Detections from ET")
    ax1.plot(distances_NON_mergers,fractionsNM[:,2],'g',label="Detections from LIGO o1")
    ax1.grid('both')
    ax1.set_xscale('log')
    ax1.legend()
    fig1.savefig("SNR INVESTIGATION NM LOOOG CORRECT.png")

    fig2,ax2 =plt.subplots(dpi=300)
    ax2.set_xlabel(r"Distance, Mpc")
    ax2.set_ylabel(r"Fraction of Detections")
    ax2.set_title("Fraction of snrs above 8 for all Mergers")
    ax2.plot(distances_mergers,fractionsMM[:,0],'r',label="Detections from CE")
    ax2.plot(distances_mergers,fractionsMM[:,1],'b',label="Detections from ET")
    ax2.plot(distances_mergers,fractionsMM[:,2],'g',label="Detections from LIGO o1")
    ax2.grid('both')
    ax2.set_xscale('log')
    ax2.legend()
    fig2.savefig("SNR INVESTIGATION MM LOOOG CORRECT.png")

def all_populations():
    mergers=[]
    non_mergers=[]
    mergers, non_mergers = sort_populations(mergers,non_mergers)
    do_mergers(mergers)
    do_non_mergers(non_mergers)
    trial()

'''This function extracts the SNRs of waveforms of given set from a given detector'''
def snr_fraction_all_population(distances,set,detector):
    fraction=[]
    if detector == "CE":
        for distance in distances:
            snr_list_seen=0
            snr_list_tot=0
            waveform_number=0
            for file in set:
                for location in np.vstack(locations).T:
                    sourcex=[]
                    sourcex=minke.sources.Hyperbolic(datafile=file,total_mass=100,distance=distance,time=1126259642.75,ra=location[0],dec=location[1],psi=location[2]); #1126630000
                    sourcex.datafile=file
                    cesnr = ceLSD.SNR(sourcex, ['H1'])
                    print(cesnr[1][0])
                    snr_list_tot+=1
                    waveform_number+=1
                    if float(cesnr[1][0]) > 8:
                        snr_list_seen+=1
            fraction.append(snr_list_seen/snr_list_tot)
            print("fraction is",snr_list_seen/snr_list_tot)
    elif detector == "ET":
        for distance in distances:
            snr_list_seen=0
            snr_list_tot=0
            waveform_number=0
            for file in set:
                for location in np.vstack(locations).T:
                    sourcex=[]
                    sourcex=minke.sources.Hyperbolic(datafile=file,total_mass=100,distance=distance,time=1126259642.75,ra=location[0],dec=location[1],psi=location[2]); #1126630000
                    sourcex.datafile=file
                    etsnr = ETLSD.SNR(sourcex, ['V1'])
                    print(etsnr[1][0])
                    snr_list_tot+=1
                    waveform_number+=1
                    if float(etsnr[1][0]) > 8:
                        snr_list_seen+=1
            fraction.append(snr_list_seen/snr_list_tot)
            print("fraction is",snr_list_seen/snr_list_tot)
    elif detector == "ligo":
        for distance in distances:
            snr_list_seen=0
            snr_list_tot=0
            waveform_number=0
            for file in set:
                for location in np.vstack(locations).T:
                    sourcex=[]
                    sourcex=minke.sources.Hyperbolic(datafile=file,total_mass=100,distance=distance,time=1126259642.75,ra=location[0],dec=location[1],psi=location[2]); #1126630000
                    sourcex.datafile=file
                    o1snr, o1combo = o1LSD.SNR(sourcex, ['V1','L1', 'H1'])
                    print(o1snr)
                    snr_list_tot+=1
                    waveform_number+=1
                    if float(o1snr) > 8:
                        snr_list_seen+=1
            fraction.append(snr_list_seen/snr_list_tot)
            print("fraction is",snr_list_seen/snr_list_tot)
    else:
        print("wrong detector for now")

    return fraction

############################################################
############################################################
'''SECOND SNR INVESTIGATION'''
############################################################
############################################################

import gravpy
import gravpy.sources as sources
import gravpy.interferometers as detectors
from astropy import units as u
import minke.sources
from scipy.signal import windows
import scipy.interpolate as interp
from matplotlib import rc, font_manager
lato = {'family': 'Lato',
        'color':  'black',
        'weight': 'light',
        'size': 10,
        }
source_code_pro = {'family': 'Source Code Pro',
        'weight': 'normal',
        'size': 6,
        }
ticks_font = font_manager.FontProperties(**source_code_pro)
import matplotlib.patheffects as path_effects


'''Define a function used to make SNR contour plots. This will take a waveform and a set of distances and loop through a series of masses
   (and distances and sky locations) and find the SNRs in each case'''

def make_snr_contours(i,distances,snr_specifics=[8]):
    file=i
    snrs=[]
    masses=np.linspace(5,100,40)
    for mass in masses:
        for distance in distances:
            print("Doing distance"+str(distance)+"for mass"+str(mass))
            average_snr=0
            for location in np.vstack(locations).T:
                sourcex=[]
                sourcex=minke.sources.Hyperbolic(datafile=file,total_mass=mass,distance=distance,time=1126259642.75,ra=location[0],dec=location[1],psi=location[2]); #1126630000
                sourcex.datafile=file
                etsnr = ETLSD.SNR(sourcex, ['H1'])
                average_snr=average_snr+etsnr[1][0]
            print(average_snr/len(np.vstack(locations).T))
            snrs.append(average_snr/len(np.vstack(locations).T))

    f, ax = plt.subplots(1,1,dpi=300, figsize=(6,6/1.616))
    snrs = np.array(snrs).reshape(len(masses),len(distances)) #
    contours = ax.contour(distances,masses, snrs, origin="lower", levels=np.arange(0,40,2)[np.arange(0,40,2) != 8],linewidths=0.5,colors="deepskyblue") #, levels=np.arange(0,40,2)[np.arange(0,40,2) != 8]
    contours = ax.contour(distances, masses, snrs, levels=snr_specifics, origin="lower", colors="lightcoral", linewidths=2)#cmap='Greys' cmap="YlGnBu"
    c_labels = ax.clabel(contours, fmt="%.0f")
    ax.grid(which="both", color='#348ABD', alpha=0.4, lw=0.3,)

    for label in c_labels:
        label.set_font_properties(ticks_font)
        label.set_path_effects([path_effects.Stroke(linewidth=1, foreground='white'),path_effects.Normal()])

    ax.set_xlabel("Distance [Mpc]")
    ax.set_ylabel(r"Total Mass [$\mathrm{M}_\odot$]")

    for label in ax.get_xticklabels():
        label.set_fontproperties(ticks_font)

    for label in ax.get_yticklabels():
        label.set_fontproperties(ticks_font)

    f.tight_layout()
    return f

'''Define a quick function to produce SNR plots for all populations'''
def mass_plot(waveform,distances,set):
    f = make_snr_contours("waveforms/"+waveform,distances=distances, snr_specifics=[5,8])
    if set == "waveforms_non_merge":
        f.savefig("snr plots/ET SNR contour NON merge "+waveform+".png")
    else:
        f.savefig("snr plots/ET SNR contour merge "+waveform+".png")

def all_mass_plot():
    for i in (1,2,4,8,16):
        mass_plot(waveforms_non_merge[i], distances_NON_mergers,"waveforms_non_merge")
        mass_plot(waveforms_merge[i], distances_mergers,"waveforms_merge")


#####################################
#####################################
'''THIRD SNR INVESTIGATION'''
#####################################
#####################################

'''Using the Probability distribution function for the mass distribution of Binary Black Hole masses, create the
   associated CDF and a function to sample the distribution'''

from scipy.stats import powerlaw
mass_range=np.linspace(5,45,40)
alpha=-0.4
normalisation=sum(mass_range**alpha)
normalise=1/normalisation
mass_pdf=(mass_range**alpha)/normalisation

def draw_mass(N):
    draws=[]
    np.random.seed(1)
    randoms=np.random.uniform(0,1,N)
    for rand in randoms:
        draw=(normalisation*rand)**(1/alpha)
        draws.append(draw)
    return draws

def random_mass_array(N):
    m_1 = []
    rand_num = np.random.uniform(0,1,N)
    for u in rand_num:
        m =((3/(5*normalise))*u+5**(3/5))**(5/3)  # obtained by solving
        m_1.append(m)                               # CDF
    return m_1

'''Calculate the total mass given m1 and q (note that q is the ratio of the BBH masses)'''
def total_mass_from_m1_q(m1, q):
    return m1 + m1/float(q)

'''Now create a table: sample a number of masses, and for each mass analyse every merger waveform (i.e. every mass ratio avaliable)
   along the set of distances below and using 100 different sky locations. Note: the function "mass_merger" will require long
   computational times (50 masses will be equal to approximately 30 hours!). Hence, the function saves the results in a text file'''
distances_long = np.logspace(2, 5, 50)

def mass_merger():
    mass_1 = random_mass_array(50)
    q = [1,2,4,8,16]
    mq =  np.transpose([np.tile(mass_1, len(q)), np.repeat(q, len(mass_1))])
    masses_total = map(total_mass_from_m1_q, mq[:,0], mq[:,1])
    data = np.vstack([mq.T, np.array(list(masses_total))]).T

    results = []
    data_dist = np.array([np.hstack([row, distance]) for row in data for distance in distances_long])
    print("entering the first loop")
    for x in data_dist:
        print("Doing waveform",x[1])
        print("Doing distance",x[3])
        average_snr_ce=0
        average_snr_et=0
        average_snr_o1=0
        q=x[1]
        for location in np.vstack(locations).T:
            sourcex=[]
            sourcex=minke.sources.Hyperbolic(datafile="waveforms/"+waveforms_non_merge[q],total_mass=x[2],distance=(x[3]/1e3),time=1126259642.75,ra=location[0],dec=location[1],psi=location[2]); #1126630000
            sourcex.datafile="waveforms/"+waveforms_non_merge[q]
            cesnr = ceLSD.SNR(sourcex, ['H1'])
            etsnr = ETLSD.SNR(sourcex, ['V1'])
            o1snr, o1combo = o1LSD.SNR(sourcex, ['V1','L1', 'H1'])
            average_snr_ce=average_snr_ce+cesnr[1][0]
            average_snr_et=average_snr_et+etsnr[1][0]
            average_snr_o1=average_snr_o1+o1snr
        results.append([average_snr_o1/len(np.vstack(locations).T), average_snr_ce/len(np.vstack(locations).T), average_snr_et/len(np.vstack(locations).T)])
    results_table = np.hstack([data_dist, results])
    np.savetxt("ResultsTable3.txt",results_table)
    return results_table


    '''Plot the resulting fraction of detections'''
def plot_results_table():
    results_table=np.loadtxt("ResultsTable3.txt")
    efficiencies_et = []
    efficiencies_ligo = []
    efficiencies_ce = []
    print("now doing distance loop")

    for distance in distances_long:

        efficiency_ligo = np.count_nonzero(results_table[results_table[:,3]==distance, -3]>8)/float(np.count_nonzero(results_table[:,3]==distance))
        efficiencies_ligo.append(efficiency_ligo)
        print(efficiency_ligo)

        efficiency_ce = np.count_nonzero(results_table[results_table[:,3]==distance, -2]>8)/float(np.count_nonzero(results_table[:,3]==distance))
        efficiencies_ce.append(efficiency_ce)

        efficiency_et = np.count_nonzero(results_table[results_table[:,3]==distance, -1]>8)/float(np.count_nonzero(results_table[:,3]==distance))
        efficiencies_et.append(efficiency_et)

    f, ax = plt.subplots(1,1, dpi=300)
    rc("mathtext", fontset="custom", sf="Source Code Pro", tt="Source Code Pro", rm="Source Code Pro")
    ax.semilogx(distances_long/1e3, efficiencies_ligo, label="LIGO o1")
    ax.semilogx(distances_long/1e3, efficiencies_ce, label="CE")
    ax.semilogx(distances_long/1e3, efficiencies_et, label="ET")
    ax.set_xlabel("Distance, Mpc")
    ax.set_ylabel(r"Detection efficiency (SNR$>8$)")
    ax.set_xlim([1e0, 1e2])
    ax.set_ylim([0,1])
    ax.grid('both')
    ax.legend()
    plt.savefig("snr plots/Detection efficiencies.png")
    print("FINISHED")
    return

'''This function will produce an estimation of the horizon distance of the Einstein Telescope and of the Cosmic Explorer.
   It uses the data from the previous table, although it extracts from such data all lines that represent events with SNR=8. In particular, it extracts
   one event for each mass, regardless of the mass ratio/waveform'''
def plot_redshift():
    results_table=np.loadtxt("ResultsTable2.txt")
    snrS_et = []
    snrS_ligo = []
    snrS_ce = []
    print("now doing distance loop")

    snr_ligo = results_table[:, -3]
    snr_ce = results_table[:, -2]
    snr_et = results_table[:, -1]
    snrsL=[]
    snrsC=[]
    snrsE=[]
    diffs=[]
    for x in range(0,250):
        array=np.array(snr_ligo[(0+100*x):(100+100*x)])
        for i in array:
            diffs.append(abs(i-8))

        differences=np.array(diffs)

        closest=results_table[100*x+differences.argmin(),:]
        snrsL.append(closest)
        diffs=[]

    for x in range(0,250):

        array=np.array(snr_ce[(0+100*x):(100+100*x)])
        for i in array:
            diffs.append(abs(i-8))
        differences=np.array(diffs)
        closest=results_table[100*x+differences.argmin(),:]
        snrsC.append(closest)
        diffs=[]

    for x in range(0,250):

        array=np.array(snr_et[(0+100*x):(100+100*x)])
        for i in array:
            diffs.append(abs(i-8))
        differences=np.array(diffs)
        closest=results_table[100*x+differences.argmin(),:]
        snrsE.append(closest)
        diffs=[]

    np.savetxt("SNRtableLigo.txt",snrsL)
    np.savetxt("SNRtableCE.txt",snrsC)
    np.savetxt("SNRtableET.txt",snrsE)

    snrsL=np.array(snrsL)
    snrsC=np.array(snrsC)
    snrsE=np.array(snrsE)

    snrsL_new=[]
    x=0
    for i in snrsL[:,0]:
        z=0
        match = False
        for y in snrsL[:,0]:
            if float(i) == float(y):
                print("FOUND ONE")
                matchvalue=z
                match = True
            z+=1
        if match == False:
            snrsL_new.append(snrsL[x,:])
        else:
            if snrsL[x,3] > snrsL[matchvalue,3]:
                snrsL_new.append(snrsL[x,:])
            elif snrsL[matchvalue,3] > snrsL[x,3]:
                snrsL_new.append(snrsL[matchvalue,:])
        x+=1
    snrsL_new=np.array(snrsL_new)
    print(np.size(snrsL_new),np.size(snrsL))

    snrsC_new=[]
    x=0
    for i in snrsC[:,0]:
        z=0
        match = False
        for y in snrsC[:,0]:
            if float(i) == float(y):
                print("FOUND ONE")
                matchvalue=z
                match = True
            z+=1
        if match == False:
            snrsC_new.append(snrsC[x,:])
        else:
            if snrsC[x,3] > snrsC[matchvalue,3]:
                snrsC_new.append(snrsC[x,:])
            elif snrsC[matchvalue,3] > snrsC[x,3]:
                snrsC_new.append(snrsC[matchvalue,:])
        x+=1
    snrsC_new=np.array(snrsC_new)
    print(np.size(snrsC_new),np.size(snrsC))

    snrsE_new=[]
    x=0
    for i in snrsE[:,0]:
        z=0
        match = False
        for y in snrsE[:,0]:
            if float(i) == float(y):
                print("FOUND ONE")
                matchvalue=z
                match = True
            z+=1
        '''else:
            snrsL_new.append(snrsL[x,:])'''
        if match == False:
            snrsE_new.append(snrsE[x,:])
        else:
            if snrsE[x,3] > snrsE[matchvalue,3]:
                snrsE_new.append(snrsE[x,:])
            elif snrsE[matchvalue,3] > snrsE[x,3]:
                snrsE_new.append(snrsE[matchvalue,:])
        x+=1
    snrsE_new=np.array(snrsE_new)
    print(np.size(snrsE_new),np.size(snrsE))

    f, ax = plt.subplots(1,1, dpi=300)
    plt.title("Horizon distance comparison")
    rc("mathtext", fontset="custom", sf="Source Code Pro", tt="Source Code Pro", rm="Source Code Pro")
    ax.scatter(snrsC_new[:,2],snrsC_new[:,3]/1e3,color='r', label="CE")
    ax.scatter(snrsE_new[:,2], snrsE_new[:,3]/1e3,color='b', label="ET")
    ax.set_xlabel("Total Mass, [$\mathrm{M}_\odot$]")
    ax.set_ylabel(r"Distance, Mpc")
    ax.set_yscale("log")
    ax.set_ylim([1e-1,5e2])
    ax.grid('both')
    ax.legend()
    plt.savefig("snr plots/Distance horizon NONNN LIGO.png")
    print("FINISHED")
    return







###########################################################################
'''The following functions are miscellaneous and were used for the report'''
###########################################################################

'''Compare the Power Spectral Densities of non-merger waveforms and merger ones (at different values of masses) with
   the PSDs of the Einstein Telescope and Cosmic Explorer'''

def interesting_comparisons():
    waveform_path="waveforms/"+waveforms_non_merge[8]
    fig, ax = plt.subplots(2)
    ax[0].set_ylabel(r"Strain, 1/$\sqrt{Hz}$")
    ax[0].set_title("Mergers PSD and detectors PSD")

    sourcex1=minke.sources.Hyperbolic(datafile=waveform_path,total_mass=40,distance=100,time=1126259642.75,ra=4.83,dec=-0.02,psi=3.51); #1126630000
    sourcex1.datafile=waveform_path
    wave1 = generate_for_detector(sourcex1, ['H1'], sample_rate, epoch, distance=100, total_mass=40,ra=4.83,dec=-0.02,psi=3.51); #ra=4.83,dec=-0.02,psi=3.51
    data1=wave1['data']['H1']
    t1=wave1['times']['H1']
    sr1=wave1['meta']['sample rate']
    freq1=np.fft.rfftfreq(np.size(t1),1/sr1)
    Fourier_wave1=(abs((np.fft.fft(data1))))
    ax[0].semilogy(freq1,Fourier_wave1[:np.size(freq1)],'b',label="Merger 40 [$\mathrm{M}_\odot$]")

    sourcex2=minke.sources.Hyperbolic(datafile=waveform_path,total_mass=100,distance=100,time=1126259642.75,ra=4.83,dec=-0.02,psi=3.51); #1126630000
    sourcex2.datafile=waveform_path
    wave2 = generate_for_detector(sourcex2, ['H1'], sample_rate, epoch, distance=100, total_mass=100,ra=4.83,dec=-0.02,psi=3.51); #ra=4.83,dec=-0.02,psi=3.51
    data2=wave2['data']['H1']
    t2=wave2['times']['H1']
    sr2=wave2['meta']['sample rate']
    freq2=np.fft.rfftfreq(np.size(t2),1/sr2)
    Fourier_wave2=(abs((np.fft.fft(data2))))
    ax[0].semilogy(freq2,Fourier_wave2[:np.size(freq2)],'r',label="Merger 100 [$\mathrm{M}_\odot$]")

    ax[0].set_xlim([5, 120])
    ax[0].set_ylim([1e-25,1e-22])

    sens = np.loadtxt("ce1.txt")
    ysens=sens[:,1]
    xsens=sens[:,0]
    print(ysens)
    sensdf=xsens[1]-xsens[0]
    N=len(xsens)

    sens2 = np.loadtxt("et_d.txt")
    ysens2=sens2[:,1]
    xsens2=sens2[:,0]
    sensdf2=xsens2[1]-xsens2[0]
    N2=len(xsens2)


    ax[1].semilogy(xsens,abs(ysens)*(N*sensdf),'g',label="PSD for CE")
    ax[1].semilogy(xsens2,abs(ysens2)*(N2*sensdf2),'m',label="PSD for ET")
    ax[1].set_xlim([5,120])
    ax[1].set_ylim([1e-25,1e-22])
    ax[1].set_xlabel(r"Frequency, Hz")
    ax[1].set_ylabel(r"Strain, 1/$\sqrt{Hz}$")
    ax[0].legend()
    ax[1].legend()
    fig.tight_layout()
    plt.savefig("PSD and noise curve.png")


'''Compare a single waveform prior to the windowing process and the same waveform after. The window is overplotted too'''

def compare_windows():
    unwindowed="h_psi4/"+waveforms_non_merge[2]
    windowed="waveforms/"+waveforms_non_merge[2]

    file_data = np.loadtxt(unwindowed)
    t=file_data[:,0]
    y1=file_data[:,1]
    y2=file_data[:,2]
    dt=t[1]-t[0]
    t=np.append(t,t[-1]+(dt*np.array(range(1,151))))
    y10=np.append(y1, np.zeros(150))
    y20=np.append(y2, np.zeros(150))
    '''find the mean of the data, looking only at array points that are different than zero'''
    sum=0
    len=0
    for x in y10:
        if x != 0:
            sum=sum+x
            len+=1

    m=sum/len
    data_meaned=y10-m*np.ones(np.size(y10))

    '''find index where data crosses zero by looking at the change of sign -- +1 is necessary
    if we want to find the value after the crossing'''
    zero_crossings = (np.where(np.diff(np.sign(data_meaned)))[0])
    print("zeros are",zero_crossings)
    '''Apply the tukey window: identify if the file is of a merger or non merger by the amount of times the
    data crosses the mean (more than 7 and it will definitely be a merger, and won't require windowing, as
    there is no offset); one waveform has only 2 crossings, so needed to identify the particular case;
    the window is produced and applied to the data'''
    if np.size(zero_crossings) > 3 and np.size(zero_crossings) < 11:
        window=signal.tukey(np.size(t[:zero_crossings[2]]),alpha=0.05)
        while np.size(window) < np.size(t):
            window=np.append(window,0)
        y1_windowed=y10*window
        y2_windowed=y20*window
        print("used window 3-7")
    elif np.size(zero_crossings) < 3:
        window=signal.tukey(np.size(t[:zero_crossings[1]]),alpha=0.05)
        while np.size(window) < np.size(t):
            window=np.append(window,0)
        y1_windowed=y10*window
        y2_windowed=y20*window
        print("used window -3")
    else:
        y1_windowed=y10
        y2_windowed=y20
        print("not windowed")
    new_file_data = np.stack((t, y10, y20), axis=1)
    np.savetxt("Unwindowed file", new_file_data)
    unwindowed_data=np.genfromtxt("Unwindowed file")
    windowed_data=np.genfromtxt(windowed)
    print("WINDOW IS",window)
    fig,ax =plt.subplots(figsize=(6,4),dpi=300)
    plt.tight_layout()
    plt.xlabel(r"Time series, au")
    plt.ylabel(r"Strain, au")
    ax.plot(unwindowed_data[:,0],unwindowed_data[:,1],'r--',label="Unwindowed data")
    ax.plot(windowed_data[:,0],windowed_data[:,1],'b',alpha=0.4,label="Windowed data")
    ax.plot(unwindowed_data[:,0],window*3e-4,'g',label="Window")
    ax.legend()
    plt.savefig("Windowing.png",bbox_inches="tight")


'''Plot all windowed waveforms'''

def all_graphs():
    f, axes = plt.subplots(5, 1, figsize=(7,20), sharex=True)
    f.tight_layout()
    for num, i in enumerate((1,2,4,8,16)):
        ax = axes[num]
        waveform = np.genfromtxt("waveforms/"+waveforms_non_merge[i])
        ax.plot(waveform[:,0], waveform[:,1])
        ax.set_title(waveforms_non_merge[i])
    plt.savefig("All SELECTED NM waveforms windowed")

    f2, axes2 = plt.subplots(5, 1, figsize=(7,20), sharex=True)
    f2.tight_layout()
    for num, i in enumerate((1,2,4,8,16)):
        ax2 = axes2[num]
        waveform = np.genfromtxt("h_psi4/"+waveforms_non_merge[i])
        ax2.plot(waveform[:,0], waveform[:,1])
        ax2.set_title(waveforms_non_merge[i])
    plt.savefig("All SELECTED NM waveforms unwindowed")
