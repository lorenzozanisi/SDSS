import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib as mpl
mpl.use('agg')
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.weightstats import DescrStatsW
import sklearn
from astropy.stats import  jackknife_stats


class SDSS_Plots:

    def __init__(self, HM_min, HM_max, HM_bin,string='original'):
        self.AnalyticHaloMass_min = HM_min
        self.AnalyticHaloMass_max = HM_max
        self.AnalyticHaloBin = HM_bin
        self.AnalyticHaloMass = np.arange(self.AnalyticHaloMass_min, self.AnalyticHaloMass_max, self.AnalyticHaloBin)
        self.choice=string
        
        """Cutting SDSS Data"""
        if self.choice=='original':

        
            df = pd.read_csv('./Catalog_SDSS_complete.dat',delim_whitespace=True)
            #df = df.rename(columns={'z':'zMeert'})
            df = df[(df.logR80NotTrunc!=-999) &(df.MsMendSerExp>9) & (df.logReSerExp!=-999)]
            fracper=len(df)/670722
            
            skycov=8000.
            self.fracsky=(skycov*fracper)/(4*np.pi*(180./np.pi)**2.)
            #df.drop('logReSerExp',axis=1, inplace=True)
            #df.rename({'logReSerExpCirc':'logReSerExp'}, inplace=True, axis=1)
            #df = df[df.BT<0.2]
        elif self.choice=='cleaned':
            Header=['galcount','finalflag','z','Vmaxwt','MsMendSerExp','AbsMag','logReSerExp',
                                   'BT','n_bulge','NewLCentSat','NewMCentSat'
                                   ,'MhaloL','probaE','probaEll',
                                 'probaS0','probaSab','probaScd','TType','P_S0',
                               'veldisp','veldisperr','raSDSS7','decSDSS7']
            
            df = pd.read_csv('./new_catalog_morph_flag_finalflag_Rmaj.dat', header = None, names = Header, skiprows = 1, delim_whitespace = True)
            goodness_cut = (df.finalflag==3 ) | (df.finalflag==5) | (df.finalflag==1)

            df = df[goodness_cut]

            df = df[df.Vmaxwt>0]
            df.loc[df.finalflag==5,'BT']=0
            df.loc[df.finalflag==1,'BT']=1

            fracper=len(df)/670722
            skycov=8000.
            self.fracsky=(skycov*fracper)/(4*np.pi*(180./np.pi)**2.)
            
        else:
            raise ValueError('choice must be either "cleaned" or "original"')
        #Clears NAN/ unsuable data
        #Clears NAN/ unsuable data
        df_noNAN = df.dropna()
       # df_noNAN = df_noNAN[df.Msflagserexp == 0]
        df_noNAN = df_noNAN[df.Vmaxwt > 0]
        df_noNAN = df_noNAN[df.MsMendSerExp > 0]
        #df_noNAN = df_noNAN[df.flagserexp >=0]
        #df_noNAN = df_noNAN[df.flagserexp <=3]
        #Redshift Cut, making Cent and Sat DB
        df_z = df_noNAN[df_noNAN.zMeert < 0.25]
        #df_z = df_z[df_z.probaE>0]
        #df_z = df_z[df_z.MhaloL>0]
        self.df_cent = df_z[df_z.NewLCentSat == 1]
        self.df_sat = df_z[df_z.NewLCentSat == 0]
        self.df_z=df_z
       # Data_mh_orig = np.array(self.df_sat.MhaloL)
        Data_vmax_orig = np.array(self.df_sat.Vmaxwt)
        Data_ms = np.array(self.df_sat.MsMendSerExp)



    def test_BT_TType_correlation(self):
        df = self.df_cent.copy()
        mask = (df.MsMendSerExp.values>10.) & (df.MsMendSerExp.values<11.)
        df = df[mask]
        BTbins = np.arange(0,1.1,0.1)
        TTypebins = [0,1,2,3,10]
        
        for j in range(len(TTypebins)-1):
                tt = 0.5*(TTypebins[j]+TTypebins[j+1])
                mask = (df.TType.values>TTypebins[j]) & (df.TType.values<TTypebins[j+1])
                df_temp = df[mask]
                hist = np.histogram(df_temp, bins=BTbins, density=True)[0]
                
                arr = np.array([BTbins[1:]-0.05, hist]).T
                
                np.savetxt('./BT_TTypes/LTGs_test_BT_TType_correlation_TType'+str(tt)+'.txt',arr)
              
        TTypebins = [-10,-3,-2,-1,0]
        
        for j in range(len(TTypebins)-1):
                tt = 0.5*(TTypebins[j]+TTypebins[j+1])
                mask = (df.TType.values>TTypebins[j]) & (df.TType.values<TTypebins[j+1])
                df_temp = df[mask]
                hist = np.histogram(df_temp, BTbins, density=True)[0]
                
                arr = np.array([BTbins[1:]-0.05, hist]).T
                
                np.savetxt('./BT_TTypes/ETGs_test_BT_TType_correlation_TType'+str(tt)+'.txt',arr)
        return
            

    def print_DF(self):
        print(self.df_cent)

    def calc_fractions(self, SM_LB, SM_UB):
        df = self.df_cent.copy()
        ETGs = df.TType<=0
        LTGs = df.TType>0

        dfETGs=df[ETGs]
        dfLTGs=df[LTGs]
        Mstarbins=np.arange(SM_LB,SM_UB,0.1)
        fracL = []
        for i in range(len(Mstarbins)-1):
            m = ma.masked_inside(df.MsMendSerExp.values, Mstarbins[i], Mstarbins[i+1]).mask
            mL = ma.masked_inside(dfLTGs.MsMendSerExp.values, Mstarbins[i], Mstarbins[i+1]).mask
            fracL.append(len(dfLTGs[mL])/len(df[m]))
            
        return np.array(fracL)
    
    def calc_ReMstar(self,df, binstart, s, w=None, choice='logReSerExp'):
        
       # if s=='etgs':
        #    Vmaxwt= df.Vmaxwt.values*( df.probaEll.values + df.probaS0.values)
        #elif s=='ltgs':
         #   Vmaxwt= df.Vmaxwt.values*( df.probaSab.values + df.probaScd.values)

        
        if w is not None:
            Weights=w.copy()
        Ms = abs(df.MsMendSerExp.values)
        Re = df[choice].values

        Vmaxwt= df.Vmaxwt.values
        Weights=Vmaxwt.copy()
        HM_Mask = ma.masked_inside(Ms, binstart, binstart+0.1).mask#HM_Bin+0.5).mask
        #print(len(Re[HM_Mask]),'SMR')
        if len(Re[HM_Mask]) < 10:
            perc = np.zeros(5)
            perc.fill(np.nan)
        else:
     #   
                #print(max(Mhalo[HM_Mask]), min(Mhalo[HM_Mask]))
            Weights = Weights[HM_Mask]
            sizes = Re[HM_Mask]
     #      print(len(sizes))
            ds = DescrStatsW(sizes, weights=Weights)
            perc= ds.quantile([0.0015,0.16,0.5,0.84,0.9985], return_pandas=False)
        return perc.T

    def size_distributions(self,SM_LB,SM_UB):
        
       # LTGs = self.df_cent.TType>0
        df= self.df_cent.copy()#[LTGs]
        ReBins=np.arange(-1,1.7,0.1)
        m = np.ma.masked_inside(df.MsMendSerExp, SM_LB, SM_UB).mask
        hist = np.histogram(df.logReSerExp.values[m], weights = df.Vmaxwt.values[m],  
                            bins=ReBins, density=True)[0]
        return hist
            
    
    def JK_resampling(self, phiETGs, phiLTGs, phiAll):
        
        Rebins =  np.arange(-1,3.,0.1)
        phi_ETGs=np.empty(len(Rebins)-1)
        err_ETGs= np.empty(len(Rebins)-1)
        phi_LTGs=np.empty(len(Rebins)-1)
        err_LTGs= np.empty(len(Rebins)-1)
        phi_all=np.empty(len(Rebins)-1)
        err_all= np.empty(len(Rebins)-1) 
        
        statistic=np.mean
        for i in range(len(Rebins)-1):
            phi_ETGs[i],bias,err_ETGs[i],conf =  jackknife_stats(phiETGs.T[i],statistic)
            phi_LTGs[i],bias,err_LTGs[i],conf =  jackknife_stats(phiLTGs.T[i],statistic)
            phi_all[i],bias,err_all[i],conf =  jackknife_stats(phiAll.T[i],statistic)
            
            
        return phi_ETGs,err_ETGs, phi_LTGs,err_LTGs,phi_all,err_all
    
    def Jackknifed_sizefunct(self,ReE, VmaxE, ReL, VmaxL, SMmean, count=None, use_astropy=False,
                            path = '/home/lorenzo/PhD/data/SDSS/SDSS_Processing/new_catalogs/ReF/Rmaj'):
        
 
        TTypeE=np.repeat(1,len(ReE))
        dictE = {'Re':ReE,'Vmaxwt':VmaxE, 'TType':TTypeE}
        dfE = pd.DataFrame.from_dict(dictE)
        
        TTypeL=np.repeat(0,len(ReL))
        dictL = {'Re':ReL,'Vmaxwt':VmaxL, 'TType':TTypeL}
        dfL = pd.DataFrame.from_dict(dictL)
        
        df = pd.concat([dfE,dfL])
        df = sklearn.utils.shuffle(df)
        
       
        Rebins=np.arange(-1.,3.,0.1)
        Rewidth = 0.1
        N=100
        phiETGs=np.zeros((N, len(Rebins)-1))
        phiLTGs=np.copy(phiETGs)
        phiAll =np.copy(phiETGs)
        L = len(df)
        Nmax=int(L/N)
        #stars=df[df.TType<=0].MsMendSerExp.values
        #m=np.ma.masked_inside(stars,9.3,9.4).mask.astype(int)
    #print(np.count_nonzero(m))
        for i in range(N):
            if i%50==0:
                print(i)
            if i < N-1:
                ind = np.random.randint(0,len(df), size=Nmax).astype(int)
                subcat = df.iloc[ind]
                
                ETGs = subcat.TType.values == 1
                LTGs =subcat.TType.values == 0
                
                if count is not None:
                    newfracsky =self.fracsky*(len(self.df_z)-count)/len(self.df_z)
                    fracnew = newfracsky*len(subcat)/L
                else:
                    fracnew = self.fracsky*len(subcat)/L
                phiETGs[i] = np.histogram(subcat[ETGs].Re.values, bins=Rebins,
                                weights=subcat[ETGs].Vmaxwt.values)[0]
                phiETGs[i] = phiETGs[i]/fracnew/Rewidth
        
                phiLTGs[i] = np.histogram(subcat[LTGs].Re.values, bins=Rebins, 
                            weights=subcat[LTGs].Vmaxwt.values)[0]
                phiLTGs[i] = phiLTGs[i]/fracnew/Rewidth
        
                phiAll[i] = np.histogram(subcat.Re.values, bins=Rebins, 
                                  weights=subcat.Vmaxwt.values)[0]
                phiAll[i]  = phiAll[i]/fracnew/Rewidth
        
                df = df.drop(ind)
                df.index = np.arange(len(df)).astype(int)
            else:       
                subcat=df.copy()
                
                ETGs = subcat.TType ==1
                LTGs =subcat.TType ==0
            
                fracnew = self.fracsky*len(subcat)/L
                
                phiETGs[i] = np.histogram(subcat[ETGs].Re.values, bins=Rebins, 
                            weights=subcat[ETGs].Vmaxwt.values)[0]
                phiETGs[i] = phiETGs[i]/fracnew/Rewidth
                
                phiLTGs[i] = np.histogram(subcat[LTGs].Re.values, bins=Rebins, 
                                weights=subcat[LTGs].Vmaxwt.values)[0] 
                phiLTGs[i] = phiLTGs[i]/fracnew/Rewidth
                
                phiAll[i] = np.histogram(subcat.Re.values, bins=Rebins, 
                                  weights=subcat.Vmaxwt.values)[0]
                phiAll[i]  = phiAll[i]/fracnew/Rewidth
                
               
        
        if use_astropy:
            
            phi_ETGs,err_ETGs, phi_LTGs,err_LTGs,phi_all,err_all = self.JK_resampling( phiETGs, phiLTGs, phiAll)
#            path='/home/lorenzo/PhD/data/SDSS/SDSS_Processing/new_catalogs/ReF/'

            Reok=Rebins[1:]-0.05
            array=np.array([Reok,phi_ETGs,err_ETGs])
     
            np.savetxt(path+'/'+str(SMmean)+'/Re_ETGs'+str(SMmean)+'centJK'+str(self.choice)+'.txt',array.T)
        

            array=np.array([Reok,phi_LTGs,err_LTGs])
            np.savetxt(path+'/'+str(SMmean)+'/Re_LTGs'+str(SMmean)+'centJK'+str(self.choice)+'.txt',array.T)
        
            array=np.array([Reok,phi_all,err_all])
            np.savetxt(path+'/'+str(SMmean)+'/Re_all'+str(SMmean)+'centJK'+str(self.choice)+'.txt',array.T)
        
            return 
    
        phi_all = np.array(list(map(np.mean,phiAll.T)))
        err_all = np.array(list(map(np.std,phiAll.T)))

        phi_ETGs = np.array(list(map(np.mean,phiETGs.T)))
        err_ETGs = np.array(list(map(np.std,phiETGs.T)))

        phi_LTGs = np.array(list(map(np.mean,phiLTGs.T)))
        err_LTGs = np.array(list(map(np.std,phiLTGs.T)))
            
         
       # path='/home/lorenzo/PhD/data/SDSS/SDSS_Processing/new_catalogs/ReF'

        Reok=Rebins[1:]-0.05
        array=np.array([Reok,phi_ETGs,err_ETGs])
        print(array.T.shape)
        np.savetxt(path+'/'+str(SMmean)+'Re_ETGs'+str(SMmean)+'centJK'+str(self.choice)+'.txt',array.T)  
        
        array=np.array([Reok,phi_LTGs,err_LTGs])
        np.savetxt(path+'/'+str(SMmean)+'Re_LTGs'+str(SMmean)+'centJK'+str(self.choice)+'.txt',array.T)
        
        array=np.array([Reok,phi_all,err_all])
        np.savetxt(path+'/'+str(SMmean)+'Re_all'+str(SMmean)+'centJK'+str(self.choice)+'.txt',array.T)
            
        return 
                
    def sizefunctions_pruned_and_S0s(self, SM_LB,SM_UB, return_Jackknife=False,use_astropy=False):
        
        SMmean = (SM_UB+SM_LB)/2.
        df= self.df_cent.copy()
        
        #df=df[ np.ma.masked_inside(df.MsMendSerExp.values, SM_LB, SM_UB).mask ]
        
        ETGs = (df.TType<=0) & (df.P_S0<0.5)
        S0s = (df.TType<=0) & (df.P_S0>0.5)
        LTGs = df.TType>0

        dfETGs=df[ETGs]
        dfLTGs=df[LTGs]
        dfS0s=df[S0s]
        
        Re_BinWidth=0.1
        Mstarbins=np.arange(SM_LB,SM_UB,0.1)
        Re_LB = -1; Re_UB =1.7
        Rebins = np.arange(Re_LB, Re_UB, Re_BinWidth)     
        
        percE=np.zeros((len(Mstarbins),5))
        percL=percE.copy()
        ReE=np.array([])
        ReL=np.copy(ReE)
        VmaxE=np.copy(ReE)
        VmaxL=np.copy(ReE)
        
        percS=percE.copy()
        ReS=np.array([])
        VmaxS=np.copy(ReS)
        for i, b in enumerate(Mstarbins):
            countE=0
            countL=0
            
            me = ma.masked_inside(dfETGs.MsMendSerExp.values, b,b+0.1).mask
            ml = ma.masked_inside(dfLTGs.MsMendSerExp.values, b,b+0.1).mask
            mS0s = ma.masked_inside(dfS0s.MsMendSerExp.values, b,b+0.1).mask
            
            ne= np.count_nonzero(me.astype(int))  #True = 1
            nl= np.count_nonzero(ml.astype(int))
            nS0s= np.count_nonzero(mS0s.astype(int))
            
            dfE=dfETGs[me]
            dfL=dfLTGs[ml]
            dfS =dfS0s[mS0s]
            
            if (ne >10): 
                percE[i]=self.calc_ReMstar(dfE,b,s='none' )
                
                maskE = np.ma.masked_inside(dfE.logReSerExp.values,percE[i][0],percE[i][4]).mask
                df_tempe= dfE[maskE]
                ReE_tempe = df_tempe.logReSerExp.values
                
                for r in Rebins:
                    maskrE = np.ma.masked_inside(ReE_tempe, r,r+0.1).mask
                    le = np.count_nonzero(maskrE.astype(int))
                   
                    if le >=4:
                    
                        ReE = np.append(ReE, ReE_tempe[maskrE])
                        VmaxE = np.append(VmaxE, df_tempe.Vmaxwt.values[maskrE])
                        
                        
            if (nl>10):
                percL[i]=self.calc_ReMstar(dfL,b,s='none')
                maskL = np.ma.masked_inside(dfL.logReSerExp.values,percL[i][0],percL[i][4]).mask
                df_templ= dfL[maskL]
                ReL_templ = df_templ.logReSerExp.values                

                for r in Rebins:
                    maskrL = np.ma.masked_inside(ReL_templ, r,r+0.1).mask
                    ll = np.count_nonzero(maskrL.astype(int))
                  
                    if ll >=4:
                    
                        ReL = np.append(ReL, ReL_templ[maskrL])
                        VmaxL = np.append(VmaxL, df_templ.Vmaxwt.values[maskrL])
                        
            if nS0s>10:
                percS[i]=self.calc_ReMstar(dfS,b,s='none')
 
                maskS = np.ma.masked_inside(dfS.logReSerExp.values,percS[i][0],percS[i][4]).mask
                df_tempS= dfS[maskS]
                ReS_tempS = df_tempS.logReSerExp.values
                
                for r in Rebins:
                    maskrS = np.ma.masked_inside(ReS_tempS, r,r+0.1).mask
                    lS = np.count_nonzero(maskrS.astype(int))
                   
                    if lS >=4:
                    
                        ReS = np.append(ReS, ReS_tempS[maskrS])
                        VmaxS = np.append(VmaxS, df_tempS.Vmaxwt.values[maskrS])



                    #    countE += np.count_nonzero(maskE.astype(int)-1) # how many objects are we discarding?
                
                


          #      ReE = np.append(ReE, dfE.logReSerExp.values)
          #      VmaxE = np.append(VmaxE, dfE.Vmaxwt.values)
          #      ReL = np.append(ReL, dfL.logReSerExp.values)
          #      VmaxL = np.append(VmaxL, dfL.Vmaxwt.values)  
                
                
        if return_Jackknife:
            
            try:
                self.Jackknifed_sizefunct(ReE, VmaxE, ReL, VmaxL, SMmean, use_astropy)
            except:
                
                pass
            return 
       # countTot=countL+countE   
        countTot= 11536    #objects discarded
        newfracsky =self.fracsky*(len(self.df_z)-countTot)/len(self.df_z)
       
   #     plt.scatter(dfE.MsMendSerExp, dfE.logReSerExp)
        
        SMmean = (SM_UB+SM_LB)/2.
   
        path='/home/lorenzo/PhD/data/SDSS/SDSS_Processing/new_catalogs/ReF/disk_ell_S0s/'
        Re_LB = -1; Re_UB =1.7
        Re_Bins = np.arange(Re_LB, Re_UB, 0.1)
        bins=Re_Bins[1:]-0.1/2.            
        
        hist_ETGs_, edges = np.histogram(ReE, bins = Re_Bins, weights = VmaxE)
        num_per_bin = np.histogram(ReE, bins = Re_Bins)[0]
        
        mask = np.ma.masked_greater(num_per_bin, 5).mask
        hist_ETGs = hist_ETGs_[mask]
        num_per_bin =num_per_bin[mask]
        bins_okETGs=bins[mask]
        yETGs = np.divide(hist_ETGs,self.fracsky*Re_BinWidth)
        poiserrETGs = yETGs/np.sqrt(num_per_bin)
        array=np.array([bins_okETGs,yETGs,poiserrETGs])
        print(yETGs,'E')
        if len(yETGs)>0:
            np.savetxt(path+'/'+str(SMmean)+'/Re_ETGs'+str(SMmean)+'cent.txt',array.T)
        else:
            array=np.zeros(3)
            np.savetxt(path+'/'+str(SMmean)+'/Re_ETGs'+str(SMmean)+'cent.txt',array.T)
        
        hist_S0s_, edges = np.histogram(ReS, bins = Re_Bins, weights = VmaxS)
        num_per_bin = np.histogram(ReS, bins = Re_Bins)[0]
        
        mask = np.ma.masked_greater(num_per_bin, 5).mask
        hist_S0s = hist_S0s_[mask]
        num_per_bin =num_per_bin[mask]
        bins_okS0s=bins[mask]
        yS0s = np.divide(hist_S0s,self.fracsky*Re_BinWidth)
        poiserrS0s = yS0s/np.sqrt(num_per_bin)
        array=np.array([bins_okS0s,yS0s,poiserrS0s])
        
        print(yS0s,'S0')
        if  len(yS0s)>0:
            np.savetxt(path+'/'+str(SMmean)+'/Re_S0s'+str(SMmean)+'cent.txt',array.T)
        else:
            array=np.zeros(3)
            np.savetxt(path+'/'+str(SMmean)+'/Re_S0s'+str(SMmean)+'cent.txt',array.T)
        
        
        hist_LTGs_, edges = np.histogram(ReL, bins = Re_Bins, weights = VmaxL)
        num_per_bin = np.histogram(ReL, bins = Re_Bins)[0]
        
        mask = np.ma.masked_greater(num_per_bin, 5).mask
        hist_LTGs = hist_LTGs_[mask]
        num_per_bin =num_per_bin[mask]
        bins_okLTGs=bins[mask]
        yLTGs = np.divide(hist_LTGs,self.fracsky*Re_BinWidth)
        poiserrLTGs = yLTGs/np.sqrt(num_per_bin)
        array=np.array([bins_okLTGs,yLTGs,poiserrLTGs])
        
        if  len(yLTGs)>0:
            np.savetxt(path+'/'+str(SMmean)+'/Re_LTGs'+str(SMmean)+'cent.txt',array.T)
        else:
            array=np.zeros(3)
            np.savetxt(path+'/'+str(SMmean)+'/Re_LTGs'+str(SMmean)+'cent.txt',array.T)
        
        
        ReAll = np.append(ReE,np.append(ReL,ReS))
        VmaxAll= np.append(VmaxE,np.append(VmaxL,VmaxS))
        hist_All_, edges = np.histogram(ReAll, bins = Re_Bins, weights = VmaxAll)
        num_per_bin = np.histogram(ReAll, bins = Re_Bins)[0]
        
        mask = np.ma.masked_greater(num_per_bin, 10).mask
        hist_All = hist_All_[mask]
        num_per_bin =num_per_bin[mask]
        bins_okAll=bins[mask]
        yAll = np.divide(hist_All,self.fracsky*Re_BinWidth)
        poiserrAll = yAll/np.sqrt(num_per_bin)        
        array=np.array([bins_okAll,yAll,poiserrAll])
        np.savetxt(path+'/'+str(SMmean)+'/Re_all'+str(SMmean)+'cent.txt',array.T)        
    
        return yETGs, poiserrETGs,bins_okETGs, yLTGs, poiserrLTGs,bins_okLTGs, yS0s, poiserrS0s,bins_okS0s
               
        
        
    def sizefunctions_pruned(self, SM_LB,SM_UB, return_Jackknife=False,use_astropy=False,
                             choice='logReSerExp'):
        
        SMmean = (SM_UB+SM_LB)/2.
        df= self.df_cent.copy()
        if choice=='logR80NotTrunc':
            corr = df['logReSerExp']-df['logReSerExpNotTrunc']
            df['logR80NotTrunc'] = df['logR80NotTrunc'] + corr # truncation suggestion by Mariangela
        #df=df[ np.ma.masked_inside(df.MsMendSerExp.values, SM_LB, SM_UB).mask ]
        
        Ellip = df.TType<=0 & df.P_S0<0.5 & df.BT>0.6 & df.ba_tot>0.5
        S0s = df.TType<=0 & df.P_S0>0
        ETGs = Ellip | S0s
        LTGs = df.TType>0

        dfETGs=df[ETGs]
        dfLTGs=df[LTGs]
        Re_BinWidth=0.1
        Mstarbins=np.arange(SM_LB,SM_UB,0.1)
        Re_LB = -1; Re_UB =3
        Rebins = np.arange(Re_LB, Re_UB, Re_BinWidth)        
        percE=np.zeros((len(Mstarbins),5))
        percL=percE.copy()
        ReE=np.array([])
        ReL=np.copy(ReE)
        VmaxE=np.copy(ReE)
        VmaxL=np.copy(ReE)
        
     
        for i, b in enumerate(Mstarbins):
            countE=0
            countL=0
            
            me = ma.masked_inside(dfETGs.MsMendSerExp.values, b,b+0.1).mask
            ml = ma.masked_inside(dfLTGs.MsMendSerExp.values, b,b+0.1).mask
            ne= np.count_nonzero(me.astype(int))  #True = 1
            nl= np.count_nonzero(ml.astype(int))
            
            dfE=dfETGs[me]
            dfL=dfLTGs[ml]
   
            if (ne >=10 and nl >= 10):

                percE[i]=self.calc_ReMstar(dfE,b,s='none', choice=choice )
                percL[i]=self.calc_ReMstar(dfL,b,s='none', choice=choice)
            
                
                maskE = np.ma.masked_inside(dfE[choice].values,percE[i][0],percE[i][4]).mask
                df_tempe= dfE[maskE]
                ReE_tempe = df_tempe[choice].values
                
                for r in Rebins:
                    maskrE = np.ma.masked_inside(ReE_tempe, r,r+0.1).mask
                    le = np.count_nonzero(maskrE.astype(int))
                   
                    if le >=4:
                    
                        ReE = np.append(ReE, ReE_tempe[maskrE])
                        VmaxE = np.append(VmaxE, df_tempe.Vmaxwt.values[maskrE])
                        countE += np.count_nonzero(maskE.astype(int)-1) # how many objects are we discarding?
                
                
                maskL = np.ma.masked_inside(dfL[choice].values,percL[i][0],percL[i][4]).mask
                df_templ= dfL[maskL]
                ReL_templ = df_templ[choice].values                

                for r in Rebins:
                    maskrL = np.ma.masked_inside(ReL_templ, r,r+0.1).mask
                    ll = np.count_nonzero(maskrL.astype(int))
                  
                    if ll >=4:
                    
                        ReL = np.append(ReL, ReL_templ[maskrL])
                        VmaxL = np.append(VmaxL, df_templ.Vmaxwt.values[maskrL])
                        countL += np.count_nonzero(maskL.astype(int)-1) # how many objects are we discarding?
          #      ReE = np.append(ReE, dfE.logReSerExp.values)
          #      VmaxE = np.append(VmaxE, dfE.Vmaxwt.values)
          #      ReL = np.append(ReL, dfL.logReSerExp.values)
          #      VmaxL = np.append(VmaxL, dfL.Vmaxwt.values)  
                
        countTot=countL+countE   
        print(countTot)  
        if choice=='logR80NotTrunc':
            path='/home/lorenzo/PhD/data/SDSS/SDSS_Processing/new_catalogs/R80'
        elif choice=='logReSerExp':
            path = '/home/lorenzo/PhD/data/SDSS/SDSS_Processing/new_catalogs/ReF/Rmaj'
        else:
            raise ValueError#'either logR80NotTrunc or logReSerExp'
                
        if return_Jackknife:
            
            self.Jackknifed_sizefunct(ReE, VmaxE, ReL, VmaxL, SMmean, countTot, use_astropy, path=path)
            
            return 

        #countTot= 11536    #objects discarded
        newfracsky =self.fracsky*(len(self.df_z)-countTot)/len(self.df_z)
       
   #     plt.scatter(dfE.MsMendSerExp, dfE.logReSerExp)
        
        SMmean = (SM_UB+SM_LB)/2.
   
   #     path='/home/lorenzo/PhD/data/SDSS/SDSS_Processing/new_catalogs/R80'
        Re_LB = -1; Re_UB =3
        Re_Bins = np.arange(Re_LB, Re_UB, 0.1)
        bins=Re_Bins[1:]-0.1/2.            
        
        hist_ETGs_, edges = np.histogram(ReE, bins = Re_Bins, weights = VmaxE)
        num_per_bin = np.histogram(ReE, bins = Re_Bins)[0]
        
        mask = np.ma.masked_greater(num_per_bin, 5).mask
        hist_ETGs = hist_ETGs_[mask]
        num_per_bin =num_per_bin[mask]
        bins_okETGs=bins[mask]
        yETGs = np.divide(hist_ETGs,self.fracsky*Re_BinWidth)
        poiserrETGs = yETGs/np.sqrt(num_per_bin)
        array=np.array([bins_okETGs,yETGs,poiserrETGs])
        np.savetxt(path+'/'+str(SMmean)+'/Re_ETGs'+str(SMmean)+'JKcent.txt',array.T)
        print(path+'/'+str(SMmean)+'/Re_ETGs'+str(SMmean)+'JKcent.txt')
        
        
        
        hist_LTGs_, edges = np.histogram(ReL, bins = Re_Bins, weights = VmaxL)
        num_per_bin = np.histogram(ReL, bins = Re_Bins)[0]
        
        mask = np.ma.masked_greater(num_per_bin, 5).mask
        hist_LTGs = hist_LTGs_[mask]
        num_per_bin =num_per_bin[mask]
        bins_okLTGs=bins[mask]
        yLTGs = np.divide(hist_LTGs,self.fracsky*Re_BinWidth)
        poiserrLTGs = yLTGs/np.sqrt(num_per_bin)
        array=np.array([bins_okLTGs,yLTGs,poiserrLTGs])
        np.savetxt(path+'/'+str(SMmean)+'/Re_LTGs'+str(SMmean)+'JKcent.txt',array.T)
        
        
        ReAll = np.append(ReE,ReL)
        VmaxAll= np.append(VmaxE,VmaxL)
        hist_All_, edges = np.histogram(ReAll, bins = Re_Bins, weights = VmaxAll)
        num_per_bin = np.histogram(ReAll, bins = Re_Bins)[0]
        
        mask = np.ma.masked_greater(num_per_bin, 10).mask
        hist_All = hist_All_[mask]
        num_per_bin =num_per_bin[mask]
        bins_okAll=bins[mask]
        yAll = np.divide(hist_All,self.fracsky*Re_BinWidth)
        poiserrAll = yAll/np.sqrt(num_per_bin)        
        array=np.array([bins_okAll,yAll,poiserrAll])
        np.savetxt(path+'/'+str(SMmean)+'/Re_all'+str(SMmean)+'JKcent.txt',array.T)        
    
        return yETGs, poiserrETGs,bins_okETGs, yLTGs, poiserrLTGs,bins_okLTGs, yAll, poiserrAll,bins_okAll
       
        
        
    def sizefunctions_pruned_morph_TTypebins(self, SM_LB,SM_UB):
        
        SMmean = (SM_UB+SM_LB)/2.
        df= self.df_cent.copy()
        
        #df=df[ np.ma.masked_inside(df.MsMendSerExp.values, SM_LB, SM_UB).mask ]
        
        ETGs = df.TType<=0
        LTGs = df.TType>0

        dfETGs=df[ETGs]
        dfLTGs=df[LTGs]
        Re_BinWidth=0.1
        Mstarbins=np.arange(SM_LB,SM_UB,0.1)
        Re_LB = -1; Re_UB =1.7
        Rebins = np.arange(Re_LB, Re_UB, Re_BinWidth)        
        percE=np.zeros((len(Mstarbins),5))
        percL=percE.copy()
        ReE=np.array([])
        ReL=np.copy(ReE)
        VmaxE=np.copy(ReE)
        VmaxL=np.copy(ReE)
        TTypeE=np.copy(ReE)
        TTypeL=np.copy(ReE)

        for i, b in enumerate(Mstarbins):
            countE=0
            countL=0
            
            me = ma.masked_inside(dfETGs.MsMendSerExp.values, b,b+0.1).mask
            ml = ma.masked_inside(dfLTGs.MsMendSerExp.values, b,b+0.1).mask
            ne= np.count_nonzero(me.astype(int))  #True = 1
            nl= np.count_nonzero(ml.astype(int))
            
            dfE=dfETGs[me]
            dfL=dfLTGs[ml]
         
            if (ne >=10 ):

                percE[i]=self.calc_ReMstar(dfE,b,s='none' )
                percL[i]=self.calc_ReMstar(dfL,b,s='none')
            
                
                maskE = np.ma.masked_inside(dfE.logReSerExp.values,percE[i][0],percE[i][4]).mask
                df_tempe= dfE[maskE]
                ReE_tempe = df_tempe.logReSerExp.values
                
                for r in Rebins:
                    maskrE = np.ma.masked_inside(ReE_tempe, r,r+0.1).mask
                    le = np.count_nonzero(maskrE.astype(int))
                   
                    if le >=4:
                    
                        ReE = np.append(ReE, ReE_tempe[maskrE])
                        VmaxE = np.append(VmaxE, df_tempe.Vmaxwt.values[maskrE])
                        TTypeE = np.append(TTypeE,df_tempe.TType.values[maskrE])
                    #    countE += np.count_nonzero(maskE.astype(int)-1) # how many objects are we discarding?
                
            if (nl>=10):
                maskL = np.ma.masked_inside(dfL.logReSerExp.values,percL[i][0],percL[i][4]).mask
                df_templ= dfL[maskL]
                ReL_templ = df_templ.logReSerExp.values                

                for r in Rebins:
                    maskrL = np.ma.masked_inside(ReL_templ, r,r+0.1).mask
                    ll = np.count_nonzero(maskrL.astype(int))
                  
                    if ll >=4:
                    
                        ReL = np.append(ReL, ReL_templ[maskrL])
                        VmaxL = np.append(VmaxL, df_templ.Vmaxwt.values[maskrL])
                        TTypeL = np.append(TTypeL,df_templ.TType.values[maskrL])

                
        countTot= 11536    #objects discarded
        newfracsky =self.fracsky*(len(self.df_z)-countTot)/len(self.df_z)
       
   #     plt.scatter(dfE.MsMendSerExp, dfE.logReSerExp)
        
        SMmean = (SM_UB+SM_LB)/2.
   
        path='/home/lorenzo/PhD/data/SDSS/SDSS_Processing/new_catalogs/ReF/Rmaj'
        Re_LB = -1; Re_UB =1.7
        Re_Bins = np.arange(Re_LB, Re_UB, 0.1)
        bins=Re_Bins[1:]-0.1/2.         
        
        TTypebins = [0,1,2,3,10]
        
        for i in range(len(TTypebins)-1):
            btmean = np.round(0.5*(TTypebins[i]+TTypebins[i+1]),2)
            maskBTL = np.ma.masked_inside(TTypeL,TTypebins[i],TTypebins[i+1]).mask
            ReL_ = ReL[maskBTL]
            VmaxL_ = VmaxL[maskBTL]

            hist_LTGs_, edges = np.histogram(ReL_, bins = Re_Bins, weights = VmaxL_)
            
            num_per_bin = np.histogram(ReL_, bins = Re_Bins)[0]
        
            mask = np.ma.masked_greater(num_per_bin, 5).mask
            hist_LTGs = hist_LTGs_[mask]
            num_per_bin =num_per_bin[mask]
            bins_okLTGs=bins[mask]
            yLTGs = np.divide(hist_LTGs,self.fracsky*Re_BinWidth)
            poiserrLTGs = yLTGs/np.sqrt(num_per_bin)
            array=np.array([bins_okLTGs,yLTGs,poiserrLTGs])

            if len(yLTGs)>0:
                np.savetxt(path+'/TTypecuts/'+str(SMmean)+'/Re_LTGs'+str(SMmean)+'cent'+str(btmean)+str(self.choice)+'.txt',array.T)
            else:
                array=np.zeros(3)
                np.savetxt(path+'/TTypecuts/'+str(SMmean)+'/Re_LTGs'+str(SMmean)+'cent'+str(btmean)+str(self.choice)+'.txt',array.T)
        
        
        
        TTypebins = [-10,-3,-2,-1,0]

        for i in range(len(TTypebins)-1):
            
            btmean = np.round(0.5*(TTypebins[i]+TTypebins[i+1]),2)
            maskBTE = np.ma.masked_inside(TTypeE,TTypebins[i],TTypebins[i+1]).mask
            ReE_ = ReE[maskBTE]
            VmaxE_ = VmaxE[maskBTE]
        
        
            hist_ETGs_, edges = np.histogram(ReE_, bins = Re_Bins, weights = VmaxE_)
            num_per_bin = np.histogram(ReE_, bins = Re_Bins)[0]
            
            mask = np.ma.masked_greater(num_per_bin, 5).mask
            hist_ETGs = hist_ETGs_[mask]
            num_per_bin =num_per_bin[mask]
            bins_okETGs=bins[mask]
            yETGs = np.divide(hist_ETGs,self.fracsky*Re_BinWidth)
            poiserrETGs = yETGs/np.sqrt(num_per_bin)
            array=np.array([bins_okETGs,yETGs,poiserrETGs])

            if len(yETGs>0):
                np.savetxt(path+'/TTypecuts/'+str(SMmean)+'/Re_ETGs'+str(SMmean)+'cent'+str(btmean)+str(self.choice)+'.txt',array.T)
            else:
                array=np.zeros(3)
                np.savetxt(path+'/TTypecuts/'+str(SMmean)+'/Re_ETGs'+str(SMmean)+'cent'+str(btmean)+str(self.choice)+'.txt',array.T)
                       

        return 
    
            
    def sizefunctions_pruned_morph_BTbins(self, SM_LB,SM_UB):
        
        SMmean = (SM_UB+SM_LB)/2.
        df= self.df_cent.copy()
        
        #df=df[ np.ma.masked_inside(df.MsMendSerExp.values, SM_LB, SM_UB).mask ]
        
        ETGs = df.TType<=0
        LTGs = df.TType>0

        dfETGs=df[ETGs]
        dfLTGs=df[LTGs]
        Re_BinWidth=0.1
        Mstarbins=np.arange(SM_LB,SM_UB,0.1)
        Re_LB = -1; Re_UB =1.7
        Rebins = np.arange(Re_LB, Re_UB, Re_BinWidth)        
        percE=np.zeros((len(Mstarbins),5))
        percL=percE.copy()
        ReE=np.array([])
        ReL=np.copy(ReE)
        VmaxE=np.copy(ReE)
        VmaxL=np.copy(ReE)
        BTE=np.copy(ReE)
        BTL=np.copy(ReE)

        for i, b in enumerate(Mstarbins):
            countE=0
            countL=0
            
            me = ma.masked_inside(dfETGs.MsMendSerExp.values, b,b+0.1).mask
            ml = ma.masked_inside(dfLTGs.MsMendSerExp.values, b,b+0.1).mask
            ne= np.count_nonzero(me.astype(int))  #True = 1
            nl= np.count_nonzero(ml.astype(int))
            
            dfE=dfETGs[me]
            dfL=dfLTGs[ml]
         
            if (ne >=10 ):

                percE[i]=self.calc_ReMstar(dfE,b,s='none' )
                percL[i]=self.calc_ReMstar(dfL,b,s='none')
            
                
                maskE = np.ma.masked_inside(dfE.logReSerExp.values,percE[i][0],percE[i][4]).mask
                df_tempe= dfE[maskE]
                ReE_tempe = df_tempe.logReSerExp.values
                
                for r in Rebins:
                    maskrE = np.ma.masked_inside(ReE_tempe, r,r+0.1).mask
                    le = np.count_nonzero(maskrE.astype(int))
                   
                    if le >=4:
                    
                        ReE = np.append(ReE, ReE_tempe[maskrE])
                        VmaxE = np.append(VmaxE, df_tempe.Vmaxwt.values[maskrE])
                        BTE = np.append(BTE,df_tempe.BT.values[maskrE])
                    #    countE += np.count_nonzero(maskE.astype(int)-1) # how many objects are we discarding?
                
            if (nl>=10):
                maskL = np.ma.masked_inside(dfL.logReSerExp.values,percL[i][0],percL[i][4]).mask
                df_templ= dfL[maskL]
                ReL_templ = df_templ.logReSerExp.values                

                for r in Rebins:
                    maskrL = np.ma.masked_inside(ReL_templ, r,r+0.1).mask
                    ll = np.count_nonzero(maskrL.astype(int))
                  
                    if ll >=4:
                    
                        ReL = np.append(ReL, ReL_templ[maskrL])
                        VmaxL = np.append(VmaxL, df_templ.Vmaxwt.values[maskrL])
                        BTL = np.append(BTL,df_templ.BT.values[maskrL])

                
        countTot= 11536    #objects discarded
        newfracsky =self.fracsky*(len(self.df_z)-countTot)/len(self.df_z)
       
   #     plt.scatter(dfE.MsMendSerExp, dfE.logReSerExp)
        
        SMmean = (SM_UB+SM_LB)/2.
   
        path='/home/lorenzo/PhD/data/SDSS/SDSS_Processing/new_catalogs/ReF'
        Re_LB = -1; Re_UB =1.7
        Re_Bins = np.arange(Re_LB, Re_UB, 0.1)
        bins=Re_Bins[1:]-0.1/2.         
        
        BTbins = [0.,0.2,0.4,0.6,0.8,1.]

        for i,bt in enumerate(BTbins[:len(BTbins)-1]):
        
    
        
            btmean = np.round(0.5*(bt+bt+0.2),2)
            maskBTL = np.ma.masked_inside(BTL,bt,bt+0.2).mask
            ReL_ = ReL[maskBTL]
            VmaxL_ = VmaxL[maskBTL]

            maskBTE = np.ma.masked_inside(BTE,bt,bt+0.2).mask
            ReE_ = ReE[maskBTE]
            VmaxE_ = VmaxE[maskBTE]
 
        
        
            hist_ETGs_, edges = np.histogram(ReE_, bins = Re_Bins, weights = VmaxE_)
            num_per_bin = np.histogram(ReE_, bins = Re_Bins)[0]
            
            mask = np.ma.masked_greater(num_per_bin, 5).mask
            hist_ETGs = hist_ETGs_[mask]
            num_per_bin =num_per_bin[mask]
            bins_okETGs=bins[mask]
            yETGs = np.divide(hist_ETGs,self.fracsky*Re_BinWidth)
            poiserrETGs = yETGs/np.sqrt(num_per_bin)
            array=np.array([bins_okETGs,yETGs,poiserrETGs])

            if len(yETGs>0):
                np.savetxt(path+'/BTcuts/'+str(SMmean)+'/Re_ETGs'+str(SMmean)+'cent'+str(btmean)+str(self.choice)+'.txt',array.T)
            else:
                array=np.zeros(3)
                np.savetxt(path+'/BTcuts/'+str(SMmean)+'/Re_ETGs'+str(SMmean)+'cent'+str(btmean)+str(self.choice)+'.txt',array.T)
                
            hist_LTGs_, edges = np.histogram(ReL_, bins = Re_Bins, weights = VmaxL_)
            num_per_bin = np.histogram(ReL_, bins = Re_Bins)[0]
        
            mask = np.ma.masked_greater(num_per_bin, 5).mask
            hist_LTGs = hist_LTGs_[mask]
            num_per_bin =num_per_bin[mask]
            bins_okLTGs=bins[mask]
            yLTGs = np.divide(hist_LTGs,self.fracsky*Re_BinWidth)
            poiserrLTGs = yLTGs/np.sqrt(num_per_bin)
            array=np.array([bins_okLTGs,yLTGs,poiserrLTGs])

            if len(yLTGs)>0:
                np.savetxt(path+'/BTcuts/'+str(SMmean)+'/Re_LTGs'+str(SMmean)+'cent'+str(btmean)+str(self.choice)+'.txt',array.T)
            else:
                array=np.zeros(3)
                np.savetxt(path+'/BTcuts/'+str(SMmean)+'/Re_LTGs'+str(SMmean)+'cent'+str(btmean)+str(self.choice)+'.txt',array.T)
        
        

        return 
       

    def size_functions(self, SM_UB, SM_LB, Re_BinWidth=0.1, compute_SMR=False, 
                       make_plots=False,saved=True):
        if make_plots:
            fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(16,6))
            

        SMmean = (SM_UB+SM_LB)/2.
        path='/home/lorenzo/PhD/data/SDSS/SDSS_Processing/new_catalogs/ReF'
        Re_LB = -1; Re_UB =1.7
        Re_Bins = np.arange(Re_LB, Re_UB, Re_BinWidth)
        bins=Re_Bins[1:]-Re_BinWidth/2.
        select_HC =self.df_cent.probaE >0 #self.df_cent.copy()
        select_ReRange = (Re_LB < self.df_cent.logReSerExp) & (self.df_cent.logReSerExp < Re_UB)
        select_SMRange = (SM_LB < self.df_cent.MsMendSerExp) & (self.df_cent.MsMendSerExp < SM_UB)
        index = select_HC & select_ReRange & select_SMRange
        df_HC = self.df_cent.copy()         #self.df_cent[index]
        df_HC =df_HC[select_ReRange & select_HC & select_SMRange]
        
        Re = np.array(df_HC.logReSerExp)
        
        Vmax = np.array(df_HC.Vmaxwt)
        ETGwt = df_HC.probaEll.values + df_HC.probaS0.values
        ######## all galaxies
        Weights = Vmax
        hist_all, edges = np.histogram(Re, bins = Re_Bins, weights = Weights)
        num_per_bin_all = np.histogram(Re, bins = Re_Bins)[0]
        
        mask_all = np.ma.masked_greater(num_per_bin_all, 5).mask
        hist_all = hist_all[mask_all]
        num_per_bin_all =num_per_bin_all[mask_all]
        bins_ok=bins[mask_all]
        
        y_all = np.divide(hist_all,self.fracsky*Re_BinWidth)
        poiserr_all = y_all/np.sqrt(num_per_bin_all)
        
        if make_plots:
            ax1.errorbar(bins_ok, y_all, yerr=poiserr_all, linestyle='--', color='black')
            
        if saved:
            a=np.array([bins_ok,y_all,poiserr_all])
            np.savetxt(path+'/'+str(SMmean)+'/Re_all'+str(SMmean)+'cent.txt',a.T)        
       
        ####### ETGs according to bayesian probs

        ETGwt = df_HC.probaEll.values + df_HC.probaS0.values
        
        Weights = Vmax*ETGwt
        totweight = np.sum(Weights)
        
        hist_ETGs, edges = np.histogram(Re, bins = Re_Bins, weights = Weights)
        num_per_bin = np.histogram(Re, bins = Re_Bins)[0]
        
        mask = np.ma.masked_greater(num_per_bin, 5).mask
        hist_ETGs = hist_ETGs[mask]
        num_per_bin =num_per_bin[mask]
        bins_ok=bins[mask]
        y = np.divide(hist_ETGs,self.fracsky*Re_BinWidth)
        poiserr = y/np.sqrt(num_per_bin)
        
        if make_plots:
            ax1.errorbar(bins_ok, y, yerr=poiserr, fmt='o',markersize='2',color='red',label='ETGsHC' )
        
        if saved:
            a=np.array([bins_ok,y,poiserr])
            np.savetxt(path+'/'+str(SMmean)+'/Re_ETGs'+str(SMmean)+'cent.txt',a.T)        
        #if compute_SMR:
         #       percE=self.calc_ReMstar(df_HC,SM_LB,'etgs',w=Weights)
        ####### LTGs  according to bayesian probs

        LTGwt =  df_HC.probaSab.values+ df_HC.probaScd.values
        
        Weights = Vmax*LTGwt
        totweight = np.sum(Weights)
        
        hist_LTGs, edges = np.histogram(Re, bins = Re_Bins, weights = Weights)
        num_per_bin = np.histogram(Re, bins = Re_Bins)[0]
        
        mask = np.ma.masked_greater(num_per_bin, 5).mask
        hist_LTGs = hist_LTGs[mask]
        num_per_bin =num_per_bin[mask]
        bins_ok=bins[mask]
        
        y = np.divide(hist_LTGs,self.fracsky*Re_BinWidth)
        poiserr = y/np.sqrt(num_per_bin)
        
        if make_plots:
            ax1.errorbar(bins_ok, y, yerr=poiserr, fmt='d', markersize='2',color='navy', label='LTGsHC' )
            ax1.legend(loc='upper left')
            ax1.set_ylabel('$log_{10}\phi(R_e)$')
            ax1.set_xlabel('$log_{10}Re (kpc)$')
            ax1.set_title('$logM_{star}=$'+str(SMmean))
        if saved:
            a=np.array([bins_ok,y,poiserr])
            np.savetxt(path+'/'+str(SMmean)+'/Re_Lcent'+str(SMmean)+'cent.txt',a.T)        
        
        #plt.legend()
       # plt.savefig('SMF_morph_HC.png')
        #plt.close()
       # if compute_SMR:
        #        percL=self.calc_ReMstar(df_HC,SM_LB,'ltgs',w=Weights)
         #       return percL,percE
        if make_plots:
            ax2.errorbar(bins_ok, y_all, yerr=poiserr_all, linestyle='--', color='black')

        ###### ETGs according to TType
        
        TTypeCutETGs = df_HC.TType <= 0
        df_ETGsTType = df_HC[TTypeCutETGs ]
        
        Re = df_ETGsTType.logReSerExp
        Vmax = df_ETGsTType.Vmaxwt
        print(len(Re), SMmean,'etgs')
        hist_ETGs, edges = np.histogram(Re, bins = Re_Bins, weights = Vmax)
        num_per_bin = np.histogram(Re, bins = Re_Bins)[0]
        #poiserr = hist_ETGs/totweight/np.sqrt(num_per_bin)

        mask = np.ma.masked_greater(num_per_bin, 5).mask
        hist_ETGs = hist_ETGs[mask]
        num_per_bin =num_per_bin[mask]
        bins_ok=bins[mask]
        
        y = np.divide(hist_ETGs,self.fracsky*Re_BinWidth)
        poiserr = y/np.sqrt(num_per_bin)
        
        if make_plots:
            ax2.errorbar(bins_ok, y, yerr=poiserr, label='TType<0', fmt='v', lw=3,color='orange')
        if saved:
            a=np.array([bins_ok,y,poiserr])
            np.savetxt(path+'/'+str(SMmean)+'/Re_ETGs_TT'+str(SMmean)+'cent.txt',a.T)       
        
        if compute_SMR:
            percE=self.calc_ReMstar(df_ETGsTType,SM_LB,'etgs')
            
        ######## LTGs according to TType
        TTypeCutLTGs = df_HC.TType > 0
        
        df_LTGsTType = df_HC[TTypeCutLTGs]
        
        Re = df_LTGsTType.logReSerExp  
        Vmax = df_LTGsTType.Vmaxwt
        print(len(Re), SMmean,'ltgs')
        hist_LTGs, edges = np.histogram(Re, bins = Re_Bins, weights = Vmax)
        num_per_bin = np.histogram(Re, bins = Re_Bins)[0]
        
        mask = np.ma.masked_greater(num_per_bin, 5).mask
        hist_LTGs = hist_LTGs[mask]
        num_per_bin =num_per_bin[mask]
        bins_ok=bins[mask]
        
        y = np.divide(hist_LTGs,self.fracsky*Re_BinWidth)
        poiserr = y/np.sqrt(num_per_bin)        

        
        if make_plots:
            ax2.errorbar(bins_ok, y, yerr=poiserr, label='TType>0', fmt='^',lw=3, color='cyan')
        
            ax2.legend(loc='upper left')
            ax2.set_xlabel('$log_{10}Re (kpc)$')
            ax2.set_title('$logM_{star}=$'+str(SMmean))
            
        if saved:
            a=np.array([bins_ok,y,poiserr])
            np.savetxt(path+'/'+str(SMmean)+'/Re_LTGs_TT'+str(SMmean)+'cent.txt',a.T)  
        if compute_SMR:
            percL=self.calc_ReMstar(df_LTGsTType,SM_LB,'ltgs')
            return percL,percE
        return
        
        
        
        
    def SMFmorphologies_masserrors(self, f,nsteps,SMF_BinWidth=0.1):
        
        
        #fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(16,6))
        path='/home/lorenzo/PhD/data/SDSS/SDSS_Processing/new_catalogs/SMF'
        SMF_LB = 8.5; SMF_UB =13
        SMF_Bins = np.arange(SMF_LB, SMF_UB, SMF_BinWidth)
        
        #select_HC = self.df_cent.probaE > 0
        select_SMRange = (SMF_LB < self.df_cent.MsMendSerExp) & (self.df_cent.MsMendSerExp < SMF_UB)
        index =  select_SMRange
        df_HC = self.df_cent[index]
        
        
        SM_all = np.array(df_HC.MsMendSerExp)
        Vmax_all = np.array(df_HC.Vmaxwt)
        
        
        TTypeCutETGs = df_HC.TType <= 0
        df_ETGsTType = df_HC[TTypeCutETGs]
        SM_ETGs = df_ETGsTType.MsMendSerExp.values
        Vmax_ETGs = df_ETGsTType.Vmaxwt.values
        
        
        
        TTypeCutLTGs = df_HC.TType > 0
        df_LTGsTType = df_HC[TTypeCutLTGs]
        SM_LTGs = df_LTGsTType.MsMendSerExp.values
        Vmax_LTGs = df_LTGsTType.Vmaxwt.values
        
        for i in range(nsteps): #bootstrapping
            
            
            Mstar = np.random.normal(loc=10**SM_all, scale=f*10**SM_all)
            Mstar=np.log10(Mstar)
            
           # print(Mstar[np.ma.masked_invalid(Mstar).mask])
            mask = np.logical_not(np.ma.masked_invalid(Mstar).mask)
            Mstar=Mstar[mask]
            weights = Vmax_all[mask]
            _hist_all_ = np.histogram(Mstar, bins = SMF_Bins, weights = weights)[0]
            num_per_bin_all = np.histogram(Mstar, bins = SMF_Bins)[0]
            hist_all_ = np.divide(_hist_all_,self.fracsky*SMF_BinWidth)
            if i==0:
                hist_all=np.copy(hist_all_)
            else:
                hist_all=np.vstack((hist_all,hist_all_))
                
                ###### ETGs according to TType
                
            df_newETGs=df_HC[mask]
            select_ETGs=df_newETGs.TType<=0
            df_newETGs = df_newETGs[select_ETGs]
            Vmax_ETGs = df_newETGs.Vmaxwt.values
           # Mstar_ETGs = np.random.normal(loc=, scale=0.1*Mstar[TTypeCutETGs])
            _hist_ETGs_ = np.histogram(Mstar[select_ETGs], bins = SMF_Bins, weights = Vmax_ETGs)[0]
            hist_ETGs_ = np.divide(_hist_ETGs_,self.fracsky*SMF_BinWidth)
            num_per_bin_ETGs = np.histogram(Mstar[select_ETGs], bins = SMF_Bins)[0]
            if i==0:
                hist_ETGs=np.copy(hist_ETGs_)
            else:
                hist_ETGs=np.vstack((hist_ETGs,hist_ETGs_))
                
                
                ###### LTGs according to TType
        
            #Mstar_LTGs = np.random.normal(loc=, scale=0.1*Mstar[TTypeCutLTGs])
            df_newLTGs=df_HC[mask]
            select_LTGs=df_newLTGs.TType>0           
            df_newLTGs = df_newLTGs[select_LTGs]
            Vmax_LTGs = df_newLTGs.Vmaxwt.values
            _hist_LTGs_ = np.histogram(Mstar[select_LTGs], bins = SMF_Bins, weights = Vmax_LTGs)[0]
            hist_LTGs_ = np.divide(_hist_LTGs_,self.fracsky*SMF_BinWidth)
            num_per_bin_LTGs = np.histogram(Mstar[select_LTGs], bins = SMF_Bins)[0]
            if i==0:
                hist_LTGs=np.copy(hist_LTGs_)
            else:
                hist_LTGs=np.vstack((hist_LTGs,hist_LTGs_))
                            

  
        _all, edges = np.histogram(SM_all, bins = SMF_Bins, weights = Vmax_all)
        poiserr_all=_all/np.sqrt(num_per_bin_all)
        y_all=np.divide(_all,self.fracsky*SMF_BinWidth)
        _ETGs, edges = np.histogram(SM_ETGs, bins = SMF_Bins, weights = Vmax_ETGs)
        y_ETGs=np.divide(_ETGs,self.fracsky*SMF_BinWidth)
        poiserr_ETGs=_ETGs/np.sqrt(num_per_bin_ETGs)
        _LTGs, edges = np.histogram(SM_LTGs, bins = SMF_Bins, weights = Vmax_LTGs)
        y_LTGs=np.divide(_LTGs,self.fracsky*SMF_BinWidth)
        poiserr_LTGs=_LTGs/np.sqrt(num_per_bin_LTGs)
        
        perc_all =np.array(list(map(lambda v: [v[1],v[2]-v[1],v[1]-v[0]], zip(*np.percentile(hist_all.T,[16,50,84],axis=1)))))
        phi_all,up_all,low_all=np.array(list(zip(*perc_all)))
        err_all = np.maximum(up_all,low_all)
        err_all = np.sqrt(err_all**2+poiserr_all**2)
        
        perc_ETGs =np.array(list(map(lambda v: [v[1],v[2]-v[1],v[1]-v[0]], zip(*np.percentile(hist_ETGs.T,[16,50,84],axis=1)))))
        phi_ETGs,up_ETGs,low_ETGs=np.array(list(zip(*perc_ETGs)))
        err_ETGs = np.maximum(up_ETGs,low_ETGs)
        err_ETGs = np.sqrt(err_ETGs**2+poiserr_ETGs**2)
        perc_LTGs =np.array(list(map(lambda v: [v[1],v[2]-v[1],v[1]-v[0]], zip(*np.percentile(hist_LTGs.T,[16,50,84],axis=1)))))
        phi_LTGs,up_LTGs,low_LTGs=np.array(list(zip(*perc_LTGs)))
        err_LTGs = np.maximum(up_LTGs,low_LTGs)     
        err_LTGs = np.sqrt(err_LTGs**2+poiserr_LTGs**2)
        
        print(phi_all)
        fig1=plt.errorbar(SMF_Bins[1:]-SMF_BinWidth/2., phi_all ,yerr=err_all, label='all centrals', fmt='o', color='black')
        fig2=plt.errorbar(SMF_Bins[1:]-SMF_BinWidth/2., phi_ETGs ,yerr=err_ETGs, label='ETGs', fmt='^', color='red')
        fig3=plt.errorbar(SMF_Bins[1:]-SMF_BinWidth/2., phi_LTGs ,yerr=err_LTGs, label='LTGs', fmt='v', color='blue')
        #fig1=plt.plot(SMF_Bins[1:]-SMF_BinWidth/2.,np.log10(phi_all),  label='all centrals', marker='o')
        #fig2=plt.plot(SMF_Bins[1:]-SMF_BinWidth/2., np.log10(phi_ETGs),  label='ETGs', marker='^')
        #fig3=plt.plot(SMF_Bins[1:]-SMF_BinWidth/2., np.log10(phi_LTGs), label='LTGs', marker='v')        
        leg=plt.legend()
        a_all=np.array([SMF_Bins[6:]-SMF_BinWidth/2.,phi_all[5:],err_all[5:]]).T
        a_ETGs=np.array([SMF_Bins[6:]-SMF_BinWidth/2., phi_ETGs[5:], err_ETGs[5:]]).T
        a_LTGs=np.array([SMF_Bins[6:]-SMF_BinWidth/2., phi_LTGs[5:], err_LTGs[5:]]).T
        
        np.savetxt(path+'/SMF_all.txt',a_all)
        np.savetxt(path+'/SMF_ETGS_TT.txt', a_ETGs)
        np.savetxt(path+'/SMF_LTGS_TT.txt', a_LTGs)
        
        return fig1,fig2,fig3
        
    def SMF_morphologies(self, SMF_BinWidth=0.1):
        
        
        fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(16,6))
        
        SMF_LB = 9; SMF_UB =12.5
        SMF_Bins = np.arange(SMF_LB, SMF_UB, SMF_BinWidth)
        
        select_HC = self.df_cent.probaE > 0
        select_SMRange = (SMF_LB < self.df_cent.MsMendSerExp) & (self.df_cent.MsMendSerExp < SMF_UB)
        index = select_HC & select_SMRange
        df_HC = self.df_cent[index]
        
        
        SM = np.array(df_HC.MsMendSerExp)
        Vmax = np.array(df_HC.Vmaxwt)
        
        
        ax1.errorbar(SMF_Bins[1:]-SMF_BinWidth/2., y, yerr=poiserr, fmt='o',markersize='2',color='red',label='ETGsHC' )
        
        a=np.array([SMF_Bins[1:]-SMF_BinWidth/2.,y,poiserr])
        np.savetxt('./SMF/SMF_ETGS_HCoriginal.txt',a.T)
        ####### LTGs  according to bayesian probs

        LTGwt =  df_HC.probaSab[select_SMRange] + df_HC.probaScd[select_SMRange]
        
        Weights = Vmax*LTGwt
        totweight = np.sum(Weights)
        
        hist_LTGs, edges = np.histogram(SM, bins = SMF_Bins, weights = Weights)
        num_per_bin = np.histogram(SM, bins = SMF_Bins)[0]
        
        
        y = np.divide(hist_LTGs,self.fracsky*SMF_BinWidth)
        poiserr = y/num_per_bin
        ax1.errorbar(SMF_Bins[1:]-SMF_BinWidth/2., y, yerr=poiserr, fmt='d', markersize='2',color='navy', label='LTGsHC' )
        ax1.legend(loc='lower left')
        ax1.set_ylabel('$log_{10}\phi(M_{star})$')
        ax1.set_xlabel('$log_{10}M_{star}/M_{\odot}$')
        
        a=np.array([SMF_Bins[1:]-SMF_BinWidth/2.,y,poiserr])
        np.savetxt('./SMF/SMF_LTGS_HCoriginal.txt',a.T)        
        
        #plt.legend()
       # plt.savefig('SMF_morph_HC.png')
        #plt.close()
        
        
        ax2.errorbar(SMF_Bins[1:]-SMF_BinWidth/2., y_all, yerr=poiserr_all, linestyle='--', color='black')

        ###### ETGs according to TType
        
        TTypeCutETGs = df_HC.TType <= 0
        
        df_ETGsTType = df_HC[TTypeCutETGs]
        
        SM = df_ETGsTType.MsMendSerExp
        Vmax = df_ETGsTType.Vmaxwt
        
        hist_ETGs, edges = np.histogram(SM, bins = SMF_Bins, weights = Vmax)
        num_per_bin = np.histogram(SM, bins = SMF_Bins)[0]
        #poiserr = hist_ETGs/totweight/np.sqrt(num_per_bin)
        
        yETGs = np.divide(hist_ETGs,self.fracsky*SMF_BinWidth)
        poiserr = yETGs/num_per_bin
        ax2.errorbar(SMF_Bins[1:]-SMF_BinWidth/2., yETGs, yerr=poiserr, label='TType<0', fmt='v', lw=3,color='orange')
        
        a=np.array([SMF_Bins[1:]-SMF_BinWidth/2.,yETGs,poiserr])
        np.savetxt('./SMF/SMF_ETGS_TToriginal.txt',a.T)       
        
        ######## LTGs according to TType
        TTypeCutLTGs = df_HC.TType > 0
        
        df_LTGsTType = df_HC[TTypeCutLTGs]
        
        SM = df_LTGsTType.MsMendSerExp
        Vmax = df_LTGsTType.Vmaxwt
        
        hist_LTGs, edges = np.histogram(SM, bins = SMF_Bins, weights = Vmax)
        num_per_bin = np.histogram(SM, bins = SMF_Bins)[0]
        
        yLTGs = np.divide(hist_LTGs,self.fracsky*SMF_BinWidth)
        poiserr = yLTGs/num_per_bin
        ax2.errorbar(SMF_Bins[1:]-SMF_BinWidth/2., yLTGs, yerr=poiserr, label='TType>0', fmt='^',lw=3, color='cyan')
        
        ax2.legend(loc='lower left')
        ax2.set_xlabel('$log_{10}M_{star}/M_{\odot}$')
        a=np.array([SMF_Bins[1:]-SMF_BinWidth/2.,yLTGs,poiserr])
        np.savetxt('./SMF/SMF_LTGS_TToriginal.txt',a.T)  
        
        a = np.array([SMF_Bins[1:]-SMF_BinWidth/2.,yLTGs/y_all])
        np.savetxt('./SMF/fraction_LTGs_vs_Mstar.txt',a.T)
        plt.plot(SMF_Bins[1:]-SMF_BinWidth/2.,yLTGs/y_all)
        #plt.legend()
        #plt.savefig('SMF_TType.png')
        #plt.close()
        #SDSS_Legend = plt.legend(handles = [ETG_plot, ETG_plotTT, LTG_plot, LTG_plotTT], loc = 'lower left')
        #ax = plt.gca().add_artist(SDSS_Legend)
        return #ETG_plot,ETG_plotTT, LTG_plot,LTG_plotTT, ax        
        
        
        
    def SMF_SFR(self, SMF_BinWidth=0.1):
        
        
        fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(16,6))
        
        SMF_LB = 9; SMF_UB =12.5
        SMF_Bins = np.arange(SMF_LB, SMF_UB, SMF_BinWidth)
        
        select_HC = self.df_cent.probaE > 0
        select_SMRange = (SMF_LB < self.df_cent.MsMendSerExp) & (self.df_cent.MsMendSerExp < SMF_UB)
        index = select_HC & select_SMRange
        df_HC = self.df_cent[index]
        
        
        SM = np.array(df_HC.MsMendSerExp)
        Vmax = np.array(df_HC.Vmaxwt)
        
        ######## all galaxies
        Weights = Vmax
        

        hist_all, edges = np.histogram(SM, bins = SMF_Bins, weights = Weights)
        y_all=np.divide(hist_all,self.fracsky*SMF_BinWidth)
        num_per_bin_all = np.histogram(SM, bins = SMF_Bins)[0]
        poiserr_all = y_all/num_per_bin_all
        
        ax1.errorbar(SMF_Bins[1:]-SMF_BinWidth/2., y_all, yerr=poiserr_all, linestyle='--', color='black')
        a=np.array([SMF_Bins[1:]-SMF_BinWidth/2.,y_all,poiserr_all])
        np.savetxt('./SMF/SMF_alloriginal.txt',a.T)        
        ####### ETGs according to bayesian probs

        ETGwt = df_HC.probaEll[select_SMRange] + df_HC.probaS0[select_SMRange]
        
        Weights = Vmax*ETGwt
        totweight = np.sum(Weights)
        
        hist_ETGs, edges = np.histogram(SM, bins = SMF_Bins, weights = Weights)
        num_per_bin = np.histogram(SM, bins = SMF_Bins)[0]
        
        
        y = np.divide(hist_ETGs,self.fracsky*SMF_BinWidth)
        poiserr = y/num_per_bin
        
        ax1.errorbar(SMF_Bins[1:]-SMF_BinWidth/2., y, yerr=poiserr, fmt='o',markersize='2',color='red',label='ETGsHC' )
        
        a=np.array([SMF_Bins[1:]-SMF_BinWidth/2.,y,poiserr])
        np.savetxt('./SMF/SMF_ETGS_HCoriginal.txt',a.T)
        ####### LTGs  according to bayesian probs

        LTGwt =  df_HC.probaSab[select_SMRange] + df_HC.probaScd[select_SMRange]
        
        Weights = Vmax*LTGwt
        totweight = np.sum(Weights)
        
        hist_LTGs, edges = np.histogram(SM, bins = SMF_Bins, weights = Weights)
        num_per_bin = np.histogram(SM, bins = SMF_Bins)[0]
        
        
        y = np.divide(hist_LTGs,self.fracsky*SMF_BinWidth)
        poiserr = y/num_per_bin
        ax1.errorbar(SMF_Bins[1:]-SMF_BinWidth/2., y, yerr=poiserr, fmt='d', markersize='2',color='navy', label='LTGsHC' )
        ax1.legend(loc='lower left')
        ax1.set_ylabel('$log_{10}\phi(M_{star})$')
        ax1.set_xlabel('$log_{10}M_{star}/M_{\odot}$')
        
        a=np.array([SMF_Bins[1:]-SMF_BinWidth/2.,y,poiserr])
        np.savetxt('./SMF/SMF_LTGS_HCoriginal.txt',a.T)        
        
        #plt.legend()
       # plt.savefig('SMF_morph_HC.png')
        #plt.close()
        
        
        ax2.errorbar(SMF_Bins[1:]-SMF_BinWidth/2., y_all, yerr=poiserr_all, linestyle='--', color='black')

        ###### ETGs according to TType
        
        TTypeCutETGs = df_HC.TType <= 0
        
        df_ETGsTType = df_HC[TTypeCutETGs]
        
        SM = df_ETGsTType.MsMendSerExp
        Vmax = df_ETGsTType.Vmaxwt
        
        hist_ETGs, edges = np.histogram(SM, bins = SMF_Bins, weights = Vmax)
        num_per_bin = np.histogram(SM, bins = SMF_Bins)[0]
        #poiserr = hist_ETGs/totweight/np.sqrt(num_per_bin)
        
        yETGs = np.divide(hist_ETGs,self.fracsky*SMF_BinWidth)
        poiserr = yETGs/num_per_bin
        ax2.errorbar(SMF_Bins[1:]-SMF_BinWidth/2., yETGs, yerr=poiserr, label='TType<0', fmt='v', lw=3,color='orange')
        
        a=np.array([SMF_Bins[1:]-SMF_BinWidth/2.,yETGs,poiserr])
        np.savetxt('./SMF/SMF_ETGS_TToriginal.txt',a.T)       
        
        ######## LTGs according to TType
        TTypeCutLTGs = df_HC.TType > 0
        
        df_LTGsTType = df_HC[TTypeCutLTGs]
        
        SM = df_LTGsTType.MsMendSerExp
        Vmax = df_LTGsTType.Vmaxwt
        
        hist_LTGs, edges = np.histogram(SM, bins = SMF_Bins, weights = Vmax)
        num_per_bin = np.histogram(SM, bins = SMF_Bins)[0]
        
        yLTGs = np.divide(hist_LTGs,self.fracsky*SMF_BinWidth)
        poiserr = yLTGs/num_per_bin
        ax2.errorbar(SMF_Bins[1:]-SMF_BinWidth/2., yLTGs, yerr=poiserr, label='TType>0', fmt='^',lw=3, color='cyan')
        
        ax2.legend(loc='lower left')
        ax2.set_xlabel('$log_{10}M_{star}/M_{\odot}$')
        a=np.array([SMF_Bins[1:]-SMF_BinWidth/2.,yLTGs,poiserr])
        np.savetxt('./SMF/SMF_LTGS_TToriginal.txt',a.T)  
        
        a = np.array([SMF_Bins[1:]-SMF_BinWidth/2.,yLTGs/y_all])
        np.savetxt('./SMF/fraction_LTGs_vs_Mstar.txt',a.T)
        plt.plot(SMF_Bins[1:]-SMF_BinWidth/2.,yLTGs/y_all)
        #plt.legend()
        #plt.savefig('SMF_TType.png')
        #plt.close()
        #SDSS_Legend = plt.legend(handles = [ETG_plot, ETG_plotTT, LTG_plot, LTG_plotTT], loc = 'lower left')
        #ax = plt.gca().add_artist(SDSS_Legend)
        return 
        
        
    def SMF(self, SMF_BinWidth = 0.1):
        SMF_LB = 9; SMF_UB = 12.5
        SMF_Bins = np.arange(SMF_LB, SMF_UB, SMF_BinWidth)

        ######## centrals
        
        SM = np.array(self.df_cent.MsMendSerExp[(SMF_LB < self.df_cent.MsMendSerExp) & (self.df_cent.MsMendSerExp < SMF_UB)])
        Vmax = np.array(self.df_cent.Vmaxwt[(SMF_LB < self.df_cent.MsMendSerExp) & (self.df_cent.MsMendSerExp < SMF_UB)])

        Weights = Vmax
        Weightsum = np.sum(Vmax)
        totVmax = Weightsum/self.fracsky

        hist_cent, edges = np.histogram(SM, bins = SMF_Bins, weights = Vmax)
        Cent_Plot = plt.plot(SMF_Bins[1:], np.log10(np.divide(hist_cent, self.fracsky*SMF_BinWidth)), 'bx', label = "SDSS: Central")[0]

        
        
        ######### Satellites
        SM = np.array(self.df_sat.MsMendSerExp[(SMF_LB < self.df_sat.MsMendSerExp) & (self.df_sat.MsMendSerExp < SMF_UB)])
        Vmax = np.array(self.df_sat.Vmaxwt[(SMF_LB < self.df_sat.MsMendSerExp) & (self.df_sat.MsMendSerExp < SMF_UB)])

        Weights = Vmax
        Weightsum = np.sum(Vmax)
        totVmax = Weightsum/self.fracsky

        hist_sat, edges = np.histogram(SM, bins = SMF_Bins, weights = Vmax)
        Sat_Plot = plt.plot(SMF_Bins[1:], np.log10(np.divide(hist_sat, self.fracsky*SMF_BinWidth)), 'rx', label = "SDSS: Satellite")[0]
        #hist_sat=hist_z-hist_cent
        #Sat_Plot = plt.plot(SMF_Bins[1:], np.log10(np.divide(hist_sat, self.fracsky*SMF_BinWidth)),'rx', label = "SDSS: Satellite")[0]
        
        
        
        ##### Cen + Sat
        Tot_Plot = plt.plot(SMF_Bins[1:], np.log10(np.divide(hist_cent + hist_sat, self.fracsky*SMF_BinWidth)), 'gx', label = "SDSS: cen+sat")[0]

        
        
        ###### Total
        SM = np.array(self.df_z.MsMendSerExp[(SMF_LB < self.df_z.MsMendSerExp) & (self.df_z.MsMendSerExp < SMF_UB)])
        Vmax = np.array(self.df_z.Vmaxwt[(SMF_LB < self.df_z.MsMendSerExp) & (self.df_z.MsMendSerExp < SMF_UB)])

        Weights = Vmax
        Weightsum = np.sum(Vmax)
        totVmax = Weightsum/self.fracsky

        hist_z, edges = np.histogram(SM, bins = SMF_Bins, weights = Vmax)
        Z_Plot = plt.plot(SMF_Bins[1:], np.log10(np.divide(hist_z, self.fracsky*SMF_BinWidth)), 'mo',markersize=3, label = "SDSS: total")[0]


        
        SDSS_Legend = plt.legend(handles = [Cent_Plot,Z_Plot, Sat_Plot,Tot_Plot ], loc = 1)
        #ax = plt.gca().add_artist(SDSS_Legend)
        
        return Tot_Plot, Sat_Plot, Cent_Plot,Z_Plot#, ax
    
    def FracPlot(self, fig, SM_Cut = 10):
        #ActualCut
        df_sat_frac = self.df_sat[(self.AnalyticHaloMass_min < self.df_sat.newMhaloL) & (self.df_sat.newMhaloL < self.AnalyticHaloMass_max) & (self.df_sat.MsMendSerExp > SM_Cut)]

        #MassCut
        Data_mh = np.array(df_sat_frac.newMhaloL)
        Data_vmax = np.array(df_sat_frac.Vmaxwt)

        #Weighted count for SDSS data
        VmaxSum = np.sum(Data_vmax)
        totVmax = VmaxSum/self.fracsky
        Y, X = np.histogram(Data_mh, bins = self.AnalyticHaloMass, weights = Data_vmax, density = True)

        #UpperCut
        df_sat_frac = self.df_sat[(self.AnalyticHaloMass_min < self.df_sat.newMhaloL) & (self.df_sat.newMhaloL < self.AnalyticHaloMass_max) & (self.df_sat.MsMendSerExp > SM_Cut + 0.1)]

        #MassCut
        Data_mh = np.array(df_sat_frac.newMhaloL)
        Data_vmax = np.array(df_sat_frac.Vmaxwt)

        #Weighted count for SDSS data
        VmaxSum = np.sum(Data_vmax)
        totVmax = VmaxSum/self.fracsky
        Y_U, X_U = np.histogram(Data_mh, bins = self.AnalyticHaloMass, weights = Data_vmax, density = True)

        #Lowercut
        df_sat_frac = self.df_sat[(self.AnalyticHaloMass_min < self.df_sat.newMhaloL) & (self.df_sat.newMhaloL < self.AnalyticHaloMass_max) & (self.df_sat.MsMendSerExp > SM_Cut - 0.1)]

        #MassCut
        Data_mh = np.array(df_sat_frac.newMhaloL)
        Data_vmax = np.array(df_sat_frac.Vmaxwt)

        #Weighted count for SDSS data
        VmaxSum = np.sum(Data_vmax)
        totVmax = VmaxSum/self.fracsky
        Y_L, X_L = np.histogram(Data_mh, bins = self.AnalyticHaloMass, weights = Data_vmax, density = True)

        Yplot_U = np.maximum(Y_U, Y_L) - Y
        Yplot_L = Y - np.minimum(Y_U, Y_L)
        Y_Err = np.vstack((Yplot_L, Yplot_U))

        fig.fill_between(X[1:], Y - Y_Err[0], Y + Y_Err[1], alpha = 0.5, color = 'tab:gray')
        return X[1:], Y
    
    def NoFracPlot(self, fig, SM_Cut = 10):
        #ActualCut
        df_sat_frac = self.df_sat[(self.AnalyticHaloMass_min < self.df_sat.newMhaloL) & (self.df_sat.newMhaloL < self.AnalyticHaloMass_max) & (self.df_sat.MsMendSerExp > SM_Cut)]

        #MassCut
        Data_mh = np.array(df_sat_frac.newMhaloL)
        Data_vmax = np.array(df_sat_frac.Vmaxwt)

        #Weighted count for SDSS data
        VmaxSum = np.sum(Data_vmax)
        totVmax = VmaxSum/self.fracsky
        Y, X = np.histogram(Data_mh, bins = self.AnalyticHaloMass, weights = Data_vmax, density = False)
        
        
        #UpperCut
        df_sat_frac = self.df_sat[(self.AnalyticHaloMass_min < self.df_sat.newMhaloL) & (self.df_sat.newMhaloL < self.AnalyticHaloMass_max) & (self.df_sat.MsMendSerExp > SM_Cut + 0.1)]

        #MassCut
        Data_mh = np.array(df_sat_frac.newMhaloL)
        Data_vmax = np.array(df_sat_frac.Vmaxwt)

        #Weighted count for SDSS data
        VmaxSum = np.sum(Data_vmax)
        totVmax = VmaxSum/self.fracsky
        Y_U, X_U = np.histogram(Data_mh, bins = self.AnalyticHaloMass, weights = Data_vmax, density = False)
        
        #Lowercut
        df_sat_frac = self.df_sat[(self.AnalyticHaloMass_min < self.df_sat.newMhaloL) & (self.df_sat.newMhaloL < self.AnalyticHaloMass_max) & (self.df_sat.MsMendSerExp > SM_Cut - 0.1)]

        #MassCut
        Data_mh = np.array(df_sat_frac.newMhaloL)
        Data_vmax = np.array(df_sat_frac.Vmaxwt)

        #Weighted count for SDSS data
        VmaxSum = np.sum(Data_vmax)
        totVmax = VmaxSum/self.fracsky
        Y_L, X_L = np.histogram(Data_mh, bins = self.AnalyticHaloMass, weights = Data_vmax, density = False)
        
        #Combining to form range
        Yplot_U = np.maximum(Y_U, Y_L) - Y
        Yplot_L = Y - np.minimum(Y_U, Y_L)
        Y_Err = np.vstack((Yplot_L, Yplot_U))

        fig.fill_between(X[1:], np.log10(np.divide(Y - Y_Err[0], self.fracsky*self.AnalyticHaloBin)), np.log10(np.divide(Y + Y_Err[1], self.fracsky*self.AnalyticHaloBin)), alpha = 0.5, color = 'tab:gray')
        return X[1:], Y

    
    
    def EllipticalPlot(self):
        Cent_SM = np.array(self.df_cent.MsMendSerExp[self.df_cent.MsMendSerExp > 10])
        #Cent_PE = np.array(self.df_cent.probell[self.df_cent.MsMendSerExp > 10])
        Cent_PE = np.array(self.df_cent.probe[self.df_cent.MsMendSerExp > 10] )#+ self.df_cent.probell[self.df_cent.MsMendSerExp > 10])
        Cent_Vmax = np.array(self.df_cent.Vmaxwt[self.df_cent.MsMendSerExp > 10])
        print(np.max(Cent_PE), np.min(Cent_PE))
        Cent_PE[Cent_PE < 0] = 0

        X_Plot = np.arange(10, 12.5, 0.1)
        Bins = np.digitize(Cent_SM, X_Plot)

        Y_Plot = []
        for i in range(len(X_Plot)):
            Cent_Vmax[Bins == i] = np.divide(Cent_Vmax[Bins == i ],np.sum(Cent_Vmax[Bins == i]))
            Y_Plot.append(np.sum(Cent_PE[Bins==i]*Cent_Vmax[Bins == i]))

        #Y_Plot = np.array([ np.sum(Cent_PE[Bins == i] * (Cent_Vmax[Bins == i]/np.sum(Cent_Vmax[Bins == i])))/np.size(Cent_PE[Bins == i]) for i in range(len(X_Plot))])
        #Y_Plot = np.array([ (np.sum(Cent_PE[Bins == i] * Cent_Vmax[Bins == i])/(np.sum(Cent_Vmax[Bins == i]))) for i in range(len(X_Plot))])

        #Y_Plot = np.array([ (np.sum(Cent_PE[Bins == i])/(np.size(Cent_PE[Bins == i]))) for i in range(len(X_Plot))])
        """
        Y, X = np.histogram(Cent_SM, bins = X_Plot, weights = Cent_PE*Cent_Vmax)
        Bins = np.digitize(Cent_SM, X)
        Div = np.array([(np.sum(Cent_Vmax[Bins == i])) for i in range(1,len(X))])
        Y = np.divide(Y,Div)
        """

        plot = plt.plot(X_Plot[1:], Y_Plot[1:], "x", label = "P(E) (SDSS)")
        return plot
        #plot = plt.plot(X[1:], Y)

    def SMF_Elliptical(self):
        SMF_BinWidth = 0.1
        SMF_LB = 9; SMF_UB = 12.5
        SMF_Bins = np.arange(SMF_LB, SMF_UB, SMF_BinWidth)

        SM = np.array(self.df_cent.MsMendSerExp[(SMF_LB < self.df_cent.MsMendSerExp) & (self.df_cent.MsMendSerExp < SMF_UB)])
        Vmax = np.array(self.df_cent.Vmaxwt[(SMF_LB < self.df_cent.MsMendSerExp) & (self.df_cent.MsMendSerExp < SMF_UB)])
        Prob_E = np.array(self.df_cent.probe[(SMF_LB < self.df_cent.MsMendSerExp) & (self.df_cent.MsMendSerExp < SMF_UB)])
        Prob_E[Prob_E < 0.5] = 0
        Prob_E[Prob_E > 0.5] = 1

        Weights = Vmax*Prob_E
        Weightsum = np.sum(Vmax)
        totVmax = Weightsum/self.fracsky


        hist, edges = np.histogram(SM, bins = SMF_Bins, weights = Weights)
        plot = plt.plot(SMF_Bins[1:], np.log10(np.divide(hist, self.fracsky*SMF_BinWidth)), 'rx', label = "SDSS:Elliptical")

        hist, edges = np.histogram(SM, bins = SMF_Bins, weights = Vmax)
        plot = plt.plot(SMF_Bins[1:], np.log10(np.divide(hist, self.fracsky*SMF_BinWidth)), 'bx', label = "SDSS:Total")

        return plot

    def SMF_Fixed_HM(self, fig1, fig2, Min, Max, Bin):
        SMF_BinWidth = 0.1
        SMF_LB = 9; SMF_UB = 12.5
        SMF_Bins = np.arange(SMF_LB, SMF_UB, SMF_BinWidth)


        SM = np.array(self.df_sat.MsMendSerExp[(SMF_LB < self.df_sat.MsMendSerExp) & (self.df_sat.MsMendSerExp < SMF_UB)])
        Vmax = np.array(self.df_sat.Vmaxwt[(SMF_LB < self.df_sat.MsMendSerExp) & (self.df_sat.MsMendSerExp < SMF_UB)])
        HM = np.array(self.df_sat.newMhaloL[(SMF_LB < self.df_sat.MsMendSerExp) & (self.df_sat.MsMendSerExp < SMF_UB)])

        for i, HM_Bin in enumerate(np.arange(Min, Max, Bin)):
            HM_Mask = ma.masked_inside(HM, HM_Bin, Max).mask#HM_Bin+0.5).mask
            Weightsum = np.sum(Vmax[HM_Mask])
            totVmax = Weightsum/self.fracsky
            hist, edges = np.histogram(SM[HM_Mask], bins = SMF_Bins, weights = Vmax[HM_Mask])
            fig1.plot(SMF_Bins[1:], np.log10(np.divide(hist, self.fracsky*SMF_BinWidth)), 'xC{}'.format(i), label = "SDSS: {}<HM<{}".format(HM_Bin, HM_Bin+0.5))

        for i, HM_Bin in enumerate(np.arange(Min, Max, Bin)):
            HM_Mask = ma.masked_inside(HM, HM_Bin, HM_Bin+0.5).mask
            Weightsum = np.sum(Vmax[HM_Mask])
            totVmax = Weightsum/self.fracsky
            hist, edges = np.histogram(SM[HM_Mask], bins = SMF_Bins, weights = Vmax[HM_Mask])
            fig2.plot(SMF_Bins[1:], np.log10(np.divide(hist, self.fracsky*SMF_BinWidth)), 'xC{}'.format(i), label = "SDSS: {}<HM<{}".format(HM_Bin, HM_Bin+0.5))

        Weightsum = np.sum(Vmax)
        totVmax = Weightsum/self.fracsky
        hist, edges = np.histogram(SM, bins = SMF_Bins, weights = Vmax)
        #fig1.plot(SMF_Bins[1:], np.log10(np.divide(hist, self.fracsky*SMF_BinWidth)), 'bx', label = "SDSS:Total")
        fig2.plot(SMF_Bins[1:], np.log10(np.divide(hist, self.fracsky*SMF_BinWidth)), 'bx', label = "SDSS:Total")

    def SMHM_Sat_Cent(self, fig):
        HM = np.array(self.df_sat.newMhaloL)
        Bins = np.digitize(HM, self.AnalyticHaloMass)
        Y_Plot = []
        Yplot_Err = []
        for i in range(0,len(self.AnalyticHaloMass)):
            SM = np.array(self.df_sat.MsMendSerExp)[Bins == i]
            Vmax = np.array(self.df_sat.Vmaxwt)[Bins == i]
            VmaxSum = np.sum(Vmax)
            weighted_stats = DescrStatsW(SM, weights=Vmax, ddof=0)
            Y_Plot.append(weighted_stats.mean)
            Yplot_Err.append(weighted_stats.std)
            """Y_Plot.append(np.mean(SM))
            Yplot_Err.append(np.std(SM))"""
        Y_Plot = np.array(Y_Plot)
        Yplot_Err = np.array(Yplot_Err)
        fig.fill_between(np.array(self.AnalyticHaloMass), Y_Plot + Yplot_Err, Y_Plot - Yplot_Err, color = "tab:gray", alpha = 0.5, label = "SDSS CentHalo")

    
    def Old_SMF(self, SMF_BinWidth = 0.1):
        SMF_LB = 9; SMF_UB = 13
        SMF_Bins = np.arange(SMF_LB, SMF_UB, SMF_BinWidth)



        logMstar_cent = np.array( self.df_cent.MsMendSerExp[(SMF_LB < self.df_cent.MsMendSerExp) & (self.df_cent.MsMendSerExp < SMF_UB)])
        Vmax_cent = np.array(self.df_cent.Vmaxwt[(SMF_LB < self.df_cent.MsMendSerExp) & (self.df_cent.MsMendSerExp < SMF_UB)])
        VmaxSum = np.sum(Vmax_cent)
        totVmax=VmaxSum/self.fracsky

        #Bins and calculates VmaxWt per bin and number per bin
        Binned = np.digitize(logMstar_cent, SMF_Bins)
        Totals_Cent = (np.array([ np.divide(np.sum(Vmax_cent[Binned == i]), VmaxSum*SMF_BinWidth) for i in range (0, len(SMF_Bins)) ])[1:])*totVmax
        Totals_Cent_num = np.array([ np.size(Vmax_cent[Binned == i]) for i in range (0, len(SMF_Bins)) ])[1:]


        #possion err
        Totals_Cent_err = (np.divide(Totals_Cent,np.sqrt(Totals_Cent_num)))

        TotalsPlot_Cent = np.log10(np.vstack((Totals_Cent, Totals_Cent-Totals_Cent_err, Totals_Cent+Totals_Cent_err)))
        TotalsPlot_Cent[1:] = np.abs(TotalsPlot_Cent[1:]-TotalsPlot_Cent[0])
        Cent_fig = plt.errorbar(SMF_Bins[1:], TotalsPlot_Cent[0], yerr = TotalsPlot_Cent[1:], fmt ='bx', label = "Central")



        logMstar_Sat = np.array( self.df_sat.MsMendSerExp[(SMF_LB < self.df_sat.MsMendSerExp) & (self.df_sat.MsMendSerExp < SMF_UB)])
        Vmax_Sat = np.array(self.df_sat.Vmaxwt[(SMF_LB < self.df_sat.MsMendSerExp) & (self.df_sat.MsMendSerExp < SMF_UB)])
        VmaxSum = np.sum(Vmax_Sat)
        totVmax=VmaxSum/self.fracsky

        #Bins and calculates VmaxWt per bin and number per bin
        Binned = np.digitize(logMstar_Sat, SMF_Bins)
        Totals_Sat = (np.array([ np.divide(np.sum(Vmax_Sat[Binned == i]), VmaxSum*SMF_BinWidth) for i in range (0, len(SMF_Bins)) ])[1:])*totVmax
        Totals_Sat_num = np.array([ np.size(Vmax_Sat[Binned == i]) for i in range (0, len(SMF_Bins)) ])[1:]

        #possion err
        Totals_Sat_err = (np.divide(Totals_Sat,np.sqrt(Totals_Sat_num)))

        TotalsPlot_Sat = np.log10(np.vstack((Totals_Sat, Totals_Sat-Totals_Sat_err, Totals_Sat+Totals_Sat_err)))
        TotalsPlot_Sat[1:] = np.abs(TotalsPlot_Sat[1:]-TotalsPlot_Sat[0])
        Sat_fig = plt.errorbar(SMF_Bins[1:], TotalsPlot_Sat[0], yerr = TotalsPlot_Sat[1:],fmt ='rx', label = "Satilite")



        logMstar_Tot = np.append(np.array( self.df_cent.MsMendSerExp[(SMF_LB < self.df_cent.MsMendSerExp) & (self.df_cent.MsMendSerExp < SMF_UB)]), np.array( self.df_sat.MsMendSerExp[(SMF_LB < self.df_sat.MsMendSerExp) & (self.df_sat.MsMendSerExp < SMF_UB)]))
        Vmax_Tot = np.append(np.array(self.df_cent.Vmaxwt[(SMF_LB < self.df_cent.MsMendSerExp) & (self.df_cent.MsMendSerExp < SMF_UB)]), np.array(self.df_sat.Vmaxwt[(SMF_LB < self.df_sat.MsMendSerExp) & (self.df_sat.MsMendSerExp < SMF_UB)]))
        VmaxSum = np.sum(Vmax_Tot)
        totVmax=VmaxSum/self.fracsky

        #Bins and calculates VmaxWt per bin and number per bin
        Binned = np.digitize(logMstar_Tot, SMF_Bins)
        Totals_Tot = (np.array([ np.divide(np.sum(Vmax_Tot[Binned == i]), VmaxSum*SMF_BinWidth) for i in range (0, len(SMF_Bins)) ])[1:])*totVmax
        Totals_Tot_num = np.array([ np.size(Vmax_Tot[Binned == i]) for i in range (0, len(SMF_Bins)) ])[1:]

        #possion err
        Totals_Tot_err = (np.divide(Totals_Tot,np.sqrt(Totals_Tot_num)))

        TotalsPlot_Tot = np.log10(np.vstack((Totals_Tot, Totals_Tot-Totals_Tot_err, Totals_Tot+Totals_Tot_err)))
        TotalsPlot_Tot[1:] = np.abs(TotalsPlot_Tot[1:]-TotalsPlot_Tot[0])
        Tot_fig = plt.errorbar(SMF_Bins[1:], TotalsPlot_Tot[0], yerr = TotalsPlot_Tot[1:],fmt ='gx', label = "Total")
        SDSS_Legend = plt.legend(handles = [Tot_fig, Sat_fig, Cent_fig], loc = 1)
        ax = plt.gca().add_artist(SDSS_Legend)

        np.savetxt("./Bernardi_SDSS/SMF.dat", np.vstack((SMF_Bins[1:], TotalsPlot_Tot[0],  TotalsPlot_Tot[1:], TotalsPlot_Sat[0], TotalsPlot_Sat[1:], TotalsPlot_Cent[0], TotalsPlot_Cent[1:])))


        return Cent_fig, Sat_fig, Tot_fig, ax
