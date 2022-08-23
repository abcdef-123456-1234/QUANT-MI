from scipy.stats import rankdata
import scipy as sp
import numpy as np
import pandas as pd
from numpy import abs

def linreg(x,B):
    slope, intercept, rvalue, pvalue, stderr = sp.stats.linregress(x,B)
    return([slope, intercept, rvalue, pvalue, stderr])


class Alpha_191:

    def __init__(self,  price, date, benchmark_price=None):
        self.date = date
        #price['prev_close'] = price[['code','close']].groupby('code').shift()
        #benchmark_price = QA_fetch_index_day_adv('000300', self.end_date, self.date).data.reset_index()[['date','code','open','close','low','high','amount','volume']].set_index(['date','code']).to_panel()
        ###分别取开盘价，收盘价，最高价，最低价，最低价，均价，成交量#######
        self.open_price = price.pivot_table(columns='code',index='date',values='open').fillna(method = 'ffill')
        self.close      = price.pivot_table(columns='code',index='date',values='close').fillna(method = 'ffill')
        self.low        = price.pivot_table(columns='code',index='date',values='low').fillna(method = 'ffill')
        self.high       = price.pivot_table(columns='code',index='date',values='high').fillna(method = 'ffill')
        self.avg_price  = price.pivot_table(columns='code',index='date',values='avg_price').fillna(method = 'ffill')
        self.prev_close = price.pivot_table(columns='code',index='date',values='prev_close', dropna=False).fillna(method = 'ffill')
        self.volume     = price.pivot_table(columns='code',index='date',values='volume').fillna(method = 'ffill')
        self.amount     = price.pivot_table(columns='code',index='date',values='amount').fillna(method = 'ffill')
        if benchmark_price is not None:
            self.benchmark_open_price = benchmark_price.pivot_table(columns='code',index='date',values='open').fillna(method = 'ffill')
            self.benchmark_close_price = benchmark_price.pivot_table(columns='code',index='date',values='close').fillna(method = 'ffill')
        else:
            self.benchmark_open_price = None
            self.benchmark_close_price = None
        #self.benchmark_open_price = benchmark_price.loc[:, 'open']
        #self.benchmark_close_price = benchmark_price.loc[:, 'close']
        #########################################################################

    # TSRANK 函数
    def func_rank(self, na):
        return rankdata(na)[-1]/rankdata(na).max()

    # DECAYLINEAR函数
    def func_decaylinear(self, na):
        n = len(na)
        decay_weights = np.arange(1,n+1,1)
        decay_weights = decay_weights / decay_weights.sum()

        return (na * decay_weights).sum()

    # HIGHDAY 函数
    def func_highday(self, na):
        return len(na) - na.values.argmax()

    # LOWDAY 函数
    def func_lowday(self, na):
        return len(na) - na.values.argmin()

    #############################################################################
    #(-1 * CORR(RANK(DELTA(LOG(VOLUME),1)),RANK(((CLOSE-OPEN)/OPEN)),6)
    def alpha_001(self):
        #####成交量与日内价格波动秩关系
        data1 = np.log(self.volume).diff(1).rank(axis=1,pct=True)
        data2 = ((self.close - self.open_price)/self.open_price).rank(axis=1,pct=True)
        alpha = -data1.iloc[-6:,:].corrwith(data2.iloc[-6:,:]).dropna()
        alpha = alpha.dropna()
        return alpha


    def alpha_002(self):
        #####成交量与日内价格波动秩关系
        ##### -1 * delta((((close-low)-(high-close))/((high-low)),1))####
        result=((self.close-self.low)-(self.high-self.close))/((self.high-self.low)).diff()
        m=result.iloc[-1,:].dropna()
        alpha=m[(m<np.inf)&(m>-np.inf)]      #去除值为inf
        return alpha.dropna()


        ################################################################
    def alpha_003(self):
        delay1 = self.close.shift()   #计算close_{i-1}
        condtion1 = (self.close == delay1)
        condition2 = (self.close > delay1)
        condition3 = (self.close < delay1)

        part2 = (self.close-np.minimum(delay1[condition2],self.low[condition2])).iloc[-6:,:] #取最近的6位数据
        part3 = (self.close-np.maximum(delay1[condition3],self.low[condition3])).iloc[-6:,:]

        result=part2.fillna(0)+part3.fillna(0)
        alpha=result.sum()
        return alpha.dropna()

    ########################################################################
    def alpha_004(self):
        condition1=(self.close.rolling(8).std()<self.close.rolling(2).sum()/2)
        condition2=(self.close.rolling(2).sum()/2<(self.close.rolling(8).sum()/8-self.close.rolling(8).std()))
        condition3=(1<=self.volume/self.volume.rolling(20).mean())

        indicator1=pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)#[condition2]
        indicator2=-pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)#[condition3]

        part0=self.close.rolling(8).sum()/8
        part1=indicator2[condition1].fillna(0)
        part2=(indicator1[~condition1][condition2]).fillna(0)
        part3=(indicator1[~condition1][~condition2][condition3]).fillna(0)
        part4=(indicator2[~condition1][~condition2][~condition3]).fillna(0)

        result=part0+part1+part2+part3+part4
        alpha=result.iloc[-1,:]
        return alpha.dropna()

    ################################################################
    def alpha_005(self):
        ts_volume=(self.volume.iloc[-7:,:]).rank(axis=0,pct=True)
        ts_high=(self.high.iloc[-7:,:]).rank(axis=0,pct=True)
        corr_ts=ts_high.rolling(5).corr(ts_volume)
        alpha=corr_ts.max().dropna()
        alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)] #去除inf number
        return alpha

        ###############################################################
    def alpha_006(self):
        condition1=((self.open_price*0.85+self.high*0.15).diff(4)>1)
        condition2=((self.open_price*0.85+self.high*0.15).diff(4)==1)
        condition3=((self.open_price*0.85+self.high*0.15).diff(4)<1)
        indicator1=pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)
        indicator2=pd.DataFrame(np.zeros(self.close.shape),index=self.close.index,columns=self.close.columns)
        indicator3=-pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)
        part1=indicator1[condition1].fillna(0)
        part2=indicator2[condition2].fillna(0)
        part3=indicator3[condition3].fillna(0)
        result=part1+part2+part3
        alpha=(result.rank(axis=1,pct=True)).iloc[-1,:]    #cross section rank
        return alpha.dropna()

    ##################################################################
    def alpha_007(self):
        part1=(np.maximum(self.avg_price-self.close,3)).rank(axis=1,pct=True)
        part2=(np.minimum(self.avg_price-self.close,3)).rank(axis=1,pct=True)
        part3=(self.volume.diff(3)).rank(axis=1,pct=True)
        result=part1+part2*part3
        alpha=result.iloc[-1,:]
        return alpha.dropna()

    ##################################################################
    def alpha_008(self):
        temp=(self.high+self.low)*0.2/2+self.avg_price*0.8
        result=-temp.diff(4)
        alpha=result.rank(axis=1,pct=True)
        alpha=alpha.iloc[-1,:]
        return alpha.dropna()

    ##################################################################
    def alpha_009(self):
        temp=(self.high+self.low)*0.5-(self.high.shift()+self.low.shift())*0.5*(self.high-self.low)/self.volume #计算close_{i-1}
        result=pd.DataFrame.ewm(temp,alpha=2/7).mean()
        alpha=result.iloc[-1,:]
        return alpha.dropna()

    ##################################################################
    def alpha_010(self):
        ret=self.close.pct_change()
        condtion=(ret<0)
        part1=(ret.rolling(20).std()[condtion]).fillna(0)
        part2=(self.close[~condtion]).fillna(0)
        result=np.maximum((part1+part2)**2,5)
        alpha=result.rank(axis=1,pct=True)
        alpha=alpha.iloc[-1,:]
        return alpha.dropna()

    ##################################################################
    def alpha_011(self):
        temp=((self.close-self.low)-(self.high-self.close))/(self.high-self.low)
        result=temp*self.volume
        alpha=result.iloc[-6:,:].sum()
        return alpha.dropna()


    ##################################################################
    def alpha_012(self):
        vwap10=self.avg_price.rolling(10).sum()/10
        temp1=self.open_price-vwap10
        part1=temp1.rank(axis=1,pct=True)
        temp2=(self.close-self.avg_price).abs()
        part2=-temp2.rank(axis=1,pct=True)
        result=part1*part2
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_013(self):
        result=((self.high-self.low)**0.5)-self.avg_price
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_014(self):
        result=self.close-self.close.shift(5)
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_015(self):
        result=self.open_price/self.close.shift()-1
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_016(self):
        temp1=self.volume.rank(axis=1,pct=True)
        temp2=self.avg_price.rank(axis=1,pct=True)
        part=temp1.rolling(5).corr(temp2)#
        part=part[(part<np.inf)&(part>-np.inf)]
        result=part.iloc[-5:,:]
        result=result.dropna(axis=1)
        alpha=-result.max()  #序列按 axis=0排序后，加负号
        return alpha.dropna()


    ##################################################################
    def alpha_017(self):
        #RANK((VWAP-MAX(VWAP,15)))^DELTA(CLOSE,5)
        temp1=self.avg_price.rolling(15).max()
        temp2=(self.avg_price-temp1)
        part1=temp2.rank(axis=1,pct=True)
        part2=self.close.diff(5)
        result=part1**part2
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_018(self):
        delay5=self.close.shift(5)
        alpha=self.close/delay5
        alpha=alpha.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_019(self):
        delay5=self.close.shift(5)
        condition1=(self.close<delay5)
        condition3=(self.close>delay5)
        part1=(self.close[condition1]-delay5[condition1])/delay5[condition1]
        part1=part1.fillna(0)
        part2=(self.close[condition3]-delay5[condition3])/self.close[condition3]
        part2=part2.fillna(0)
        result=part1+part2
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_020(self):
        delay6=self.close.shift(6)
        result=(self.close-delay6)*100/delay6
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_021(self):
        A=self.close.rolling(6).mean().iloc[-6:,:]
        B=np.arange(1,7)   #等差Sequence 1:6
        temp=A.apply(lambda x:linreg(x,B) ,axis=0)  #linear regression
        index = [i[0] for i in temp.items()]
        alpha = pd.Series([np.nan if i[1][3] > 0.05 else i[1][0] for i in temp.items()],index=index)
        return alpha

    ##################################################################
    def alpha_022(self):
        part1=(self.close-self.close.rolling(6).mean())/self.close.rolling(6).mean()
        temp=(self.close-self.close.rolling(6).mean())/self.close.rolling(6).mean()
        part2=temp.shift(3)
        result=part1-part2
        result=pd.DataFrame.ewm(result,alpha=1.0/12).mean()
        alpha=result.iloc[-1,:]
        return alpha.dropna()


        ##################################################################
    def alpha_023(self):
        condition1=(self.close>self.close.shift())
        temp1=self.close.rolling(20).std()[condition1]
        temp1=temp1.fillna(0)
        temp2=self.close.rolling(20).std()[~condition1]
        temp2=temp2.fillna(0)
        part1=pd.DataFrame.ewm(temp1,alpha=1.0/20).mean()
        part2=pd.DataFrame.ewm(temp2,alpha=1.0/20).mean()
        result=part1*100/(part1+part2)
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_024(self):
        delay5=self.close.shift(5)
        result=self.close-delay5
        result=pd.DataFrame.ewm(result,alpha=1.0/5).mean()
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_025(self):
        n=9
        part1=(self.close.diff(7)).rank(axis=1,pct=True)
        part1=part1.iloc[-1,:]
        temp=self.volume/self.volume.rolling(20).mean()
        temp1=temp.iloc[-9:,:]
        seq=[2*i/(n*(n+1)) for i in range(1,n+1)]   #Decaylinear
        # weight=np.array(seq[::-1])
        weight=np.array(seq)

        temp1=temp1.apply(lambda x: x*weight)   #dataframe * numpy array
        ret=self.close.pct_change()   #return
        rank_sum_ret=(ret.sum()).rank(pct=True)
        part2=1-temp1.sum()
        part3=1+rank_sum_ret
        alpha=-part1*part2*part3
        return alpha.dropna()


    ##################################################################
    def alpha_026(self):
        part1=self.close.rolling(7).sum()/7-self.close
        part1=part1.iloc[-1,:]
        delay5=self.close.shift(5)
        part2=self.avg_price.rolling(230).corr(delay5)
        part2=part2.iloc[-1,:]
        alpha=part1+part2
        return alpha.dropna()


    ##################################################################
    def alpha_027(self):
        #公式表达不清楚
        return 0


    ##################################################################
    def alpha_028(self):
        temp1=self.close-self.low.rolling(9).min()
        temp2=self.high.rolling(9).max()-self.low.rolling(9).min()
        part1=3*pd.DataFrame.ewm(temp1*100/temp2,alpha=1.0/3).mean()
        temp3=pd.DataFrame.ewm(temp1*100/temp2,alpha=1.0/3).mean()
        part2=2*pd.DataFrame.ewm(temp3,alpha=1.0/3).mean()
        result=part1-part2
        alpha=result.iloc[-1,:]#.dropna()
        return alpha.dropna()


    ##################################################################
    def alpha_029(self):
        delay6=self.close.shift(6)
        result=(self.close-delay6)*self.volume/delay6
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_030(self):
        #公式表达不清楚
        return 0


    ##################################################################
    def alpha_031(self):
        result=(self.close-self.close.rolling(12).mean())*100/self.close.rolling(12).mean()
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_032(self):
        temp1=self.high.rank(axis=1,pct=True)
        temp2=self.volume.rank(axis=1,pct=True)
        temp3=temp1.rolling(3).corr(temp2)#.dropna()
        temp3=temp3[(temp3<np.inf)&(temp3>-np.inf)].fillna(0) #去inf和fill na 为0
        result=(temp3.rank(axis=1,pct=True)).iloc[-3:,:]
        alpha=-result.sum()
        return alpha.dropna()


    ##################################################################
    def alpha_033(self):
        ret=self.close.pct_change()
        temp1=self.low.rolling(5).min()  #TS_MIN
        part1=temp1.shift(5)-temp1
        part1=part1.iloc[-1,:]
        temp2=(ret.rolling(240).sum()-ret.rolling(20).sum())/220
        part2=temp2.rank(axis=1,pct=True)
        part2=part2.iloc[-1,:]
        temp3=self.volume.iloc[-5:,:]
        part3=temp3.rank(axis=0,pct=True)   #TS_RANK
        part3=part3.iloc[-1,:]
        alpha=part1+part2+part3
        return alpha.dropna()


    ##################################################################
    def alpha_034(self):
        result=self.close.rolling(12).mean()/self.close
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_035(self):
        n=15
        m=7
        temp1=self.open_price.diff()
        temp1=temp1.iloc[-n:,:]
        seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]   #Decaylinear 1
        seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]   #Decaylinear 2
        # weight1=np.array(seq1[::-1])
        # weight2=np.array(seq2[::-1])
        weight1=np.array(seq1)
        weight2=np.array(seq2)
        part1=temp1.apply(lambda x: x*weight1)   #dataframe * numpy array
        part1=part1.rank(axis=1,pct=True)

        temp2=0.65*self.open_price+0.35*self.open_price
        temp2=temp2.rolling(17).corr(self.volume)
        temp2=temp2.iloc[-m:,:]
        part2=temp2.apply(lambda x: x*weight2)
        alpha=np.minimum(part1.iloc[-1,:],-part2.iloc[-1,:])
        alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)] #去除inf number
        alpha=alpha.dropna()
        return alpha


    ##################################################################
    def alpha_036(self):
        temp1=self.volume.rank(axis=1,pct=True)
        temp2=self.avg_price.rank(axis=1,pct=True)
        part1=temp1.rolling(6).corr(temp2)
        result=part1.rolling(2).sum()
        result=result.rank(axis=1,pct=True)
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_037(self):
        ret=self.close.pct_change()
        temp=self.open_price.rolling(5).sum()*ret.rolling(5).sum()
        part1=temp.rank(axis=1,pct=True)
        part2=temp.shift(10)
        result=-part1-part2
        alpha=result.iloc[-1,:].dropna()
        return alpha


    ##################################################################
    def alpha_038(self):
        sum_20=self.high.rolling(20).sum()/20
        delta2=self.high.diff(2)
        condition=(sum_20<self.high)
        result=-delta2[condition].fillna(0)
        alpha=result.iloc[-1,:]
        return alpha


    ##################################################################
    def alpha_039(self):
        n=8
        m=12
        temp1=self.close.diff(2)
        temp1=temp1.iloc[-n:,:]
        seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]   #Decaylinear 1
        seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]   #Decaylinear 2
        # weight1=np.array(seq1[::-1])
        # weight2=np.array(seq2[::-1])
        weight1=np.array(seq1)
        weight2=np.array(seq2)
        part1=temp1.apply(lambda x: x*weight1)   #dataframe * numpy array
        part1=part1.rank(axis=1,pct=True)

        temp2=0.3*self.avg_price+0.7*self.open_price
        volume_180=self.volume.rolling(180).mean()
        sum_vol=volume_180.rolling(37).sum()
        temp3=temp2.rolling(14).corr(sum_vol)
        temp3=temp3.iloc[-m:,:]
        part2=-temp3.apply(lambda x: x*weight2)
        part2=part2.rank(axis=1,pct=True)
        result=part1.iloc[-1,:]-part2.iloc[-1,:]
        alpha=result
        alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)]
        alpha=alpha.dropna()
        return alpha


    ##################################################################
    def alpha_040(self):
        delay1=self.close.shift()
        condition=(self.close>delay1)
        vol=self.volume[condition].fillna(0)
        vol_sum=vol.rolling(26).sum()
        vol1=self.volume[~condition].fillna(0)
        vol1_sum=vol1.rolling(26).sum()
        result=100*vol_sum/vol1_sum
        result=result.iloc[-1,:]
        alpha=result
        alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)]
        alpha=alpha.dropna()
        return alpha


    ##################################################################
    def alpha_041(self):
        delta_avg=self.avg_price.diff(3)
        part=np.maximum(delta_avg,5)
        result=-part.rank(axis=1,pct=True)
        alpha=result.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    ##################################################################
    def alpha_042(self):
        part1=self.high.rolling(10).corr(self.volume)
        part2=self.high.rolling(10).std()
        part2=part2.rank(axis=1,pct=True)
        result=-part1*part2
        alpha=result.iloc[-1,:]
        alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)]
        alpha=alpha.dropna()
        return alpha


    ##################################################################
    def alpha_043(self):
        delay1=self.close.shift()
        condition1=(self.close>delay1)
        condition2=(self.close<delay1)
        temp1=self.volume[condition1].fillna(0)
        temp2=-self.volume[condition2].fillna(0)
        result=temp1+temp2
        result=result.rolling(6).sum()
        alpha=result.iloc[-1,:].dropna()
        return alpha


    ##################################################################
    def alpha_044(self):
        n=6
        m=10
        temp1=self.low.rolling(7).corr(self.volume.rolling(10).mean())
        temp1=temp1.iloc[-n:,:]
        seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]   #Decaylinear 1
        seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]   #Decaylinear 2

        weight1=np.array(seq1)
        weight2=np.array(seq2)
        part1=temp1.apply(lambda x: x*weight1)   #dataframe * numpy array
        part1=part1.iloc[-4:,].rank(axis=0,pct=True)

        temp2=self.avg_price.diff(3)
        temp2=temp2.iloc[-m:,:]
        part2=temp2.apply(lambda x: x*weight2)
        part2=part2.iloc[-5:,].rank(axis=0,pct=True)
        alpha=part1.iloc[-1,:]+part2.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    ##################################################################
    def alpha_045(self):
        temp1=self.close*0.6+self.open_price*0.4
        part1=temp1.diff()
        part1=part1.rank(axis=1,pct=True)
        temp2=self.volume.rolling(150).mean()
        part2=self.avg_price.rolling(15).corr(temp2)
        part2=part2.rank(axis=1,pct=True)
        result=part1*part2
        alpha=result.iloc[-1,:].dropna()
        return alpha


    ##################################################################
    def alpha_046(self):
        part1=self.close.rolling(3).mean()
        part2=self.close.rolling(6).mean()
        part3=self.close.rolling(12).mean()
        part4=self.close.rolling(24).mean()
        result=(part1+part2+part3+part4)*0.25/self.close
        alpha=result.iloc[-1,:].dropna()
        return alpha


    ##################################################################
    def alpha_047(self):
        part1=self.high.rolling(6).max()-self.close
        part2=self.high.rolling(6).max()- self.low.rolling(6).min()
        result=pd.DataFrame.ewm(100*part1/part2,alpha=1.0/9).mean()
        alpha=result.iloc[-1,:].dropna()
        return alpha


        ##################################################################
    def alpha_048(self):
        condition1=(self.close>self.close.shift())
        condition2=(self.close.shift()>self.close.shift(2))
        condition3=(self.close.shift(2)>self.close.shift(3))

        indicator1=pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)[condition1].fillna(0)
        indicator2=pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)[condition2].fillna(0)
        indicator3=pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)[condition3].fillna(0)

        indicator11=-pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)[(~condition1)&(self.close!=self.close.shift())].fillna(0)
        indicator22=-pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)[(~condition2)&(self.close.shift()!=self.close.shift(2))].fillna(0)
        indicator33=-pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)[(~condition3)&(self.close.shift(2)!=self.close.shift(3))].fillna(0)

        summ=indicator1+indicator2+indicator3+indicator11+indicator22+indicator33
        result=-summ*self.volume.rolling(5).sum()/self.volume.rolling(20).sum()
        alpha=result.iloc[-1,:].dropna()
        return alpha


    ##################################################################
    def alpha_049(self):
        delay_high=self.high.shift()
        delay_low=self.low.shift()
        condition1=(self.high+self.low>=delay_high+delay_low)
        condition2=(self.high+self.low<=delay_high+delay_low)
        part1=np.maximum(np.abs(self.high-delay_high),np.abs(self.low-delay_low))
        part1=part1[~condition1]
        part1=part1.iloc[-12:,:].sum()

        part2=np.maximum(np.abs(self.high-delay_high),np.abs(self.low-delay_low))
        part2=part2[~condition2]
        part2=part2.iloc[-12:,:].sum()
        result=part1/(part1+part2)
        alpha=result.dropna()
        return alpha


    ##################################################################
    def alpha_050(self):
        #表达式无意义
        return 0


    ##################################################################
    def alpha_051(self):
        #表达式无意义
        return 0


    ##################################################################
    def alpha_052(self):
        delay=((self.high+self.low+self.close)/3).shift()
        part1=(np.maximum(self.high-delay,0)).iloc[-26:,:]

        part2=(np.maximum(delay-self.low,0)).iloc[-26:,:]
        alpha=part1.sum()+part2.sum()
        return alpha


    ##################################################################
    def alpha_053(self):
        delay=self.close.shift()
        condition=self.close>delay
        result=self.close[condition].iloc[-12:,:]
        alpha=result.count()*100/12
        return alpha.dropna()


    ##################################################################
    def alpha_054(self):
        part1=(self.close-self.open_price).abs()
        part1=part1.std()
        part2=(self.close-self.open_price).iloc[-1,:]
        part3=self.close.iloc[-10:,:].corrwith(self.open_price.iloc[-10:,:])
        result=(part1+part2+part3).dropna()
        alpha=result.rank(pct=True)
        return alpha.dropna()


    ##################################################################
    def alpha_055(self):
        # 尚未实现
        return 0


    ##################################################################
    def alpha_056(self):
        part1=self.open_price.iloc[-1,:]-self.open_price.iloc[-12:,:].min()
        part1=part1.rank(pct=1)
        temp1=(self.high+self.low)/2
        temp1=temp1.rolling(19).sum()
        temp2=self.volume.rolling(40).mean().rolling(19).sum()
        part2=temp1.iloc[-13:,:].corrwith(temp2.iloc[-13:,:])
        part2=(part2.rank(pct=1))**5
        part2=part2.rank(pct=1)

        part1[part1<part2.values]=1                        #先令part1<part2的值为1，再令part1中不为1的值为0，最后替换掉NaN的值
        part1=part1.apply(lambda x: 0 if x <1 else None)
        alpha=part1.fillna(1)
        return alpha.dropna()


    ##################################################################
    def alpha_057(self):
        part1=self.close-self.low.rolling(9).min()
        part2=self.high.rolling(9).max()-self.low.rolling(9).min()
        result=pd.DataFrame.ewm(100*part1/part2,alpha=1.0/3).mean()
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_058(self):
        delay=self.close.shift()
        condition=self.close>delay
        result=self.close[condition].iloc[-20:,:]
        alpha=result.count()*100/20
        return alpha.dropna()


        ##################################################################
    def alpha_059(self):
        delay=self.close.shift()
        condition1=(self.close>delay)
        condition2=(self.close<delay)
        part1=np.minimum(self.low[condition1],delay[condition1]).fillna(0)
        part2=np.maximum(self.high[condition2],delay[condition2]).fillna(0)
        part1=part1.iloc[-20:,:]
        part2=part2.iloc[-20:,:]
        result=self.close-part1-part2
        alpha=result.sum()
        return alpha


    ##################################################################
    def alpha_060(self):
        part1=(self.close.iloc[-20:,:]-self.low.iloc[-20:,:])-(self.high.iloc[-20:,:]-self.close.iloc[-20:,:])
        part2=self.high.iloc[-20:,:]-self.low.iloc[-20:,:]
        result=self.volume.iloc[-20:,:]*part1/part2
        alpha=result.sum()
        return alpha


    ##################################################################
    def alpha_061(self):
        n=12
        m=17
        temp1=self.avg_price.diff()
        temp1=temp1.iloc[-n:,:]
        seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]   #Decaylinear 1
        seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]   #Decaylinear 2
        # weight1=np.array(seq1[::-1])
        # weight2=np.array(seq2[::-1])
        weight1=np.array(seq1)
        weight2=np.array(seq2)
        part1=temp1.apply(lambda x: x*weight1)   #dataframe * numpy array
        part1=part1.rank(axis=1,pct=True)

        temp2=self.low
        temp2=temp2.rolling(8).corr(self.volume.rolling(80).mean())
        temp2=temp2.rank(axis=1,pct=1)
        temp2=temp2.iloc[-m:,:]
        part2=temp2.apply(lambda x: x*weight2)
        part2=-part2.rank(axis=1,pct=1)
        alpha=np.maximum(part1.iloc[-1,:],part2.iloc[-1,:])
        alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)] #去除inf number
        alpha=alpha.dropna()
        return alpha


    ##################################################################
    def alpha_062(self):
        volume_rank=self.volume.rank(axis=1,pct=1)
        result=self.high.iloc[-5:,:].corrwith(volume_rank.iloc[-5:,:])
        alpha=-result
        return alpha.dropna()


        ##################################################################
    def alpha_063(self):
        part1=np.maximum(self.close-self.close.shift(),0)
        part1=pd.DataFrame.ewm(part1,alpha=1.0/6).mean()
        part2=(self.close-self.close.shift()).abs()
        part2=pd.DataFrame.ewm(part2,alpha=1.0/6).mean()
        result=part1*100/part2
        alpha=result.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_064(self):
        n=4
        m=14
        temp1=self.avg_price.rank(axis=1,pct=1).rolling(4).corr(self.volume.rank(axis=1,pct=1))
        temp1=temp1.iloc[-n:,:]
        seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]   #Decayliner 1
        seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]   #Decayliner 2
        # weight1=np.array(seq1[::-1])
        # weight2=np.array(seq2[::-1])
        weight1=np.array(seq1)
        weight2=np.array(seq2)
        part1=temp1.apply(lambda x: x*weight1)   #dataframe * numpy array
        part1=part1.rank(axis=1,pct=True)

        temp2=self.close.rank(axis=1,pct=1)
        temp2=temp2.rolling(4).corr(self.volume.rolling(60).mean())
        temp2=np.maximum(temp2,13)
        temp2=temp2.iloc[-m:,:]
        part2=temp2.apply(lambda x: x*weight2)
        part2=-part2.rank(axis=1,pct=1)
        alpha=np.maximum(part1.iloc[-1,:],part2.iloc[-1,:])
        alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)] #去除inf number
        alpha=alpha.dropna()
        return alpha


    ##################################################################
    def alpha_065(self):
        part1=self.close.iloc[-6:,:]
        alpha=part1.mean()/self.close.iloc[-1,:]
        return alpha.dropna()


    ##################################################################
    def alpha_066(self):
        part1=self.close.iloc[-6:,:]
        alpha=(self.close.iloc[-1,:]-part1.mean())/part1.mean()
        return alpha


    ##################################################################
    def alpha_067(self):
        temp1=self.close-self.close.shift()
        part1=np.maximum(temp1,0)
        part1=pd.DataFrame.ewm(part1,alpha=1.0/24).mean()
        temp2=temp1.abs()
        part2=pd.DataFrame.ewm(temp2,alpha=1.0/24).mean()
        result=part1*100/part2
        alpha=result.iloc[-1,:].dropna()
        return alpha


    ##################################################################
    def alpha_068(self):
        part1=(self.high+self.low)/2-self.high.shift()
        part2=0.5*self.low.shift()*(self.high-self.low)/self.volume
        result=(part1+part2)*100
        result=pd.DataFrame.ewm(result,alpha=2.0/15).mean()
        alpha=result.iloc[-1,:].dropna()
        return alpha


    ##################################################################
    def alpha_069(self):
        # 尚未实现
        return 0


    ##################################################################
    def alpha_070(self):
        #### STD(AMOUNT, 6)
        ## writen by Lin Qitao
        alpha = self.amount.iloc[-6:,:].std().dropna()
        return alpha


    #############################################################################
    def alpha_071(self):
        # (CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
        # Written by Jianing Lu
        data = self.close - self.close.rolling(24).mean() / self.close.rolling(24).mean()
        alpha = data.iloc[-1].dropna()
        return alpha


    #############################################################################
    def alpha_072(self):
        #SMA((TSMAX(HIGH,6)-CLOSE)/(TSMAX(HIGH,6)-TSMIN(LOW,6))*100,15,1)
        # Written by Jianing Lu
        data1 = self.high.rolling(6).max() - self.close
        data2 = self.high.rolling(6).max() - self.low.rolling(6).min()
        alpha = pd.DataFrame.ewm(data1 / data2 * 100, alpha=1/15).mean().iloc[-1].dropna()
        return alpha


    #############################################################################
    def alpha_073(self):
        #((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), VOLUME, 10), 16), 4), 5) - RANK(DECAYLINEAR(CORR(VWAP, MEAN(VOLUME,30), 4),3))) * -1)
        # Written by Jianing Lu
        # 未实现
        #    data1 = self.close.rolling(window=10).corr(self.volume).iloc[-16:,:]
        #    decay_weights1 = np.arange(1,16+1,1)[::-1]
        #    decay_weights1 = decay_weights1 / decay_weights1.sum()
        #   data2 = data1.apply(lambda x : x * decay_weights1)
        return 0


    #############################################################################
    def alpha_074(self):
        #(RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(VOLUME,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(VOLUME), 6)))
        # Written by Jianing Lu
        data1 = (self.low * 0.35 + self.avg_price * 0.65).rolling(window=20).sum()
        data2 = self.volume.rolling(window=40).mean()
        rank1 = data1.rolling(window=7).corr(data2).rank(axis=1, pct=True)
        data3 = self.avg_price.rank(axis=1, pct=True)
        data4 = self.volume.rank(axis=1, pct=True)
        rank2 = data3.rolling(window=6).corr(data4).rank(axis=1, pct=True)
        alpha = (rank1 + rank2).iloc[-1].dropna()
        return alpha


    #############################################################################
    def alpha_075(self):
        ### COUNT(CLOSE>OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)/COUNT(BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN,50)
        # Written by Jianing Lu
        cond1 = (self.close>self.open_price)
        condition = self.benchmark_close_price < self.benchmark_open_price
        cond2 = cond1.mul(condition.iloc[:,0],axis=0)
        count = condition.rolling(50).apply(sum)
        count2 = cond2.rolling(50).apply(sum)
        alpha = count2.div(count.iloc[:,0],axis=0).iloc[-1,:]
        return alpha


    #############################################################################
    def alpha_076(self):
        #STD(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)/MEAN(ABS((CLOSE/DELAY(CLOSE,1)-1))/VOLUME,20)
        # Written by Jianing Lu

        data1 = abs((self.close / ((self.prev_close - 1) / self.volume).shift(20))).std()
        data2 = abs((self.close / ((self.prev_close - 1) / self.volume).shift(20))).mean()
        alpha = (data1 / data2).dropna()
        return alpha


    #############################################################################
    def alpha_077(self):
        #MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH)  -  (VWAP + HIGH)), 20)), RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 3), 6)))
        # Written by Jianing Lu

        data1 = ((self.high + self.low) / 2 + self.high - (self.avg_price + self.high)).iloc[-20:,:]
        decay_weights = np.arange(1,20+1,1)[::-1]
        decay_weights = decay_weights / decay_weights.sum()
        rank1 = data1.apply(lambda x : x * decay_weights).rank(axis=1, pct=True)
        data2 = ((self.high + self.low)/2).rolling(window=3).corr(self.volume.rolling(window=40).mean()).iloc[-6:,:]
        decay_weights2 = np.arange(1,6+1,1)[::-1]
        decay_weights2 = decay_weights2 / decay_weights2.sum()
        rank2 = data2.apply(lambda x : x * decay_weights2).rank(axis=1, pct=True)
        alpha = np.minimum(rank1.iloc[-1], rank2.iloc[-1])
        return alpha


    #############################################################################
    def alpha_078(self):
        #((HIGH+LOW+CLOSE)/3-MA((HIGH+LOW+CLOSE)/3,12))/(0.015*MEAN(ABS(CLOSE-MEAN((HIGH+LOW+CLOSE)/3,12)),12))
        # Written by Jianing Lu
        data1 = (self.high + self.low + self.close) / 3 - ((self.high + self.low + self.close) / 3).rolling(window=12).mean()
        data2 = abs(self.close - ((self.high + self.low + self.close) / 3).rolling(window=12).mean())
        data3 = data2.rolling(window=12).mean() * 0.015
        alpha = (data1 / data3).iloc[-1].dropna()
        return alpha


    #############################################################################
    def alpha_079(self):
        #SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*100
        # Written by Jianing Lu
        data1 = pd.DataFrame.ewm(np.maximum((self.close - self.prev_close), 0), alpha=1/12).mean()
        data2 = pd.DataFrame.ewm(abs(self.close - self.prev_close), alpha=1/12).mean()
        alpha = (data1 / data2 * 100).iloc[-1].dropna()
        return alpha


    #############################################################################
    def alpha_080(self):
        #(VOLUME-DELAY(VOLUME,5))/DELAY(VOLUME,5)*100
        # Written by Jianing Lu
        alpha =  ((self.volume - self.volume.shift(5))/self.volume.shift(5) * 100).iloc[-1].dropna()
        return alpha


    #############################################################################
    def alpha_081(self):
        result=pd.DataFrame.ewm(self.volume,alpha=2.0/21).mean()
        alpha=result.iloc[-1,:].dropna()
        return alpha


    #############################################################################
    def alpha_082(self):
        part1=self.high.rolling(6).max()-self.close
        part2=self.high.rolling(6).max()-self.low.rolling(6).min()
        result=pd.DataFrame.ewm(100*part1/part2,alpha=1.0/20).mean()
        alpha=result.iloc[-1,:].dropna()
        return alpha


    #############################################################################
    def alpha_083(self):
        part1=self.high.rank(axis=0,pct=True)
        part1=part1.iloc[-5:,:]
        part2=self.volume.rank(axis=0,pct=True)
        part2=part2.iloc[-5:,:]
        result=part1.corrwith(part2)
        alpha=-result
        return alpha.dropna()


    #############################################################################
    def alpha_084(self):
        condition1=(self.close>self.close.shift())
        condition2=(self.close<self.close.shift())
        part1=self.volume[condition1].fillna(0)
        part2=-self.volume[condition2].fillna(0)
        result=part1.iloc[-20:,:]+part2.iloc[-20:,:]
        alpha=result.sum().dropna()
        return alpha


    #############################################################################
    def alpha_085(self):
        temp1=self.volume.iloc[-20:,:]/self.volume.iloc[-20:,:].mean()
        temp1=temp1
        part1=temp1.rank(axis=0,pct=True)
        part1=part1.iloc[-1,:]

        delta=self.close.diff(7)
        temp2=-delta.iloc[-8:,:]
        part2=temp2.rank(axis=0,pct=True).iloc[-1,:]
        part2=part2
        alpha=part1*part2
        return alpha.dropna()


    #############################################################################
    def alpha_086(self):

        delay10=self.close.shift(10)
        delay20=self.close.shift(20)
        indicator1=pd.DataFrame(-np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)
        indicator2=pd.DataFrame(np.ones(self.close.shape),index=self.close.index,columns=self.close.columns)

        temp=(delay20-delay10)/10-(delay10-self.close)/10
        condition1=(temp>0.25)
        condition2=(temp<0)
        temp2=(self.close-self.close.shift())*indicator1

        part1=indicator1[condition1].fillna(0)
        part2=indicator2[~condition1][condition2].fillna(0)
        part3=temp2[~condition1][~condition2].fillna(0)
        result=part1+part2+part3
        alpha=result.iloc[-1,:].dropna()

        return alpha

    def alpha_087(self):
        n=7
        m=11
        temp1=self.avg_price.diff(4)
        temp1=temp1.iloc[-n:,:]
        seq1=[2*i/(n*(n+1)) for i in range(1,n+1)]   #Decayliner 1
        seq2=[2*i/(m*(m+1)) for i in range(1,m+1)]   #Decayliner 2
        # weight1=np.array(seq1[::-1])
        # weight2=np.array(seq2[::-1])
        weight1=np.array(seq1)
        weight2=np.array(seq2)
        part1=temp1.apply(lambda x: x*weight1)   #dataframe * numpy array
        part1=part1.rank(axis=1,pct=True)

        temp2=self.low-self.avg_price
        temp3=self.open_price-0.5*(self.high+self.low)
        temp2=temp2/temp3
        temp2=temp2.iloc[-m:,:]
        part2=-temp2.apply(lambda x: x*weight2)

        part2=part2.rank(axis=0,pct=1)
        alpha=part1.iloc[-1,:]+part2.iloc[-1,:]
        alpha=alpha[(alpha<np.inf)&(alpha>-np.inf)] #去除inf number
        alpha=alpha.dropna()
        return alpha

    def alpha_088(self):
        #(close-delay(close,20))/delay(close,20)*100
        #################### writen by Chen Cheng
        data1=self.close.iloc[-21,:]
        alpha=((self.close.iloc[-1,:]-data1)/data1)*100
        alpha=alpha.dropna()
        return alpha


    def alpha_089(self):
        #2*(sma(close,13,2)-sma(close,27,2)-sma(sma(close,13,2)-sma(close,27,2),10,2))
        ###################### writen by Chen Cheng
        data1=pd.DataFrame.ewm(self.close,span=12,adjust=False).mean()
        data2=pd.DataFrame.ewm(self.close,span=26,adjust=False).mean()
        data3=pd.DataFrame.ewm(data1-data2,span=9,adjust=False).mean()
        alpha=((data1-data2-data3)*2).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_090(self):
        #(rank(corr(rank(vwap),rank(volume),5))*-1)
        ####################### writen by Chen Cheng
        data1=self.avg_price.rank(axis=1,pct=True)
        data2=self.volume.rank(axis=1,pct=True)
        corr=data1.iloc[-5:,:].corrwith(data2.iloc[-5:,:])
        rank1=corr.rank(pct=True)
        alpha=rank1*-1
        alpha=alpha.dropna()
        return alpha


    def alpha_091(self):
        #((rank((close-max(close,5)))*rank(corr((mean(volume,40)),low,5)))*-1)
        ################# writen by Chen Cheng
        data1=self.close
        cond=data1>5
        data1[~cond]=5
        rank1=((self.close-data1).rank(axis=1,pct=True)).iloc[-1,:]
        mean=self.volume.rolling(window=40).mean()
        corr=mean.iloc[-5:,:].corrwith(self.low.iloc[-5:,:])
        rank2=corr.rank(pct=True)
        alpha=rank1*rank2*(-1)
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_092(self):
        # (MAX(RANK(DECAYLINEAR(DELTA(((CLOSE*0.35)+(VWAP*0.65)),2),3)),TSRANK(DECAYLINEAR(ABS(CORR((MEAN(VOLUME,180)),CLOSE,13)),5),15))*-1) #
        delta = (self.close*0.35+self.avg_price*0.65)-(self.close*0.35+self.avg_price*0.65).shift(2)
        rank1 = (delta.rolling(3).apply(self.func_decaylinear,raw=True)).rank(axis=1, pct=True)
        rank2 = self.volume.rolling(180).mean().rolling(13).corr(self.close).abs().rolling(5).apply(self.func_decaylinear,raw=True).rolling(15).apply(self.func_rank,raw=True)
        cond_max = rank1>rank2
        rank2[cond_max] = rank1[cond_max]
        alpha = (-rank2).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_093(self):
        # SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20) #
        cond = self.open_price>=self.open_price.shift()
        data1 = self.open_price-self.low
        data2 = self.open_price-self.open_price.shift()
        cond_max = data1>data2
        data2[cond_max] = data1[cond_max]
        data2[cond] = 0
        alpha = data2.iloc[-20:,:].sum()
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_094(self):
        # SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)
        cond1 = self.close > self.prev_close
        cond2 = self.close < self.prev_close
        value = -self.volume
        value[~cond2] = 0
        value[cond1] = self.volume[cond1]
        alpha = value.iloc[-30:,:].sum()
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_095(self):
        # STD(AMOUNT,20) #
        alpha = self.amount.iloc[-20:,:].std()
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_096(self):
        # SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1) #
        sma1 = pd.DataFrame.ewm(100*(self.close-self.low.rolling(9).min())/(self.high.rolling(9).max()-self.low.rolling(9).min()), span=5, adjust=False).mean()
        alpha = pd.DataFrame.ewm(sma1, span=5, adjust=False).mean().iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_097(self):
        # STD(VOLUME,10) #
        alpha = self.volume.iloc[-10:,:].std()
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_098(self):
        # ((((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))<0.05)||((DELTA((SUM(CLOSE,100)/100),100)/DELAY(CLOSE,100))==0.05))?(-1*(CLOSE-TSMIN(CLOSE,100))):(-1*DELTA(CLOSE,3))) #
        sum_close = self.close.rolling(100).sum()
        cond = (sum_close/100-(sum_close/100).shift(100))/self.close.shift(100) <= 0.05
        left_value = -(self.close-self.close.rolling(100).min())
        right_value = -(self.close-self.close.shift(3))
        right_value[cond] = left_value[cond]
        alpha = right_value.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_099(self):
        # (-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5))) #
        alpha = (-self.close.rank(axis=1, pct=True).rolling(window=5).cov(self.volume.rank(axis=1, pct=True)).rank(axis=1, pct=True)).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_100(self):
        # STD(VOLUME,20) #
        alpha = self.volume.iloc[-20:,:].std()
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_101(self):
        # ((RANK(CORR(CLOSE,SUM(MEAN(VOLUME,30),37),15))<RANK(CORR(RANK(((HIGH*0.1)+(VWAP*0.9))),RANK(VOLUME),11)))*-1) #
        rank1 = (self.close.rolling(window=15).corr((self.volume.rolling(window=30).mean()).rolling(window=37).sum())).rank(axis=1, pct=True)
        rank2 = (self.high*0.1+self.avg_price*0.9).rank(axis=1, pct=True)
        rank3 = self.volume.rank(axis=1, pct=True)
        rank4 = (rank2.rolling(window=11).corr(rank3)).rank(axis=1, pct=True)
        alpha = -(rank1<rank4)
        alpha=alpha.iloc[-1,:].dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_102(self):
        # SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100 #
        max_cond = (self.volume-self.volume.shift())>0
        max_data = self.volume-self.volume.shift()
        max_data[~max_cond] = 0
        sma1 = pd.DataFrame.ewm(max_data, span=11, adjust=False).mean()
        sma2 = pd.DataFrame.ewm((self.volume-self.volume.shift()).abs(), span=11, adjust=False).mean()
        alpha = (sma1/sma2*100).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_103(self):
        ##### ((20-LOWDAY(LOW,20))/20)*100
        ## writen by Lin Qitao
        alpha = (20 - self.low.iloc[-20:,:].apply(self.func_lowday))/20*100
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_104(self):
        # (-1*(DELTA(CORR(HIGH,VOLUME,5),5)*RANK(STD(CLOSE,20)))) #
        corr = self.high.rolling(window=5).corr(self.volume)
        alpha = (-(corr-corr.shift(5))*((self.close.rolling(window=20).std()).rank(axis=1, pct=True))).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_105(self):
        # (-1*CORR(RANK(OPEN),RANK(VOLUME),10)) #
        alpha = -((self.open_price.rank(axis=1, pct=True)).iloc[-10:,:]).corrwith(self.volume.iloc[-10:,:].rank(axis=1, pct=True))
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_106(self):
        # CLOSE-DELAY(CLOSE,20) #
        alpha = (self.close-self.close.shift(20)).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_107(self):
        # (((-1*RANK((OPEN-DELAY(HIGH,1))))*RANK((OPEN-DELAY(CLOSE,1))))*RANK((OPEN-DELAY(LOW,1)))) #
        rank1 = -(self.open_price-self.high.shift()).rank(axis=1, pct=True)
        rank2 = (self.open_price-self.close.shift()).rank(axis=1, pct=True)
        rank3 = (self.open_price-self.low.shift()).rank(axis=1, pct=True)
        alpha = (rank1*rank2*rank3).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_108(self):
        # ((RANK((HIGH-MIN(HIGH,2)))^RANK(CORR((VWAP),(MEAN(VOLUME,120)),6)))*-1) #
        min_cond = self.high>2
        data = self.high
        data[min_cond] = 2
        rank1 = (self.high-data).rank(axis=1, pct=True)
        rank2 = (self.avg_price.rolling(window=6).corr(self.volume.rolling(window=120).mean())).rank(axis=1, pct=True)
        alpha = (-rank1**rank2).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_109(self):
        # SMA(HIGH-LOW,10,2)/SMA(SMA(HIGH-LOW,10,2),10,2)#
        data = self.high-self.low
        sma1 = pd.DataFrame.ewm(data, span=9, adjust=False).mean()
        sma2 = pd.DataFrame.ewm(sma1, span=9, adjust=False).mean()
        alpha = (sma1/sma2).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_110(self):
        # SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100 #
        data1 = self.high-self.prev_close
        data2 = self.prev_close-self.low
        max_cond1 = data1<0
        max_cond2 = data2<0
        data1[max_cond1] = 0
        data2[max_cond2] = 0
        sum1 = data1.rolling(window=20).sum()
        sum2 = data2.rolling(window=20).sum()
        alpha = sum1/sum2*100
        #alpha=alpha.dropna()
        return alpha.iloc[-1,:]


    def alpha_111(self):
        #sma(vol*((close-low)-(high-close))/(high-low),11,2)-sma(vol*((close-low)-(high-close))/(high-low),4,2)
        ###################### writen by Chen Cheng
        data1=self.volume*((self.close-self.low)-(self.high-self.close))/(self.high-self.low)
        x=pd.DataFrame.ewm(data1,span=10).mean()
        y=pd.DataFrame.ewm(data1,span=3).mean()
        alpha=(x-y).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_112(self):
        # (SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100 #
        cond1 = self.close>self.prev_close
        cond2 = self.close<self.prev_close
        data1 = self.close-self.prev_close
        data2 = self.close-self.prev_close
        data1[~cond1] = 0
        data2[~cond2] = 0
        data2 = data2.abs()
        sum1 = data1.rolling(window=12).sum()
        sum2 = data2.rolling(window=12).sum()
        alpha = ((sum1-sum2)/(sum1+sum2)*100).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_113(self):
        #(-1*((rank((sum(delay(close,5),20)/20))*corr(close,volume,2))*rank(corr(sum(close,5),sum(close,20),2))))
        ##################### writen by Chen Cheng
        data1=self.close.iloc[:-5,:]
        rank1=(data1.rolling(window=20).sum()/20).rank(axis=1,pct=True)
        corr1=self.close.iloc[-2:,:].corrwith(self.volume.iloc[-2:,:])
        data2=self.close.rolling(window=5).sum()
        data3=self.close.rolling(window=20).sum()
        corr2=data2.iloc[-2:,:].corrwith(data3.iloc[-2:,:])
        rank2=corr2.rank(axis=0,pct=True)
        alpha=(-1*rank1*corr1*rank2).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_114(self):
        #((rank(delay(((high-low)/(sum(close,5)/5)),2))*rank(rank(volume)))/(((high-low)/(sum(close,5)/5))/(vwap-close)))
        ##################### writen by Chen Cheng
        data1=(self.high-self.low)/(self.close.rolling(window=5).sum()/5)
        rank1=(data1.iloc[-2,:]).rank(axis=0,pct=True)
        rank2=((self.volume.rank(axis=1,pct=True)).rank(axis=1,pct=True)).iloc[-1,:]
        data2=(((self.high-self.low)/(self.close.rolling(window=5).sum()/5))/(self.avg_price-self.close)).iloc[-1,:]
        alpha=(rank1*rank2)/data2
        alpha=alpha.dropna()
        return alpha


        # @author: fuzhongjie
    def alpha_115(self):
        # RANK(CORR(((HIGH*0.9)+(CLOSE*0.1)),MEAN(VOLUME,30),10))^RANK(CORR(TSRANK(((HIGH+LOW)/2),4),TSRANK(VOLUME,10),7)) #
        data1 = (self.high*0.9+self.close*0.1)
        data2 = self.volume.rolling(window=30).mean()
        rank1 = (data1.iloc[-10:,:].corrwith(data2.iloc[-10:,:])).rank(pct=True)
        tsrank1 = ((self.high+self.low)/2).rolling(4).apply(self.func_rank,raw=True)
        tsrank2 = self.volume.rolling(10).apply(self.func_rank,raw=True)
        rank2 = tsrank1.iloc[-7:,:].corrwith(tsrank2.iloc[-7:,:]).rank(pct=True)
        alpha = rank1**rank2
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_116(self):
        # REGBETA(CLOSE,SEQUENCE,20) #
        sequence = pd.Series(range(1,21), index=self.close.iloc[-20:,].index)   # 1~20
        corr = self.close.iloc[-20:,:].corrwith(sequence)
        alpha = corr
        alpha=alpha.dropna()
        return alpha


    def alpha_117(self):
        #######((tsrank(volume,32)*(1-tsrank(((close+high)-low),16)))*(1-tsrank(ret,32)))
        #################### writen by Chen Cheng
        data1=(self.close+self.high-self.low).iloc[-16:,:]
        data2=1-data1.rank(axis=0,pct=True)
        data3=(self.volume.iloc[-32:,:]).rank(axis=0,pct=True)
        ret=(self.close/self.close.shift()-1).iloc[-32:,:]
        data4=1-ret.rank(axis=0,pct=True)
        alpha=(data2.iloc[-1,:])*(data3.iloc[-1,:])*(data4.iloc[-1,:])
        alpha=alpha.dropna()
        return alpha


    def alpha_118(self):
        ######sum(high-open,20)/sum((open-low),20)*100
        ################### writen by Chen Cheng
        data1=self.high-self.open_price
        data2=self.open_price-self.low
        data3=data1.rolling(window=20).sum()
        data4=data2.rolling(window=20).sum()
        alpha=((data3/data4)*100).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_119(self):
        # (RANK(DECAYLINEAR(CORR(VWAP,SUM(MEAN(VOLUME,5),26),5),7))-RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN),RANK(MEAN(VOLUME,15)),21),9),7),8)))
        sum1 = (self.volume.rolling(window=5).mean()).rolling(window=26).sum()
        corr1 = self.avg_price.rolling(window=5).corr(sum1)
        rank1 = corr1.rolling(7).apply(self.func_decaylinear,raw=True).rank(axis=1, pct=True)
        rank2 = self.open_price.rank(axis=1, pct=True)
        rank3 = (self.volume.rolling(window=15).mean()).rank(axis=1, pct=True)
        rank4 = rank2.rolling(window=21).corr(rank3).rolling(window=9).min().rolling(7).apply(self.func_rank,raw=True)
        rank5 = rank4.rolling(8).apply(self.func_decaylinear,raw=True).rank(axis=1, pct=True)
        alpha = (rank1 - rank5).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_120(self):
        ###############(rank(vwap-close))/(rank(vwap+close))
        ################### writen by Chen Cheng
        data1=(self.avg_price-self.close).rank(axis=1,pct=True)
        data2=(self.avg_price+self.close).rank(axis=1,pct=True)
        alpha=(data1/data2).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_121(self):
        '''
        尚未实现
        '''
        return 0


    def alpha_122(self):
        ##### (SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2) - DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1))
        ##### / DELAY(SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2),1)
        ## writen by Lin Qitao
        log_close = np.log(self.close)
        data = pd.DataFrame.ewm(pd.DataFrame.ewm(pd.DataFrame.ewm(log_close, span=12, adjust=False).mean(), span=12, adjust=False).mean(), span=12, adjust=False).mean()
        alpha = (data.iloc[-1,:] / data.iloc[-2,:]) -1
        alpha=alpha.dropna()
        return alpha


    def alpha_123(self):
        #####((RANK(CORR(SUM(((HIGH+LOW)/2), 20), SUM(MEAN(VOLUME, 60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)
        ## writen by Lin Qitao
        data1 = ((self.high + self.low)/2).rolling(20).sum()
        data2 = self.volume.rolling(60).mean().rolling(20).sum()
        rank1 = data1.iloc[-9:,:].corrwith(data2.iloc[-9:,:]).dropna().rank(axis=0, pct=True)
        rank2 = self.low.iloc[-6:,:].corrwith(self.volume.iloc[-6:,:]).dropna().rank(axis=0, pct=True)
        rank1 = rank1[rank1.index.isin(rank2.index)]
        rank2 = rank2[rank2.index.isin(rank1.index)]
        alpha = (rank1 < rank2) * (-1)
        alpha=alpha.dropna()
        return alpha


    def alpha_124(self):
        ##### (CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)
        ## writen by Lin Qitao
        data1 = self.close.rolling(30).max().rank(axis=1, pct=True)
        alpha = (self.close.iloc[-1,:] - self.avg_price.iloc[-1,:]) / (2./3*data1.iloc[-2,:] + 1./3*data1.iloc[-1,:])
        alpha=alpha.dropna()
        return alpha


    def alpha_125(self):
        ##### (RANK(DECAYLINEAR(CORR((VWAP), MEAN(VOLUME, 80), 17), 20)) / RANK(DECAYLINEAR(DELTA((CLOSE * 0.5 + VWAP * 0.5), 3), 16)))
        ## writen by Lin Qitao
        data1 = self.avg_price.rolling(window = 17).corr(self.volume.rolling(80).mean())
        decay_weights = np.arange(1,21,1)[::-1]    # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank1 = data1.iloc[-20:,:].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        data2 = (self.close * 0.5 + self.avg_price * 0.5).diff(3)
        decay_weights = np.arange(1,17,1)[::-1]    # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank2 = data2.iloc[-16:,:].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        alpha = rank1 / rank2
        alpha=alpha.dropna()
        return alpha


    def alpha_126(self):
        #### (CLOSE + HIGH + LOW) / 3
        ## writen by Lin Qitao
        alpha = (self.close.iloc[-1,:] + self.high.iloc[-1,:] + self.low.iloc[-1,:]) / 3
        alpha=alpha.dropna()
        return alpha


    def alpha_127(self):
        '''
        公式不明
        '''
        return 0


    def alpha_128(self):
        '''
        尚未实现
        '''
        return 0


    def alpha_129(self):
        #### SUM((CLOSE - DELAY(CLOSE, 1) < 0 ? ABS(CLOSE - DELAY(CLOSE, 1)):0), 12)
        ## writen by Lin Qitao
        data = self.close.diff(1)
        data[data >= 0] = 0
        data = abs(data)
        alpha = data.iloc[-12:,:].sum()
        alpha=alpha.dropna()
        return alpha


    def alpha_130(self):
        #### alpha_130
        #### (RANK(DELCAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME, 40), 9), 10)) / RANK(DELCAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7), 3)))
        ## writen by Lin Qitao
        data1 = (self.high + self.low) / 2
        data2 = self.volume.rolling(40).mean()
        data3 = data1.rolling(window=9).corr(data2)
        decay_weights = np.arange(1,11,1)[::-1]    # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank1 = data3.iloc[-10:,:].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        data1 = self.avg_price.rank(axis=1, pct=True)
        data2 = self.volume.rank(axis=1, pct=True)
        data3 = data1.rolling(window=7).corr(data2)
        decay_weights = np.arange(1,4,1)[::-1]    # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank2 = data3.iloc[-3:,:].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        alpha = (rank1 / rank2).dropna()
        return alpha

    def alpha_131(self):
        '''
        公式错误: DELAT
        '''
        return 0

    def alpha_132(self):
        #### MEAN(AMOUNT, 20)
        ## writen by Lin Qitao
        alpha = self.amount.iloc[-20:,:].mean()
        alpha=alpha.dropna()
        return alpha


    def alpha_133(self):
        #### alpha_133
        #### ((20 - HIGHDAY(HIGH, 20)) / 20)*100 - ((20 - LOWDAY(LOW, 20)) / 20)*100
        ## writen by Lin Qitao

        alpha = (20 - self.high.iloc[-20:,:].apply(self.func_highday))/20*100 \
                - (20 - self.low.iloc[-20:,:].apply(self.func_lowday))/20*100
        alpha=alpha.dropna()
        return alpha


    def alpha_134(self):
        #### (CLOSE - DELAY(CLOSE, 12)) / DELAY(CLOSE, 12) * VOLUME
        ## writen by Lin Qitao
        alpha = ((self.close.iloc[-1,:] / self.close.iloc[-13,:] - 1) * self.volume.iloc[-1,:])
        alpha=alpha.dropna()
        return alpha


    def alpha_135(self):
        #### SMA(DELAY(CLOSE / DELAY(CLOSE, 20), 1), 20, 1)
        ## writen by Lin Qitao
        def rolling_div(na):
            return na[-1]/na[-21]

        data1 = self.close.rolling(21).apply(rolling_div,raw=True).shift(periods=1)
        alpha = pd.DataFrame.ewm(data1, com=19, adjust=False).mean().iloc[-1,:]
        alpha=alpha.dropna()
        return alpha

    def alpha_136(self):
        #### ((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))
        ## writen by Lin Qitao
        data1 = -(self.close / self.prev_close - 1).diff(3).rank(axis=1, pct=True)
        data2 = self.open_price.iloc[-10:,:].corrwith(self.volume.iloc[-10:,:])
        alpha = (data1.iloc[-1,:] * data2).dropna()

        return alpha

    def alpha_137(self):
        '''
        尚未实现
        '''
        return 0


    def alpha_138(self):
        #### ((RANK(DECAYLINEAR(DELTA((((LOW * 0.7) + (VWAP * 0.3))), 3), 20)) - TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8), TSRANK(MEAN(VOLUME, 60), 17), 5), 19), 16), 7)) * -1)
        ## writen by Lin Qitao
        data1 = (self.low * 0.7 + self.avg_price * 0.3).diff(3)
        decay_weights = np.arange(1,21,1)[::-1]    # 倒序数组
        decay_weights = decay_weights / decay_weights.sum()
        rank1 = data1.iloc[-20:,:].mul(decay_weights, axis=0).sum().rank(axis=0, pct=True)

        data1 = self.low.rolling(8).apply(self.func_rank,raw=True)
        data2 = self.volume.rolling(60).mean().rolling(17).apply(self.func_rank,raw=True)
        data3 = data1.rolling(window=5).corr(data2).rolling(19).apply(self.func_rank,raw=True)
        rank2 = data3.rolling(16).apply(self.func_decaylinear,raw=True).iloc[-7:,:].rank(axis=0, pct=True).iloc[-1,:]

        alpha = (rank2 - rank1).dropna()
        return alpha

    def alpha_139(self):
        #### (-1 * CORR(OPEN, VOLUME, 10))
        ## writen by Lin Qitao
        alpha = - self.open_price.iloc[-10:,:].corrwith(self.volume.iloc[-10:,:]).dropna()
        return alpha

    def alpha_140(self):
        #### MIN(RANK(DECAYLINEAR(((RANK(OPEN) + RANK(LOW)) - (RANK(HIGH) + RANK(CLOSE))), 8)), TSRANK(DECAYLINEAR(CORR(TSRANK(CLOSE, 8), TSRANK(MEAN(VOLUME, 60), 20), 8), 7), 3))
        ## writen by Lin Qitao
        data1 = self.open_price.rank(axis=1, pct=True) + self.low.rank(axis=1, pct=True) \
                - self.high.rank(axis=1, pct=True) - self.close.rank(axis=1, pct=True)
        rank1 = data1.iloc[-8:,:].apply(self.func_decaylinear).rank(pct=True)

        data1 = self.close.rolling(8).apply(self.func_rank)
        data2 = self.volume.rolling(60).mean().rolling(20).apply(self.func_rank)
        data3 = data1.rolling(window=8).corr(data2)
        data3 = data3.rolling(7).apply(self.func_decaylinear)
        rank2 = data3.iloc[-3:,:].rank(axis=0, pct=True).iloc[-1,:]

        '''
        alpha = min(rank1, rank2)   NaN如何比较？
        '''
        return 0

    def alpha_141(self):
        #### (RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME, 15)), 9))* -1)
        ## writen by Lin Qitao
        df1 = self.high.rank(axis=1, pct=True)
        df2 = self.volume.rolling(15).mean().rank(axis=1, pct=True)
        alpha = -df1.iloc[-9:,:].corrwith(df2.iloc[-9:,:]).rank(pct=True)
        alpha=alpha.dropna()
        return alpha

    def alpha_142(self):
        #### (((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((VOLUME/MEAN(VOLUME, 20)), 5)))
        ## writen by Lin Qitao

        rank1 = self.close.iloc[-10:,:].rank(axis=0, pct=True).iloc[-1,:].rank(pct=True)
        rank2 = self.close.diff(1).diff(1).iloc[-1,:].rank(pct=True)
        rank3 = (self.volume / self.volume.rolling(20).mean()).iloc[-5:,:].rank(axis=0, pct=True).iloc[-1,:].rank(pct=True)

        alpha = -(rank1 * rank2 * rank3).dropna()
        alpha=alpha.dropna()
        return alpha

    def alpha_143(self):
        #### CLOSE > DELAY(CLOSE, 1)?(CLOSE - DELAY(CLOSE, 1)) / DELAY(CLOSE, 1) * SELF : SELF
        ## writen by Lin Qitao
        '''
        公式不明
        SELF 初始值怎么设？
        '''
        return 0

    def alpha_144(self):
        #### SUMIF(ABS(CLOSE/DELAY(CLOSE, 1) - 1)/AMOUNT, 20, CLOSE < DELAY(CLOSE, 1))/COUNT(CLOSE < DELAY(CLOSE, 1), 20)
        ## writen by Lin Qitao
        df1 = self.close < self.prev_close
        sumif = ((abs(self.close / self.prev_close - 1)/self.amount) * df1).iloc[-20:,:].sum()
        count = df1.iloc[-20:,:].sum()

        alpha = (sumif / count).dropna()
        alpha=alpha.dropna()
        return alpha


    def alpha_145(self):
        #### (MEAN(VOLUME, 9) - MEAN(VOLUME, 26)) / MEAN(VOLUME, 12) * 100
        ## writen by Lin Qitao

        alpha = (self.volume.iloc[-9:,:].mean() - self.volume.iloc[-26:,:].mean()) / self.volume.iloc[-12:,:].mean() * 100
        alpha=alpha.dropna()
        return alpha


    def alpha_146(self):
        '''
        尚未实现
        '''
        return 0


    def alpha_147(self):
        '''
        尚未实现
        '''
        return 0


    def alpha_148(self):
        #### ((RANK(CORR((OPEN), SUM(MEAN(VOLUME, 60), 9), 6)) < RANK((OPEN - TSMIN(OPEN, 14)))) * -1)
        ## writen by Lin Qitao
        df1 = self.volume.rolling(60).mean().rolling(9).sum()
        rank1 = self.open_price.iloc[-6:,:].corrwith(df1.iloc[-6:,:]).rank(pct=True)
        rank2 = (self.open_price - self.open_price.rolling(14).min()).iloc[-1,:].rank(pct=True)

        alpha = -1 * (rank1 < rank2)
        alpha=alpha.dropna()
        return alpha


    def alpha_149(self):
        '''
        尚未实现
        '''
        return 0


    def alpha_150(self):
        #### (CLOSE + HIGH + LOW)/3 * VOLUME
        ## writen by Lin Qitao

        alpha = ((self.close + self.high + self.low)/3 * self.volume).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_151(self):
        '''
        尚未实现
        '''
        return 0


    ######################## alpha_152 #######################
    # @author: fuzhongjie
    def alpha_152(self):
        # SMA(MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),12)-MEAN(DELAY(SMA(DELAY(CLOSE/DELAY(CLOSE,9),1),9,1),1),26),9,1) #
        # @author: fuzhongjie
        data1 = (pd.DataFrame.ewm(((self.close/self.close.shift(9)).shift()), span=17, adjust=False).mean()).shift().rolling(12).mean()
        data2 = (pd.DataFrame.ewm(((self.close/self.close.shift(9)).shift()), span=17, adjust=False).mean()).shift().rolling(26).mean()
        alpha = (pd.DataFrame.ewm(data1-data2, span=17, adjust=False).mean()).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    ######################## alpha_153 #######################
    # @author: fuzhongjie
    def alpha_153(self):
        # (MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4 #
        alpha = ((self.close.rolling(3).mean()+self.close.rolling(6).mean()+self.close.rolling(12).mean()+self.close.rolling(24).mean())/4).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    ######################## alpha_154 #######################
    # @author: fuzhongjie
    def alpha_154(self):
        # (((VWAP-MIN(VWAP,16)))<(CORR(VWAP,MEAN(VOLUME,180),18))) #
        compare1 = (self.avg_price-self.avg_price.rolling(16).min()).iloc[-1,:]
        compare2 = self.avg_price.iloc[-18:,:].corrwith((self.volume.rolling(180).mean()).iloc[-18:,:])
        alpha = compare1<compare2
        alpha=alpha.dropna()
        return alpha


    ######################## alpha_155 #######################
    # @author: fuzhongjie
    def alpha_155(self):
        # SMA(VOLUME,13,2)-SMA(VOLUME,27,2)-SMA(SMA(VOLUME,13,2)-SMA(VOLUME,27,2),10,2) #
        sma1 = pd.DataFrame.ewm(self.volume, span=12, adjust=False).mean()
        sma2 = pd.DataFrame.ewm(self.volume, span=26, adjust=False).mean()
        sma = pd.DataFrame.ewm(sma1-sma2, span=9, adjust=False).mean()
        alpha = sma.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    ######################## alpha_156 #######################
    def alpha_156(self):
        # (MAX(RANK(DECAYLINEAR(DELTA(VWAP,5),3)),RANK(DECAYLINEAR(((DELTA(((OPEN*0.15)+(LOW*0.85)),2)/((OPEN*0.15)+(LOW*0.85)))*-1),3)))*-1 #
        rank1 = ((self.avg_price-self.avg_price.shift(5)).rolling(3).apply(self.func_decaylinear,raw=True)).rank(axis=1, pct=True)
        rank2 = (-((self.open_price*0.15+self.low*0.85)-(self.open_price*0.15+self.low*0.85).shift(2))/(self.open_price*0.15+self.low*0.85)).rolling(3).apply(self.func_decaylinear,raw=True).rank(axis=1, pct=True)
        max_cond = rank1 > rank2
        result = rank2
        result[max_cond] = rank1[max_cond]
        alpha = (-result).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    ######################## alpha_157 #######################
    # @author: fuzhongjie
    def alpha_157(self):
        # (MIN(PROD(RANK(RANK(LOG(SUM(TSMIN(RANK(RANK((-1*RANK(DELTA((CLOSE-1),5))))),2),1)))),1),5)+TSRANK(DELAY((-1*RET),6),5)) #
        rank1 = (-((self.close-1)-(self.close-1).shift(5)).rank(axis=1, pct=True)).rank(axis=1, pct=True).rank(axis=1, pct=True)
        min1 = rank1.rolling(2).min()
        log1 = np.log(min1)
        rank2 = log1.rank(axis=1, pct=True).rank(axis=1, pct=True)
        cond_min = rank2 > 5
        rank2[cond_min] = 5
        tsrank1 = ((-((self.close/self.prev_close)-1)).shift(6)).rolling(5).apply(self.func_rank,raw=True)
        alpha = (rank2+tsrank1).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    ######################## alpha_158 #######################
    # @author: fuzhongjie
    def alpha_158(self):
        # ((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE #
        alpha = (((self.high-pd.DataFrame.ewm(self.close, span=14, adjust=False).mean())-(self.low-pd.DataFrame.ewm(self.close, span=14, adjust=False).mean()))/self.close).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_159(self):
        #########((close-sum(min(low,delay(close,1)),6))/sum(max(high,delay(close,1))-min(low,delay(close,1)),6)*12*24+(close-sum(min(low,delay(close,1)),12))/sum(max(high,delay(close,1))-min(low,delay(close,1)),12)*6*24+(close-sum(min(low,delay(close,1)),24))/sum(max(high,delay(close,1))-min(low,delay(close,1)),24)*6*24)*100/(6*12+6*24+12*24)
        ################### writen by Chen Cheng
        data1=self.low
        data2=self.close.shift()
        cond=data1>data2
        data1[cond]=data2
        data3=self.high
        data4=self.close.shift()
        cond=data3>data4
        data3[~cond]=data4
        #计算出公式核心部分x
        x=((self.close-data1.rolling(6).sum())/(data2-data1).rolling(6).sum())*12*24
        #计算出公式核心部分y
        y=((self.close-data1.rolling(12).sum())/(data2-data1).rolling(12).sum())*6*24
        #计算出公式核心部分z
        z=((self.close-data1.rolling(24).sum())/(data2-data1).rolling(24).sum())*6*24
        data5=(x+y+z)*(100/(6*12+12*24+6*24))
        alpha=data5.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_160(self):
        ################ writen by Chen Cheng
        ############sma((close<=delay(close,1)?std(close,20):0),20,1)
        data1=self.close.rolling(20).std()
        cond=self.close<=self.prev_close
        data1[~cond]=0
        data2=pd.DataFrame.ewm(data1,span=39).mean()
        alpha=data2.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_161(self):
        ###########mean((max(max(high-low),abs(delay(close,1)-high)),abs(delay(close,1)-low)),12)
        ################ writen by Chen Cheng
        data1=(self.high-self.low)
        data2=pd.Series.abs(self.close.shift()-self.high)
        cond=data1>data2
        data1[~cond]=data2
        data3=pd.Series.abs(self.close.shift()-self.low)
        cond=data1>data3
        data1[~cond]=data3
        alpha=(data1.rolling(12).mean()).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_162(self):
        ###############(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100-min(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100,12))/(max(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100),12)-min(sma(max(close-delay(close,1),0),12,1)/sma(abs(close-delay(close,1)),12,1)*100),12))
        ################# writen by Chen Cheng
        #算出公式核心部分X
        data1=self.close-self.close.shift()
        cond=data1>0
        data1[~cond]=0
        x=pd.DataFrame.ewm(data1,span=23).mean()
        #算出公式核心部分Y
        data2=pd.Series.abs(self.close-self.close.shift())
        y=pd.DataFrame.ewm(data2,span=23).mean()
        #算出公式核心部分Z
        z=(x/y)*100
        cond=z>12
        z[cond]=12
        #计算公式核心部分C
        c=(x/y)*100
        cond=c>12
        c[~cond]=12
        data3=(x/y)*100-(z/c)-c
        alpha=data3.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_163(self):
        ################ writen by Chen Cheng
        #######rank(((((-1*ret)*,ean(volume,20))*vwap)*(high-close)))
        data1=(-1)*(self.close/self.close.shift()-1)*self.volume.rolling(20).mean()*self.avg_price*(self.high-self.close)
        data2=(data1.rank(axis=1,pct=True)).iloc[-1,:]
        alpha=data2
        alpha=alpha.dropna()
        return alpha


    def alpha_164(self):
        ################ writen by Chen Cheng
        ############sma((((close>delay(close,1))?1/(close-delay(close,1)):1)-min(((close>delay(close,1))?1/(close/delay(close,1)):1),12))/(high-low)*100,13,2)
        cond=self.close>self.close.shift()
        data1=1/(self.close-self.close.shift())
        data1[~cond]=1
        data2=1/(self.close-self.close.shift())
        cond=data2>12
        data2[cond]=12
        data3=data1-data2/((self.high-self.low)*100)
        alpha=(pd.DataFrame.ewm(data3,span=12).mean()).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_165(self):
        '''
        尚未实现
        '''
        return 0


    def alpha_166(self):
        '''
        尚未实现
        '''
        return 0


    def alpha_167(self):
        ## writen by Chen Cheng
        ####sum(((close-delay(close,1)>0)?(close-delay(close,1)):0),12)####
        data1=self.close-self.close.shift()
        cond=(data1<0)
        data1[cond]=0
        data2=(data1.rolling(12).sum()).iloc[-1,:]
        alpha=data2
        alpha=alpha.dropna()
        return alpha


    def alpha_168(self):
        ## writen by Chen Cheng
        #####-1*volume/mean(volume,20)####
        data1=(-1*self.volume)/self.volume.rolling(20).mean()
        alpha=data1.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_169(self):
        ## writen by Chen Cheng
        ###sma(mean(delay(sma(close-delay(close,1),9,1),1),12)-mean(delay(sma(close-delay(close,1),1,1),1),26),10,1)#####
        data1=self.close-self.close.shift()
        data2=(pd.DataFrame.ewm(data1,span=17).mean()).shift()
        data3=data2.rolling(12).mean()-data2.rolling(26).mean()
        alpha=(pd.DataFrame.ewm(data3,span=19).mean()).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_170(self):
        ## writen by Chen Cheng
        #####((((rank((1/close))*volume)/mean(volume,20))*((high*rank((high-close)))/(sum(high,5)/5)))-rank((vwap-delay(vwap,5))))####
        #计算公式左边部分，X
        data1=(1/self.close).rank(axis=0,pct=True)
        data2=self.volume.rolling(20).mean()
        x=(data1*self.volume)/data2
        #计算公式中间部分，Y
        data3=(self.high-self.close).rank(axis=0,pct=True)
        data4=self.high.rolling(5).mean()
        y=(data3*self.high)/data4
        #计算公式右边部分，Z
        z=(self.avg_price.iloc[-1,:]-self.avg_price.iloc[-5,:]).rank(axis=0,pct=True)
        alpha=(x*y-z).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_171(self):
        ## writen by Chen Cheng
        ####(((low-close)*open^5)*-1)/((close-high)*close^5)#####
        #获取数据，求出分子
        data1=-1*(self.low-self.close)*(self.open_price**5)
        #获取数据。求出分母
        data2=(self.close-self.high)*(self.close**5)
        alpha = (data1/data2).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_172(self):
        # MEAN(ABS(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0&HD>LD)?HD:0,14)*100/(SUM((LD>0&LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM(TR,14)+SUM((HD>0&HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6) #
        hd = self.high-self.high.shift()
        ld = self.low.shift()-self.low
        temp1 = self.high-self.low
        temp2 = (self.high-self.close.shift()).abs()
        cond1 = temp1>temp2
        temp2[cond1] = temp1[cond1]
        temp3 = (self.low-self.close.shift()).abs()
        cond2 = temp2>temp3
        temp3[cond2] = temp2[cond2]
        tr = temp3   # MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
        sum_tr14 = tr.rolling(14).sum()
        cond3 = ld>0
        cond4 = ld>hd
        cond3[~cond4] = False
        data1 = ld
        data1[~cond3] = 0
        sum1 = data1.rolling(14).sum()*100/sum_tr14
        cond5 = hd>0
        cond6 = hd>ld
        cond5[~cond6] = False
        data2 = hd
        data2[~cond5] = 0
        sum2 = data2.rolling(14).sum()*100/sum_tr14
        alpha = ((sum1-sum2).abs()/(sum1+sum2)*100).rolling(6).mean().iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_173(self):
        ## writen by Chen Cheng
        ####3*sma(close,13,2)-2*sma(sma(close,13,2),13,2)+sma(sma(sma(log(close),13,2),13,2),13,2)#####
        data1=pd.DataFrame.ewm(self.close,span=12).mean()
        data2=pd.DataFrame.ewm(data1,span=12).mean()
        close_log=np.log(self.close)
        data3=pd.DataFrame.ewm(close_log,span=12).mean()
        data4=pd.DataFrame.ewm(data3,span=12).mean()
        data5=pd.DataFrame.ewm(data4,span=12).mean()
        alpha=(3*data1-2*data2+data5).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_174(self):
        ## writen by Chen Cheng
        ####sma((close>delay(close,1)?std(close,20):0),20,1)#####
        cond=self.close>self.prev_close
        data2=self.close.rolling(20).std()
        data2[~cond] = 0
        alpha=(pd.DataFrame.ewm(data2,span=39,adjust=False).mean()).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_175(self):
        ## writen by Chen Cheng
        #####mean(max(max(high-low),abs(delay(close,1)-high)),abs(delay(close,1)-low)),6)####
        #获取比较数据，进行一级比较
        data1=self.high-self.low
        data2=pd.Series.abs(self.close.shift()-self.high)
        cond=(data1>data2)
        data2[cond] = data1[cond]
        #获取比较数据，进行二级比较
        data3=pd.Series.abs(self.close.shift()-self.low)
        cond=(data2>data3)
        data3[cond] = data2[cond]
        #求和，输出
        data4=(data3.rolling(window=6).mean()).iloc[-1,:]
        alpha=data4
        alpha=alpha.dropna()
        return alpha


    def alpha_176(self):
        ## writen by Chen Cheng
        ######### #########corr(rank((close-tsmin(low,12))/(tsmax(high,12)-tsmin(low,12))),rank(volume),6)#############
        #获取数据，求出RANK1
        data1=(self.close-self.low.rolling(window=12).min())/(self.high.rolling(window=12).max()-self.low.rolling(window=12).min())
        data2=data1.rank(axis=0,pct=True)
        #获取数据求出rank2
        data3=self.volume.rank(axis=0,pct=True)
        corr=data2.iloc[-6:,:].corrwith(data3.iloc[-6:,:])
        alpha=corr
        alpha=alpha.dropna()
        return alpha


    ################## alpha_177 ####################
    # @author: Lin Qitao
    def alpha_177(self):
        ##### ((20-HIGHDAY(HIGH,20))/20)*100 #####
        alpha = (20 - self.high.iloc[-20:,:].apply(self.func_highday))/20*100
        alpha=alpha.dropna()
        return alpha


    def alpha_178(self):
        ##### (close-delay(close,1))/delay(close,1)*volume ####
        ## Writen by Chencheng
        alpha=((self.close-self.close.shift())/self.close.shift()*self.volume).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_179(self):
        #####（rank(corr(vwap,volume,4))*rank(corr(rank(low),rank(mean(volume,50)),12))####
        ## Writen by Chencheng
        rank1=(self.avg_price.iloc[-4:,:].corrwith(self.volume.iloc[-4:,:])).rank(axis=0,pct=True)
        #获取两个RANK内的所需值
        data2=self.low.rank(axis=0,pct=True)
        data3=(self.volume.rolling(window=50).mean()).rank(axis=0,pct=True)
        rank2=(data2.iloc[-12:,:].corrwith(data3.iloc[-12:,:])).rank(axis=0,pct=True)
        alpha=rank1*rank2
        alpha=alpha.dropna()
        return alpha


        ##################### alpha_180 #######################
    # @author: fuzhongjie
    def alpha_180(self):
        ##### ((MEAN(VOLUME,20)<VOLUME)?((-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7)):(-1*VOLUME))) #####
        ma = self.volume.rolling(window=20).mean()
        cond = (ma < self.volume).iloc[-20:,:]
        sign = delta_close_7 = self.close.diff(7)
        sign[sign.iloc[:,:]<0] = -1
        sign[sign.iloc[:,:]>0] = 1
        sign[sign.iloc[:,:]==0] = 0
        left = (((self.close.diff(7).abs()).iloc[-60:,:].rank(axis=0, pct=True)*(-1)).iloc[-20:,:] * sign.iloc[-20:,:]).iloc[-20:,:]
        right = self.volume.iloc[-20:,:]*(-1)
        right[cond] = left[cond]
        alpha = right.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_181(self):
        '''
        尚未实现
        '''
        return 0


    ######################## alpha_182 #######################
    # @author: fuzhongjie
    def alpha_182(self):
        ##### COUNT((CLOSE>OPEN & BANCHMARKINDEXCLOSE>BANCHMARKINDEXOPEN)OR(CLOSE<OPEN & BANCHMARKINDEXCLOSE<BANCHMARKINDEXOPEN),20)/20 #####
        cond1 = (self.close>self.open_price)
        cond2 = (self.benchmark_open_price>self.benchmark_close_price)
        cond3 = (self.close<self.open_price)
        cond4 = (self.benchmark_open_price<self.benchmark_close_price)
        cond5 = cond1.mul(cond2.iloc[:,0],axis=0)
        cond6 = cond3.mul(cond4.iloc[:,0],axis=0)
        cond = cond5|cond6
        count = cond.rolling(20).apply(sum)
        alpha = (count/20).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_183(self):
        '''
        尚未实现
        '''
        return 0


    def alpha_184(self):
        #####(rank(corr(delay((open-close),1),close,200))+rank((open-close))) ####
        ## Writen by Chencheng
        data1=self.open_price.shift()-self.close.shift()
        data2=self.open_price.iloc[-1,:] - self.close.iloc[-1,:]
        corr=data1.iloc[-200:,:].corrwith(self.close.iloc[-200:,:])
        alpha=data2.rank(axis=0,pct=True)+corr.rank(axis=0,pct=True)
        alpha=alpha.dropna()
        return alpha


    def alpha_185(self):
        ##### RANK((-1 * ((1 - (OPEN / CLOSE))^2))) ####
        ## writen by fuzhongjie/chencheng
        alpha = (-(1-self.open_price/self.close)**2).rank(axis=1, pct=True).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    # @author: fuzhongjie
    def alpha_186(self):
        # (MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6)+DELAY(MEAN(ABS(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)-SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))/(SUM((LD>0 & LD>HD)?LD:0,14)*100/SUM(TR,14)+SUM((HD>0 & HD>LD)?HD:0,14)*100/SUM(TR,14))*100,6),6))/2 #
        hd = self.high-self.high.shift()
        ld = self.low.shift()-self.low
        temp1 = self.high-self.low
        temp2 = (self.high-self.close.shift()).abs()
        cond1 = temp1>temp2
        temp2[cond1] = temp1[cond1]
        temp3 = (self.low-self.close.shift()).abs()
        cond2 = temp2>temp3
        temp3[cond2] = temp2[cond2]
        tr = temp3   # MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))
        sum_tr14 = tr.rolling(14).sum()
        cond3 = ld>0
        cond4 = ld>hd
        cond3[~cond4] = False
        data1 = ld
        data1[~cond3] = 0
        sum1 = data1.rolling(14).sum()*100/sum_tr14
        cond5 = hd>0
        cond6 = hd>ld
        cond5[~cond6] = False
        data2 = hd
        data2[~cond5] = 0
        sum2 = data2.rolling(14).sum()*100/sum_tr14
        mean1 = ((sum1-sum2).abs()/(sum1+sum2)*100).rolling(6).mean()
        alpha = ((mean1 + mean1.shift(6))/2).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_187(self):
        ##### SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20) ####
        ## writen by fuzhongjie
        cond = (self.open_price <= self.open_price.shift())
        data1 = self.high - self.low                        # HIGH-LOW
        data2 = self.open_price - self.open_price.shift()   # OPEN-DELAY(OPEN,1)
        cond_max = data2 > data1
        data1[cond_max] = data2[cond_max]
        data1[cond] = 0
        alpha = data1.iloc[-20:,:].sum()
        alpha=alpha.dropna()
        return alpha


    def alpha_188(self):
        ##### ((HIGH-LOW–SMA(HIGH-LOW,11,2))/SMA(HIGH-LOW,11,2))*100 #####
        ## writen by fuzhongjie
        sma = pd.DataFrame.ewm(self.high - self.low, span=10, adjust=False).mean()   # α=1/(s+1)   SMA=Yt+1=(m/n)*Ai+(1-m/n)*Yt
        alpha = ((self.high - self.low - sma)/sma*100).iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_189(self):
        ##### mean(abs(close-mean(close,6),6)) ####
        ## writen by fuzhongjie/chencheng
        ma6 = self.close.rolling(window=6).mean()   # 6日移动平均线
        alpha = (self.close - ma6).abs().rolling(window=6).mean().iloc[-1,:]
        alpha=alpha.dropna()
        return alpha


    def alpha_190(self):
        ##### LOG((COUNT(CLOSE/DELAY(CLOSE)-1>((CLOSE/DELAY(CLOSE,19))^(1/20)-1),20)-1)*(SUMIF(((CLOSE/DELAY(CLOSE)
        ##### -1-(CLOSE/DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1))/((
        ##### COUNT((CLOSE/DELAY(CLOSE)-1<(CLOSE/DELAY(CLOSE,19))^(1/20)-1),20))*(SUMIF((CLOSE/DELAY(CLOSE)-1-((CLOSE
        ##### /DELAY(CLOSE,19))^(1/20)-1))^2,20,CLOSE/DELAY(CLOSE)-1>(CLOSE/DELAY(CLOSE,19))^(1/20)-1)))) ####
        '''
        尚未实现
        '''
        return 0


    def alpha_191(self):
        ##### (CORR(MEAN(VOLUME,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE ####
        ## writen by fuzhongjie/chencheng
        volume_avg = self.volume.rolling(window=20).mean()
        corr = volume_avg.iloc[-5:,:].corrwith(self.low.iloc[-5:,:])
        alpha = corr + (self.high.iloc[-1,:] + self.low.iloc[-1,:])/2 - self.close.iloc[-1,:]
        alpha=alpha.dropna()
        return alpha

    def alpha(self):
        alpha_001=self.alpha_001()
        alpha_002=self.alpha_002()
        alpha_003=self.alpha_003()
        alpha_004=self.alpha_004()
        alpha_005=self.alpha_005()
        alpha_006=self.alpha_006()
        alpha_007=self.alpha_007()
        alpha_008=self.alpha_008()
        alpha_009=self.alpha_009()
        alpha_010=self.alpha_010()
        alpha_011=self.alpha_011()
        alpha_012=self.alpha_012()
        alpha_013=self.alpha_013()
        alpha_014=self.alpha_014()
        alpha_015=self.alpha_015()
        alpha_016=self.alpha_016()
        alpha_017=self.alpha_017()
        alpha_018=self.alpha_018()
        alpha_019=self.alpha_019()
        alpha_020=self.alpha_020()
        alpha_021=self.alpha_021()
        alpha_022=self.alpha_022()
        alpha_023=self.alpha_023()
        alpha_024=self.alpha_024()
        alpha_025=self.alpha_025()
        alpha_026=self.alpha_026()
        #alpha_027=self.alpha_027()
        alpha_028=self.alpha_028()
        alpha_029=self.alpha_029()
        #alpha_030=self.alpha_030()
        alpha_031=self.alpha_031()
        alpha_032=self.alpha_032()
        alpha_033=self.alpha_033()
        alpha_034=self.alpha_034()
        alpha_035=self.alpha_035()
        alpha_036=self.alpha_036()
        alpha_037=self.alpha_037()
        alpha_038=self.alpha_038()
        alpha_039=self.alpha_039()
        alpha_040=self.alpha_040()
        alpha_041=self.alpha_041()
        alpha_042=self.alpha_042()
        alpha_043=self.alpha_043()
        alpha_044=self.alpha_044()
        alpha_045=self.alpha_045()
        alpha_046=self.alpha_046()
        alpha_047=self.alpha_047()
        alpha_048=self.alpha_048()
        alpha_049=self.alpha_049()
        #alpha_050=self.alpha_050()
        #alpha_051=self.alpha_051()
        alpha_052=self.alpha_052()
        alpha_053=self.alpha_053()
        alpha_054=self.alpha_054()
        #alpha_055=self.alpha_055()
        alpha_056=self.alpha_056()
        alpha_057=self.alpha_057()
        alpha_058=self.alpha_058()
        alpha_059=self.alpha_059()
        alpha_060=self.alpha_060()
        alpha_061=self.alpha_061()
        alpha_062=self.alpha_062()
        alpha_063=self.alpha_063()
        alpha_064=self.alpha_064()
        alpha_065=self.alpha_065()
        alpha_066=self.alpha_066()
        alpha_067=self.alpha_067()
        alpha_068=self.alpha_068()
        #alpha_069=self.alpha_069()
        alpha_070=self.alpha_070()
        alpha_071=self.alpha_071()
        alpha_072=self.alpha_072()
        #alpha_073=self.alpha_073()
        alpha_074=self.alpha_074()
        if self.benchmark_open_price is not None:
            alpha_075=self.alpha_075()
        alpha_076=self.alpha_076()
        alpha_077=self.alpha_077()
        alpha_078=self.alpha_078()
        alpha_079=self.alpha_079()
        alpha_080=self.alpha_080()
        alpha_081=self.alpha_081()
        alpha_082=self.alpha_082()
        alpha_083=self.alpha_083()
        alpha_084=self.alpha_084()
        alpha_085=self.alpha_085()
        alpha_086=self.alpha_086()
        alpha_087=self.alpha_087()
        alpha_088=self.alpha_088()
        alpha_089=self.alpha_089()
        alpha_090=self.alpha_090()
        alpha_091=self.alpha_091()
        alpha_092=self.alpha_092()
        alpha_093=self.alpha_093()
        alpha_094=self.alpha_094()
        alpha_095=self.alpha_095()
        alpha_096=self.alpha_096()
        alpha_097=self.alpha_097()
        alpha_098=self.alpha_098()
        alpha_099=self.alpha_099()
        alpha_100=self.alpha_100()
        alpha_101=self.alpha_101()
        alpha_102=self.alpha_102()
        alpha_103=self.alpha_103()
        alpha_104=self.alpha_104()
        alpha_105=self.alpha_105()
        alpha_106=self.alpha_106()
        alpha_107=self.alpha_107()
        alpha_108=self.alpha_108()
        alpha_109=self.alpha_109()
        alpha_110=self.alpha_110()
        alpha_111=self.alpha_111()
        alpha_112=self.alpha_112()
        alpha_113=self.alpha_113()
        alpha_114=self.alpha_114()
        alpha_115=self.alpha_115()
        alpha_116=self.alpha_116()
        alpha_117=self.alpha_117()
        alpha_118=self.alpha_118()
        alpha_119=self.alpha_119()
        alpha_120=self.alpha_120()
        #alpha_121=self.alpha_121()
        alpha_122=self.alpha_122()
        alpha_123=self.alpha_123()
        alpha_124=self.alpha_124()
        alpha_125=self.alpha_125()
        alpha_126=self.alpha_126()
        alpha_127=self.alpha_127()
        alpha_128=self.alpha_128()
        alpha_129=self.alpha_129()
        alpha_130=self.alpha_130()
        alpha_131=self.alpha_131()
        alpha_132=self.alpha_132()
        alpha_133=self.alpha_133()
        alpha_134=self.alpha_134()
        alpha_135=self.alpha_135()
        alpha_136=self.alpha_136()
        alpha_137=self.alpha_137()
        alpha_138=self.alpha_138()
        alpha_139=self.alpha_139()
        #alpha_140=self.alpha_140()
        alpha_141=self.alpha_141()
        alpha_142=self.alpha_142()
        alpha_143=self.alpha_143()
        alpha_144=self.alpha_144()
        alpha_145=self.alpha_145()
        #alpha_146=self.alpha_146()
        #alpha_147=self.alpha_147()
        alpha_148=self.alpha_148()
        #alpha_149=self.alpha_149()
        alpha_150=self.alpha_150()
        alpha_151=self.alpha_151()
        alpha_152=self.alpha_152()
        alpha_153=self.alpha_153()
        alpha_154=self.alpha_154()
        alpha_155=self.alpha_155()
        alpha_156=self.alpha_156()
        alpha_157=self.alpha_157()
        alpha_158=self.alpha_158()
        alpha_159=self.alpha_159()
        alpha_160=self.alpha_160()
        alpha_161=self.alpha_161()
        alpha_162=self.alpha_162()
        alpha_163=self.alpha_163()
        alpha_164=self.alpha_164()
        #alpha_165=self.alpha_165()
        #alpha_166=self.alpha_166()
        alpha_167=self.alpha_167()
        alpha_168=self.alpha_168()
        alpha_169=self.alpha_169()
        alpha_170=self.alpha_170()
        alpha_171=self.alpha_171()
        alpha_172=self.alpha_172()
        alpha_173=self.alpha_173()
        alpha_174=self.alpha_174()
        alpha_175=self.alpha_175()
        alpha_176=self.alpha_176()
        alpha_177=self.alpha_177()
        alpha_178=self.alpha_178()
        alpha_179=self.alpha_179()
        alpha_180=self.alpha_180()
        #alpha_181=self.alpha_181()
        if self.benchmark_open_price is not None:
            alpha_182=self.alpha_182()
        #alpha_183=self.alpha_183()
        alpha_184=self.alpha_184()
        alpha_185=self.alpha_185()
        alpha_186=self.alpha_186()
        alpha_187=self.alpha_187()
        alpha_188=self.alpha_188()
        alpha_189=self.alpha_189()
        #alpha_190=self.alpha_190()
        alpha_191=self.alpha_191()
        if self.benchmark_open_price is not None:
            res = pd.DataFrame({
               "alpha_001":alpha_001,
               "alpha_002":alpha_002,
               "alpha_003":alpha_003,
               "alpha_004":alpha_004,
               "alpha_005":alpha_005,
               "alpha_006":alpha_006,
               "alpha_007":alpha_007,
               "alpha_008":alpha_008,
               "alpha_009":alpha_009,
               "alpha_010":alpha_010,
               "alpha_011":alpha_011,
               "alpha_012":alpha_012,
               "alpha_013":alpha_013,
               "alpha_014":alpha_014,
               "alpha_015":alpha_015,
               "alpha_016":alpha_016,
               "alpha_017":alpha_017,
               "alpha_018":alpha_018,
               "alpha_019":alpha_019,
               "alpha_020":alpha_020,
               "alpha_021":alpha_021,
               "alpha_022":alpha_022,
               "alpha_023":alpha_023,
               "alpha_024":alpha_024,
               "alpha_025":alpha_025,
               "alpha_026":alpha_026,
               #"alpha_027":alpha_027,
               "alpha_028":alpha_028,
               "alpha_029":alpha_029,
               #"alpha_030":alpha_030,
               "alpha_031":alpha_031,
               "alpha_032":alpha_032,
               "alpha_033":alpha_033,
               "alpha_034":alpha_034,
               "alpha_035":alpha_035,
               "alpha_036":alpha_036,
               "alpha_037":alpha_037,
               "alpha_038":alpha_038,
               "alpha_039":alpha_039,
               "alpha_040":alpha_040,
               "alpha_041":alpha_041,
               "alpha_042":alpha_042,
               "alpha_043":alpha_043,
               "alpha_044":alpha_044,
               "alpha_045":alpha_045,
               "alpha_046":alpha_046,
               "alpha_047":alpha_047,
               "alpha_048":alpha_048,
               "alpha_049":alpha_049,
               #"alpha_050":alpha_050,
               #"alpha_051":alpha_051,
               "alpha_052":alpha_052,
               "alpha_053":alpha_053,
               "alpha_054":alpha_054,
               #"alpha_055":alpha_055,
                "alpha_056":alpha_056,
                "alpha_057":alpha_057,
                "alpha_058":alpha_058,
                "alpha_059":alpha_059,
                "alpha_060":alpha_060,
                "alpha_061":alpha_061,
                "alpha_062":alpha_062,
                "alpha_063":alpha_063,
                "alpha_064":alpha_064,
                "alpha_065":alpha_065,
                "alpha_066":alpha_066,
                "alpha_067":alpha_067,
                "alpha_068":alpha_068,
                #"alpha_069":alpha_069,
                "alpha_070":alpha_070,
                "alpha_071":alpha_071,
                "alpha_072":alpha_072,
                #"alpha_073":alpha_073,
                "alpha_074":alpha_074,
                "alpha_075":alpha_075,
                "alpha_076":alpha_076,
                "alpha_077":alpha_077,
                "alpha_078":alpha_078,
                "alpha_079":alpha_079,
                "alpha_080":alpha_080,
                "alpha_081":alpha_081,
                "alpha_082":alpha_082,
                "alpha_083":alpha_083,
                "alpha_084":alpha_084,
                "alpha_085":alpha_085,
                "alpha_086":alpha_086,
                "alpha_087":alpha_087,
                "alpha_088":alpha_088,
                "alpha_089":alpha_089,
                "alpha_090":alpha_090,
                "alpha_091":alpha_091,
                "alpha_092":alpha_092,
                "alpha_093":alpha_093,
                "alpha_094":alpha_094,
                "alpha_095":alpha_095,
                "alpha_096":alpha_096,
                "alpha_097":alpha_097,
                "alpha_098":alpha_098,
                "alpha_099":alpha_099,
                "alpha_100":alpha_100,
                "alpha_101":alpha_101,
                "alpha_102":alpha_102,
                "alpha_103":alpha_103,
                "alpha_104":alpha_104,
                "alpha_105":alpha_105,
                "alpha_106":alpha_106,
                "alpha_107":alpha_107,
                "alpha_108":alpha_108,
                "alpha_109":alpha_109,
                "alpha_110":alpha_110,
                "alpha_111":alpha_111,
                "alpha_112":alpha_112,
                "alpha_113":alpha_113,
                "alpha_114":alpha_114,
                "alpha_115":alpha_115,
                "alpha_116":alpha_116,
                "alpha_117":alpha_117,
                "alpha_118":alpha_118,
                "alpha_119":alpha_119,
                "alpha_120":alpha_120,
                #"alpha_121":alpha_121,
                "alpha_122":alpha_122,
                "alpha_123":alpha_123,
                "alpha_124":alpha_124,
                "alpha_125":alpha_125,
                "alpha_126":alpha_126,
                "alpha_127":alpha_127,
                "alpha_128":alpha_128,
                "alpha_129":alpha_129,
                "alpha_130":alpha_130,
                #"alpha_131":alpha_131,
                "alpha_132":alpha_132,
                "alpha_133":alpha_133,
                "alpha_134":alpha_134,
                "alpha_135":alpha_135,
                "alpha_136":alpha_136,
                "alpha_137":alpha_137,
                "alpha_138":alpha_138,
                "alpha_139":alpha_139,
                #"alpha_140":alpha_140,
                "alpha_141":alpha_141,
                "alpha_142":alpha_142,
                #"alpha_143":alpha_143,
                "alpha_144":alpha_144,
                "alpha_145":alpha_145,
                #"alpha_146":alpha_146,
                #"alpha_147":alpha_147,
                "alpha_148":alpha_148,
                #"alpha_149":alpha_149,
                "alpha_150":alpha_150,
                #"alpha_151":alpha_151,
                "alpha_152":alpha_152,
                "alpha_153":alpha_153,
                "alpha_154":alpha_154,
                "alpha_155":alpha_155,
                "alpha_156":alpha_156,
                "alpha_157":alpha_157,
                "alpha_158":alpha_158,
                "alpha_159":alpha_159,
                "alpha_160":alpha_160,
                "alpha_161":alpha_161,
                "alpha_162":alpha_162,
                "alpha_163":alpha_163,
                "alpha_164":alpha_164,
                #"alpha_165":alpha_165,
                #"alpha_166":alpha_166,
                "alpha_167":alpha_167,
                "alpha_168":alpha_168,
                "alpha_169":alpha_169,
                "alpha_170":alpha_170,
                "alpha_171":alpha_171,
                "alpha_172":alpha_172,
                "alpha_173":alpha_173,
                "alpha_174":alpha_174,
                "alpha_175":alpha_175,
                "alpha_176":alpha_176,
                "alpha_177":alpha_177,
                "alpha_178":alpha_178,
                "alpha_179":alpha_179,
                "alpha_180":alpha_180,
                #"alpha_181":alpha_181,
                "alpha_182":alpha_182,
                #"alpha_183":alpha_183,
                "alpha_184":alpha_184,
                "alpha_185":alpha_185,
                "alpha_186":alpha_186,
                "alpha_187":alpha_187,
                "alpha_188":alpha_188,
                "alpha_189":alpha_189,
                #"alpha_190":alpha_190,
                "alpha_191":alpha_191,
                'date':self.date
            })
        else:
            res = pd.DataFrame({
                "alpha_001":alpha_001,
                "alpha_002":alpha_002,
                "alpha_003":alpha_003,
                "alpha_004":alpha_004,
                "alpha_005":alpha_005,
                "alpha_006":alpha_006,
                "alpha_007":alpha_007,
                "alpha_008":alpha_008,
                "alpha_009":alpha_009,
                "alpha_010":alpha_010,
                "alpha_011":alpha_011,
                "alpha_012":alpha_012,
                "alpha_013":alpha_013,
                "alpha_014":alpha_014,
                "alpha_015":alpha_015,
                "alpha_016":alpha_016,
                "alpha_017":alpha_017,
                "alpha_018":alpha_018,
                "alpha_019":alpha_019,
                "alpha_020":alpha_020,
                "alpha_021":alpha_021,
                "alpha_022":alpha_022,
                "alpha_023":alpha_023,
                "alpha_024":alpha_024,
                "alpha_025":alpha_025,
                "alpha_026":alpha_026,
                #"alpha_027":alpha_027,
                "alpha_028":alpha_028,
                "alpha_029":alpha_029,
                #"alpha_030":alpha_030,
                "alpha_031":alpha_031,
                "alpha_032":alpha_032,
                "alpha_033":alpha_033,
                "alpha_034":alpha_034,
                "alpha_035":alpha_035,
                "alpha_036":alpha_036,
                "alpha_037":alpha_037,
                "alpha_038":alpha_038,
                "alpha_039":alpha_039,
                "alpha_040":alpha_040,
                "alpha_041":alpha_041,
                "alpha_042":alpha_042,
                "alpha_043":alpha_043,
                "alpha_044":alpha_044,
                "alpha_045":alpha_045,
                "alpha_046":alpha_046,
                "alpha_047":alpha_047,
                "alpha_048":alpha_048,
                "alpha_049":alpha_049,
                #"alpha_050":alpha_050,
                #"alpha_051":alpha_051,
                "alpha_052":alpha_052,
                "alpha_053":alpha_053,
                "alpha_054":alpha_054,
                #"alpha_055":alpha_055,
                "alpha_056":alpha_056,
                "alpha_057":alpha_057,
                "alpha_058":alpha_058,
                "alpha_059":alpha_059,
                "alpha_060":alpha_060,
                "alpha_061":alpha_061,
                "alpha_062":alpha_062,
                "alpha_063":alpha_063,
                "alpha_064":alpha_064,
                "alpha_065":alpha_065,
                "alpha_066":alpha_066,
                "alpha_067":alpha_067,
                "alpha_068":alpha_068,
                #"alpha_069":alpha_069,
                "alpha_070":alpha_070,
                "alpha_071":alpha_071,
                "alpha_072":alpha_072,
                #"alpha_073":alpha_073,
                "alpha_074":alpha_074,
                #"alpha_075":alpha_075,
                "alpha_076":alpha_076,
                "alpha_077":alpha_077,
                "alpha_078":alpha_078,
                "alpha_079":alpha_079,
                "alpha_080":alpha_080,
                "alpha_081":alpha_081,
                "alpha_082":alpha_082,
                "alpha_083":alpha_083,
                "alpha_084":alpha_084,
                "alpha_085":alpha_085,
                "alpha_086":alpha_086,
                "alpha_087":alpha_087,
                "alpha_088":alpha_088,
                "alpha_089":alpha_089,
                "alpha_090":alpha_090,
                "alpha_091":alpha_091,
                "alpha_092":alpha_092,
                "alpha_093":alpha_093,
                "alpha_094":alpha_094,
                "alpha_095":alpha_095,
                "alpha_096":alpha_096,
                "alpha_097":alpha_097,
                "alpha_098":alpha_098,
                "alpha_099":alpha_099,
                "alpha_100":alpha_100,
                "alpha_101":alpha_101,
                "alpha_102":alpha_102,
                "alpha_103":alpha_103,
                "alpha_104":alpha_104,
                "alpha_105":alpha_105,
                "alpha_106":alpha_106,
                "alpha_107":alpha_107,
                "alpha_108":alpha_108,
                "alpha_109":alpha_109,
                "alpha_110":alpha_110,
                "alpha_111":alpha_111,
                "alpha_112":alpha_112,
                "alpha_113":alpha_113,
                "alpha_114":alpha_114,
                "alpha_115":alpha_115,
                "alpha_116":alpha_116,
                "alpha_117":alpha_117,
                "alpha_118":alpha_118,
                "alpha_119":alpha_119,
                "alpha_120":alpha_120,
                #"alpha_121":alpha_121,
                "alpha_122":alpha_122,
                "alpha_123":alpha_123,
                "alpha_124":alpha_124,
                "alpha_125":alpha_125,
                "alpha_126":alpha_126,
                "alpha_127":alpha_127,
                "alpha_128":alpha_128,
                "alpha_129":alpha_129,
                "alpha_130":alpha_130,
                #"alpha_131":alpha_131,
                "alpha_132":alpha_132,
                "alpha_133":alpha_133,
                "alpha_134":alpha_134,
                "alpha_135":alpha_135,
                "alpha_136":alpha_136,
                "alpha_137":alpha_137,
                "alpha_138":alpha_138,
                "alpha_139":alpha_139,
                #"alpha_140":alpha_140,
                "alpha_141":alpha_141,
                "alpha_142":alpha_142,
                #"alpha_143":alpha_143,
                "alpha_144":alpha_144,
                "alpha_145":alpha_145,
                #"alpha_146":alpha_146,
                #"alpha_147":alpha_147,
                "alpha_148":alpha_148,
                #"alpha_149":alpha_149,
                "alpha_150":alpha_150,
                #"alpha_151":alpha_151,
                "alpha_152":alpha_152,
                "alpha_153":alpha_153,
                "alpha_154":alpha_154,
                "alpha_155":alpha_155,
                "alpha_156":alpha_156,
                "alpha_157":alpha_157,
                "alpha_158":alpha_158,
                "alpha_159":alpha_159,
                "alpha_160":alpha_160,
                "alpha_161":alpha_161,
                "alpha_162":alpha_162,
                "alpha_163":alpha_163,
                "alpha_164":alpha_164,
                #"alpha_165":alpha_165,
                #"alpha_166":alpha_166,
                "alpha_167":alpha_167,
                "alpha_168":alpha_168,
                "alpha_169":alpha_169,
                "alpha_170":alpha_170,
                "alpha_171":alpha_171,
                "alpha_172":alpha_172,
                "alpha_173":alpha_173,
                "alpha_174":alpha_174,
                "alpha_175":alpha_175,
                "alpha_176":alpha_176,
                "alpha_177":alpha_177,
                "alpha_178":alpha_178,
                "alpha_179":alpha_179,
                "alpha_180":alpha_180,
                #"alpha_181":alpha_181,
                #"alpha_182":alpha_182,
                #"alpha_183":alpha_183,
                "alpha_184":alpha_184,
                "alpha_185":alpha_185,
                "alpha_186":alpha_186,
                "alpha_187":alpha_187,
                "alpha_188":alpha_188,
                "alpha_189":alpha_189,
                #"alpha_190":alpha_190,
                "alpha_191":alpha_191,
                'date':self.date
            })
        return(res)

if __name__ == '__main__':
    pass
