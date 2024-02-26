import fix_yahoo_finance as yf
from datetime import date
import pandas as pd
import plotly.express as px
from ta.trend import sma_indicator, macd
from ta.momentum import rsi
import plotly.graph_objects as go
import numpy as np

AAPL = pd.DataFrame(yf.download(
    "AAPL", start=date(1993, 1, 1), end=date(2023, 1, 1)))
SNP = pd.DataFrame(yf.download(
    "^GSPC", start=date(1993, 1, 1), end=date(2023, 1, 1)))
AAPLNorm = (AAPL - np.min(AAPL)) / (np.max(AAPL) - np.min(AAPL))
SNPNorm = (SNP - np.min(SNP)) / (np.max(SNP) - np.min(SNP))
# print(AAPL)
# print(SNP)
# Check whether the dataframes contains NaN or empty values
# print(AAPL.isnull().values.any())
# print(SNP.isnull().values.any())


# Open értékek összehasonlítása
# fig = px.line(AAPLNorm,x=AAPLNorm.index,  y=AAPLNorm['Open'])
# fig.add_scatter(x=SNPNorm.index,y=SNPNorm['Open'])
# fig.show()

# 200 day Moving average
MA200AAPL = sma_indicator(AAPLNorm['Open'], window=200)
MA200SNP = sma_indicator(SNPNorm['Open'], window=200)

fig = px.line(MA200AAPL, x=MA200AAPL.index,  y=MA200AAPL)
fig.add_scatter(x=MA200SNP.index, y=MA200SNP)
fig.show()
'''
#50 day Moving average
MA50AAPL=sma_indicator(AAPLNorm['Open'],window=50)
MA50SNP=sma_indicator(SNPNorm['Open'],window=50)

fig = px.line(MA50AAPL,x=MA50AAPL.index,  y=MA50AAPL)
fig.add_scatter(x=MA50SNP.index,y=MA50SNP)
fig.show()

#RSI
MA50AAPL=rsi(AAPLNorm['Open'],window=14)
MA50SNP=rsi(SNPNorm['Open'],window=14)

fig = px.line(MA50AAPL,x=MA50AAPL.index,  y=MA50AAPL)
fig.add_scatter(x=MA50SNP.index,y=MA50SNP)
fig.show()
'''
# MACD
MACDAAPL = macd(AAPLNorm['Open'])
MACDSNP = macd(SNPNorm['Open'])

fig = px.line(MACDAAPL, x=MACDAAPL.index,  y=MACDAAPL)
fig.add_scatter(x=MACDSNP.index, y=MACDSNP)
fig.show()

'''
#Candle body histogram
candlePercentageAAPL=[]
candlePercentageAAPL.append(0)
for i in range(1,len(AAPLNorm['Open'])):
    green=((AAPL['Close'][i]/AAPL['Open'][i])-1)*100
    red=((AAPL['Open'][i]/AAPL['Close'][i])-1)*100
    if green>=0 and green<20:
        candlePercentageAAPL.append(round(green,0))
    elif red>0 and red<20:
        candlePercentageAAPL.append(round(red*-1,0))
AAPL['CandlePercentage']=candlePercentageAAPL

candlePercentageSNP=[]
candlePercentageSNP.append(0)
for i in range(1,len(AAPLNorm['Open'])):
    green=((SNP['Close'][i]/SNP['Open'][i])-1)*100
    red=((SNP['Open'][i]/SNP['Close'][i])-1)*100
    if green>=0 and green<20:
        candlePercentageSNP.append(round(green,0))
    elif red>0 and red<20:
        candlePercentageSNP.append(round(red*-1,0))

SNP['CandlePercentage']=candlePercentageSNP

fig = go.Figure()
fig.add_trace(go.Histogram(x=AAPL['CandlePercentage'],name="AAPL"))
fig.add_trace(go.Histogram(x=SNP['CandlePercentage'],name= "SNP"))
fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.75)
fig.show()
'''
