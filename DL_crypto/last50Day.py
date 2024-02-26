from asyncore import write
import json
import time
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import requests
from urllib import request
from datetime import datetime
import matplotlib.pyplot as plt
import pandas_datareader as web
import fix_yahoo_finance as yf
import talib as ta
import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from datetime import datetime, timedelta, date
from torch import autograd
from torch.utils.data import random_split
import random
import math
from statistics import mean

cryptoNamesTr = {


}
# cryptoNamesTr = {"BTC-USD"}

stockCurrencyNames = {
    "NU", "GOOGL",
    "RIOT",
    "T", "APL", "APE", "TSLA", "NIO", "MARA", "NVDA",
    "DNA", "KVUE", "AMD", "PLTR", "PDD", "AMZN", "RIVN",
    "BAC", "AFRM",
    "COIN", "SOFI", "LCID", "AAL",
    "VALE", "META", "CSCO", "CCL", "PFE",
    "PLUG", "GRAB", "BABA", "SNAP", "OPEN", "GOLD",
    "MSFT", "PBR", "LI",
    "ET", "SU", "AI", "JBLU", "BSX",
                      "M",  "KEY", "IQ", "UBER",
                      "KGC", "SIRI", "CMCSA",
                      "C", "HPE", "MRVL", "GPS",
    "CVNA", "PYPL", "DIS", "HST", "RIVN", "HPQ", "ABCM",
    "UBER", "PDD", "ITUB", "AAL", "GOOG", "BBD", "OPEN", "CCL",
    "PLUG", "LCID", "META", "MU", "PFE", "SIRI", "VZ", "CVNA", "BABA",
    "XPEV", "RUN", "HPE", "S", "CSCO", "PYPL", "PCG", "XOM", "BEKE",
    "NUVA", "PBR", "MSFT", "IONQ", "VALE", "NOK", "CMCSA", "C",
    "MRVL", "CX", "GMED", "RGC", "KGC", "HBAN", "GOLD", "JNJ", "RIG", "WFC",
    "AI", "WBD", "M", "O", "DIS", "KEY", "HST", "USB", "DISH", "JD", "SWN", "AMBA", "GPS", "DKNG", "SHOP",
    "RBLX", "SCHW", "ABEC"

}


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
testData = []
trainData = []
lostSum = []


class GlobalParameters:
    def __init__(self, sLossP, tProfitP, testRatioP, useVolumeDataP, useCandleWickDataP, historyLengthP, useCryptoDataOnlyP, confidalityUpperLimitP, confidalityBottomLimitP):
        self.sLoss = sLossP
        self.tProfit = tProfitP
        self.testRatio = testRatioP
        self.useVolumeData = useVolumeDataP
        self.useCandleWickData = useCandleWickDataP
        self.historyLength = historyLengthP
        self.useCryptoDataOnly = useCryptoDataOnlyP
        self.confidalityUpperLimit = confidalityUpperLimitP
        self.confidalityBottomLimit = confidalityBottomLimitP

    def __str__(self):
        details = ''
        details += f'Stop Loss level        : {self.sLoss}\n'
        details += f'Take Profit level    : {self.tProfit}\n'
        details += f'Use volume data : {self.useVolumeData}\n'
        details += f'Use candlewick data : {self.useCandleWickData}\n'
        details += f'History Length : {self.historyLength}\n'
        details += f'Use crypto only : {self.useCryptoDataOnly}\n'
        details += f'Confidality upper limit : {self.confidalityUpperLimit}\n'
        details += f'Confidality bottom limit : {
            self.confidalityBottomLimit}\n'
        return details


class DataObject:
    def __init__(self, data, label):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)


class Currency:
    def __init__(self, open, volume, high, low):
        self.price = (open-np.min(open))/(np.max(open)-np.min(open))
        self.volume = (volume-np.min(volume))/(np.max(volume)-np.min(volume))
        self.high = (high-np.min(high))/(np.max(high)-np.min(high))
        self.low = (low-np.min(low))/(np.max(low)-np.min(low))

    def getCurrency(self, volume, candlestick):
        temp = []
        for item in range(0, len(self.price)):
            temp.append(self.price[item])

            if volume:
                temp.append(self.volume[item])

            if candlestick:
                temp.append(self.high[item])
                temp.append(self.low[item])

        return temp


class CryptoDataLoader:
    currencyPairData = []

    globalParameters = GlobalParameters(
        0.05, 0.1, 0.8, True, True, 10, False, 1, 0.8)

    def __init__(self, cryptoCurrencyPairsNameP, stockCurrencyPairsNameP, globalParametersP, isTest, start=date(2010, 1, 1), end=date(2023, 6, 1)):
        self.cryptoCurrencyPairsName = cryptoCurrencyPairsNameP
        self.stockCurrencyPairsName = stockCurrencyPairsNameP
        self.globalParameters = globalParametersP
        self.isTest = isTest
        self.start = start
        self.end = end
        self.fetchCurrencyPairsData()
        self.createDataSet()

    def fetchCurrencyPairsData(self):
        for item in self.cryptoCurrencyPairsName:
            if self.start == date(2010, 1, 1):
                df = yf.download(item, end=self.end)
            else:
                start = self.start - \
                    timedelta(days=globalParameters.historyLength)
                start = start.strftime('%Y-%m-%d')
                df = yf.download(item, start=start, end=self.end)
                x = df["Open"].values
                if 0.0 in df["Open"].values:
                    continue
            self.currencyPairData.append(df)
        if not globalParameters.useCryptoDataOnly:
            for item in self.stockCurrencyPairsName:
                if self.start == date(2010, 1, 1):
                    df = yf.download(item, end=self.end)
                else:
                    start = self.start - \
                        timedelta(days=globalParameters.historyLength)
                    start = start.strftime('%Y-%m-%d')
                    df = yf.download(item, start=self.start, end=self.end)

                x = df["Open"].values
                if 0.0 in df["Open"].values or 0.0 in df["High"].values or 0.0 in df["Low"].values or 0.0 in df["Close"].values or 0.0 in df["Volume"].values:
                    continue
                self.currencyPairData.append(df)

    def createDataSet(self):
        self.dataObject = []
        for item in self.currencyPairData:
            tempDataObject = []
            index = self.globalParameters.historyLength
            while index+int(globalParameters.historyLength/5) < len(item["Open"]):
                labelAdded = False
                start = index-self.globalParameters.historyLength
                currency = Currency(item["Open"][start:index],
                                    item["Volume"][start:index],
                                    item["High"][start:index],
                                    item["Low"][start:index])

                tempBasePrice = item["Open"][index]
                tempTakeProfitPrice = tempBasePrice * \
                    (1+self.globalParameters.tProfit)
                tempStopLossPrice = tempBasePrice * \
                    (1-self.globalParameters.sLoss)

                for item2 in range(index, len(item["Open"])):
                    high = item["High"][item2]
                    low = item["Low"][item2]

                    if high > tempTakeProfitPrice and low < tempStopLossPrice:
                        tempDataObject.append(DataObject(currency.getCurrency(
                            self.globalParameters.useVolumeData, self.globalParameters.useCandleWickData), 0))
                        break
                    elif high > tempTakeProfitPrice:
                        tempDataObject.append(DataObject(currency.getCurrency(
                            self.globalParameters.useVolumeData, self.globalParameters.useCandleWickData), 1))
                        break
                    elif low < tempStopLossPrice:
                        tempDataObject.append(DataObject(currency.getCurrency(
                            self.globalParameters.useVolumeData, self.globalParameters.useCandleWickData), 0))
                        break
                index += int(globalParameters.historyLength/5)
            self.dataObject.append(tempDataObject)
        self.splitData()

    def __len__(self):
        return len(self.dataObject[0])

    def __getitem__(self, idx):
        data = self.dataObject[0][idx].data
        label = self.dataObject[0][idx].label
        return data, label

    def splitData(self):
        self.train = []
        self.test = []
        for item in self.dataObject:
            itemL = len(item)
            self.train.extend(
                item[0:int(self.globalParameters.testRatio*itemL)])
            self.test.extend(
                item[int(self.globalParameters.testRatio*itemL):itemL])


class TrainDataLoader(DataLoader):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx].data
        label = self.data[idx].label
        return data, label


class TestDataLoader(DataLoader):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx].data
        label = self.data[idx].label
        return data, label


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, inputs, targets):
        multiplier = 1.0

        for item in range(len(inputs)):
            input = inputs[item]
            target = targets[item]
            if input > 0.5 and target == 0:
                multiplier += 0.1

           # if input<0.5 and target==1:
            #    loss+=-1*torch.log(1-input)
            # elif input>0.5 and target==0:
            #    loss+=-1*torch.log(input)*3
            # elif input>0.5 and target==1:
            #    loss+=-1*torch.log(1-input)
            # else:
            #    loss+=-1*torch.log(input)

            # loss=loss/len(inputs)

        # return torch.tensor(loss, dtype=torch.float32,requires_grad=True)
        loss = -1 * (targets * torch.log(inputs) +
                     (1 - targets) * torch.log(1 - inputs))
        rr = targets*torch.log(inputs)
        qr = (1 - targets) * torch.log(1 - inputs)
        lm = loss.mean()
        if lm != lm:
            a = 1
        return loss.mean()*multiplier


class NeuralNetwork(nn.Module):
    def __init__(self, globalParameters):
        super().__init__()
        self.size = globalParameters.historyLength
        if (globalParameters.useVolumeData):
            self.size += globalParameters.historyLength
        if globalParameters.useCandleWickData:
            self.size += globalParameters.historyLength*2
        self.linearReluStack = nn.Sequential(
            nn.Linear(self.size, int(self.size)),
            nn.LeakyReLU(),
            nn.Linear(int(self.size), int(self.size/2)),
            nn.LeakyReLU(),
            nn.Linear(int(self.size/2), int(self.size/4)),
            nn.LeakyReLU(),
            nn.Linear(int(self.size/4), int(self.size/8)),
            nn.LeakyReLU(),
            nn.Linear(int(self.size/8), int(self.size/16)),
            nn.LeakyReLU(),
            nn.Linear(int(self.size/16), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.linearReluStack(x)
        return output


def testing():
    testData = torch.load('testDataSet.pth')


def training(loadDataFromDisk):

    if loadDataFromDisk:
        testData = torch.load('testDataSet.pth')
        trainData = torch.load('trainDataSet.pth')
    else:
        dataset = CryptoDataLoader(cryptoNamesTr,
                                   stockCurrencyNames, globalParameters, True)
        trainData = TrainDataLoader(dataset.train)
        testData = TestDataLoader(dataset.test)
        # trainData, testData = random_split(dataset, [globalParameters.testRatio,1-globalParameters.testRatio])
        testData = DataLoader(testData, batch_size=64, shuffle=True)
        trainData = DataLoader(trainData, batch_size=10, shuffle=True)
        torch.save(testData, 'testDataSet.pth')
        torch.save(trainData, 'trainDataSet.pth')

    print(f"Using {device} device")
    loss_fn = nn.BCELoss()  # binary cross entropy
    model = NeuralNetwork(globalParameters)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.train
    model = model.to(device)

    for epoch in range(n_epochs):
        for xBatch, yBatch in trainData:
            xBatch = xBatch.to(device=device)
            yBatch = yBatch.to(device=device)
            prediction = model(xBatch.cuda())
            prediction = torch.squeeze(prediction, 1)
            loss = loss_fn(prediction, yBatch)
            optimizer.zero_grad()
            if loss == loss:
                loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')
        lostSum.append(loss.item())

        torch.save(model.state_dict(), "model1.pth")


def check_accuracy(test_loader: DataLoader, device):
    model = NeuralNetwork(globalParameters)
    model.load_state_dict(torch.load('model1.pth'))
    model = model.to(device)
    num_correct = 0
    total = 0
    xAxis = np.arange(101)
    corrPredByPercentage = np.zeros(101)
    wrongPredByPercentage = np.zeros(101)
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in test_loader:
            data = data.to(device=device)
            labels = labels.to(device=device)
            predictionsBinary = []

            predictions = model(data.cuda())
            for item in predictions:
                if globalParameters.confidalityBottomLimit <= item <= globalParameters.confidalityUpperLimit:

                    predictionsBinary.append(1)
                else:
                    predictionsBinary.append(0)
            for item in range(len(predictions)):
                if (predictionsBinary[item] == 1 and labels[item] == 1):
                    corrPredByPercentage[int(predictions[item]*100)] += 1
                    correct += 1
                    total += 1
                elif predictionsBinary[item] == 1 and labels[item] == 0:
                    wrongPredByPercentage[int(predictions[item]*100)] += 1
                    total += 1

        plt.stackplot(xAxis, wrongPredByPercentage, corrPredByPercentage,
                      colors=['r', 'c'])
        plt.show()
        f = open("results.txt", "a")

        f.write("----------------------"+"\n")
        f.write(str(datetime.now())+"\n")
        f.write(
            f"Test Accuracy of the model: {float(correct)/float(total)*100:.2f}"+"\n")
        f.write(
            f"Correct: {correct}"+"\n")
        f.write(
            f"Wrong: {total-correct}"+"\n")
        f.write(str(globalParameters))
        f.write(' '.join(cryptoNamesTr)+"\n")
        f.write(' '.join(stockCurrencyNames)+"\n")
        f.write("----------------------")
        f.write("")
        f.write("")
        f.close()


def testChart30(start, end, elementName, device):
    model = NeuralNetwork(globalParameters)
    model.load_state_dict(torch.load('model1.pth'))
    model = model.to(device)
    cryptoName = []
    cryptoName.append(elementName)
    globalParameters.testRatio = 0.0
    testingData = CryptoDataLoader(cryptoName,
                                   [], globalParameters, True, start, end)
    testingData = DataLoader(testingData, batch_size=120, shuffle=False)
    predictionsBinary = []
    predictionsGood = []
    predictionsBad = []
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in testingData:
            data = data.to(device=device)
            labels = labels.to(device=device)

            predictions = model(data.cuda())
            for item in predictions:
                if globalParameters.confidalityBottomLimit <= item <= globalParameters.confidalityUpperLimit:
                    predictionsBinary.append(1)
                else:
                    predictionsBinary.append(0)

            for item in range(0, len(predictions)):
                if (predictionsBinary[item] == 1 and labels[item] == 1):
                    correct += 1
                    total += 1
                    predictionsGood.append(1)
                    predictionsBad.append(0)
                elif predictionsBinary[item] == 1 and labels[item] == 0:
                    total += 1
                    predictionsGood.append(0)
                    predictionsBad.append(1)
                else:
                    predictionsGood.append(0)
                    predictionsBad.append(0)

    print(
        f"Test Accuracy of the model: {float(correct)/float(total)*100:.2f}"+"\n")
    print("Correct: "+str(correct))
    print("Total: "+str(total))
    chart = yf.download(elementName, end=end, start=start)
    open = chart['Open']

    good = chart['Close']
    bad = chart['High']
    predLen = len(predictionsGood)
    for item in range(predLen):
        if (predictionsGood[item] == 1):
            good[item] = open[item]
            bad[item] = 0.35
        elif predictionsBad[item] == 1:
            bad[item] = open[item]
            good[item] = 0.35
        else:
            good[item] = 0.35
            bad[item] = 0.35

    close = chart['Close'][0:predLen]
    fig, ax = plt.subplots()
    ax.plot(open[0:predLen])
    ax.plot(good[0:predLen], 'go')
    ax.plot(bad[0:predLen], 'ro')
    ax.legend()
    plt.show()


n_epochs = 100
globalParameters = GlobalParameters(sLossP=0.04,
                                    tProfitP=0.12,
                                    testRatioP=0.8,
                                    useVolumeDataP=True,
                                    useCandleWickDataP=False,
                                    historyLengthP=120,
                                    useCryptoDataOnlyP=False,
                                    confidalityUpperLimitP=1,
                                    confidalityBottomLimitP=0.5)


loadDataFromDisk = True


training(loadDataFromDisk)
# testing()
check_accuracy(torch.load('testDataSet.pth'), device=device)
# testChart30(date(2021, 1, 1), date(2023, 1, 1),
#           'BTC-USD', model=model, device=device)
