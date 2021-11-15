import datetime as dt
import matplotlib.pyplot as plt
import pandas_datareader as web

plt.style.use("dark_background")

ma_1 = 30
ma_2 = 100

start = dt.datetime.now() - dt.timedelta(days=356 * 3)
end = dt.datetime.now()

data = web.DataReader('FB', 'yahoo', start, end)
data[f'SMA_{ma_1}'] = data['Adj Close'].rolling(window=ma_1).mean()
data[f'SMA_{ma_2}'] = data['Adj Close'].rolling(window=ma_2).mean()

# display data from ma_2
data = data.iloc[ma_2:]

plt.plot(data['Adj Close'], label="Share Price", color="lightgray")
plt.plot(data[f'SMA_{ma_1}'], label=f"SMA_{ma_1}", color="orange")
plt.plot(data[f'SMA_{ma_2}'], label=f"SMA_{ma_2}", color="blue")
plt.legend(loc="upper left")
plt.show()

buy_signals = []
sell_signals = []
trigger = 0

# if ma_1 is larger than 2 and trigger not 1, we have buy signal at location x, leave sell_sig to empty, set trigger to 1, and vice verse
for x in range(len(data)):
    if data[f'SMA_{ma_1}'].iloc[x] > data[f'SMA_{ma_2}'].iloc[x] and trigger != 1:
        buy_signals.append(data['Adj Close'].iloc[x])
        sell_signals.append(float('nan'))
        trigger = 1
    elif data[f'SMA_{ma_1}'].iloc[x] < data[f'SMA_{ma_2}'].iloc[x] and trigger != -1:
        buy_signals.append(float('nan'))
        sell_signals.append(data['Adj Close'].iloc[x])
        trigger = -1
    else:
        buy_signals.append(float('nan'))
        sell_signals.append(float('nan'))
