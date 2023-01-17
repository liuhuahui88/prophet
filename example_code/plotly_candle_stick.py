import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv('../data/history/600000.csv')
fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'])
                      ])
fig.show()
