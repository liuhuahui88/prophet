import pandas as pd
import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go


class Figure:

    def __init__(self, action_name=None, value_names=None):
        self.action_name = action_name
        self.value_names = value_names

        self.vertical_spacing = 0.05
        self.row_1_height = 0.4
        self.row_2_height = 0.1
        self.row_3_height = 0.4

        self.ACTION_BUY = 'B'
        self.ACTION_SELL = 'S'

        self.DEFAULT_VALUE = 1
        self.DEFAULT_ACTION = self.ACTION_SELL

    def plot(self, df: pd.DataFrame, title='UNKNOWN', start_date=None, end_date=None):
        # Select record from start date to end date
        if start_date is not None:
            df = df[df.Date >= start_date]
        if end_date is not None:
            df = df[df.Date < end_date]

        # Create subplots and mention plot grid size
        fig = sp.make_subplots(shared_xaxes=True, rows=3, cols=1,
                               vertical_spacing=self.vertical_spacing,
                               subplot_titles=[title],
                               row_heights=[self.row_1_height, self.row_2_height, self.row_3_height])

        # Plot price on 1st row
        self.add_candle_stick(fig, 1, 1, df, 'Date', 'Open', 'High', 'Low', 'Close')

        # Plot volume on 2nd row
        df['VolumeColor'] = np.empty(len(df))
        df.loc[df.Open > df.Close, 'VolumeColor'] = 'green'
        df.loc[df.Open <= df.Close, 'VolumeColor'] = 'red'
        self.add_bar(fig, 2, 1, df, 'Date', 'Volume', 'VolumeColor')

        # Plot value on 3rd row
        if self.value_names is not None:
            for value_name in self.value_names:
                self.add_scatter(fig, 3, 1, df, 'Date', value_name)

        # Plot action
        if self.action_name is not None:
            self.add_action(fig, df, 'Date', self.action_name)

        # Hide price rangeslider plot
        fig.update(layout_xaxis_rangeslider_visible=False)

        # Add spike line
        fig.update_xaxes(showspikes=True, spikecolor="gray", spikemode="across")

        fig.show()

    @staticmethod
    def ensure(df, name, default_value):
        if name not in df:
            df[name] = default_value

    @staticmethod
    def add_scatter(fig, row, col, df, x_name, y_name, color=None):
        scatter = go.Scatter(x=df[x_name], y=df[y_name], name=y_name, marker=dict(color=color))
        fig.add_trace(scatter, row=row, col=col)

    @staticmethod
    def add_bar(fig, row, col, df, x_name, y_name, color_name=None, color=None):
        marker = dict(color=df[color_name] if color_name is not None else color)
        bar = go.Bar(x=df[x_name], y=df[y_name], name=y_name, marker=marker)
        fig.add_trace(bar, row=row, col=col)

    @staticmethod
    def add_candle_stick(fig, row, col, df, x_name, o_name, h_name, l_name, c_name):
        cs = go.Candlestick(x=df[x_name], open=df[o_name], high=df[h_name], low=df[l_name], close=df[c_name],
                            name="OHLC", increasing_line_color='red', decreasing_line_color='green')
        fig.add_trace(cs, row=row, col=col)

    def add_action(self, fig, df, x_name, y_name):
        shapes = []

        previous_action = self.ACTION_SELL
        start_dt = None
        for i in df.index:
            record = df.loc[i]
            action = record[y_name]

            if previous_action == self.ACTION_SELL and action == self.ACTION_BUY:
                start_dt = record[x_name]
            elif previous_action == self.ACTION_BUY and action == self.ACTION_SELL:
                end_dt = record[x_name]
                shapes.append(self.define_rectangle(start_dt, end_dt))

            previous_action = action

        if previous_action == self.ACTION_BUY:
            end_dt = record[x_name]
            shapes.append(self.define_rectangle(start_dt, end_dt))

        fig.update_layout(shapes=shapes)

    def define_rectangle(self, start, end):
        return dict(x0=start, x1=end, y0=1-self.row_1_height, y1=1, xref='x', yref='paper', line_width=2)
