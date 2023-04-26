import pandas as pd

from prophet.utils.graph import Graph


class Shift(Graph.Function):

    def __init__(self, offset):
        self.offset = offset

    def compute(self, inputs):
        return inputs[0].shift(self.offset).fillna(0)


class Diff(Graph.Function):

    def __init__(self, distance, future=False):
        self.distance = distance
        self.future = future

    def compute(self, inputs):
        input_df = inputs[0]
        if self.future:
            return (-(input_df - input_df.shift(-self.distance))).fillna(0)
        else:
            return (input_df - input_df.shift(self.distance)).fillna(0)


class Satisfy(Graph.Function):

    def __init__(self, cond, window):
        self.cond = cond
        self.window = window

    def compute(self, inputs):
        line = inputs[0].iloc[:, 0].tolist()

        cnt = 0
        result = []
        for i in range(len(line)):
            if i - self.window >= 0 and self.cond(line[i - self.window]):
                cnt -= 1
            if self.cond(line[i]):
                cnt += 1
            result.append(cnt / min(i + 1, self.window))

        df = pd.DataFrame({'Satisfy': result})
        return df


class Keep(Graph.Function):

    def __init__(self, cond, window):
        self.cond = cond
        self.window = window

    def compute(self, inputs):
        line = inputs[0].iloc[:, 0].tolist()

        cnt = 0
        result = []
        for i in range(len(line)):
            if self.cond(line[i]):
                cnt = min(cnt + 1, self.window)
            else:
                cnt = 0
            result.append(cnt / min(i + 1, self.window))

        df = pd.DataFrame({'Keep': result})
        return df


class Ordered(Graph.Function):

    def __init__(self, window, weighted=False):
        self.window = window
        self.weighted = weighted

    def compute(self, inputs):
        line = inputs[0].iloc[:, 0].tolist()

        score, z = 0, 0
        result = []
        for i in range(len(line)):
            window = min(self.window, i + 1)

            for j in range(i - window + 1, i):
                if i - window >= 0:
                    delta = line[j] - line[i - window]
                    reward = 1 if not self.weighted else abs(delta)
                    is_ordered = ((0 if delta == 0 else delta / abs(delta)) + 1) / 2
                    score -= reward * is_ordered
                    z -= reward

                delta = line[i] - line[j]
                reward = 1 if not self.weighted else abs(delta)
                is_ordered = ((0 if delta == 0 else delta / abs(delta)) + 1) / 2
                score += reward * is_ordered
                z += reward

            result.append(0.5 if z == 0 else score / z)

        df = pd.DataFrame({'Ordered': result})
        return df


class RRank(Graph.Function):

    def __init__(self, window):
        self.window = window

    def compute(self, inputs):
        line = inputs[0].iloc[:, 0].tolist()

        result = []
        for i in range(len(line)):
            window = min(self.window, i + 1)

            sub_line = line[i + 1 - window: i + 1]
            max_value = max(sub_line)
            min_value = min(sub_line)

            w_rank = 0.5 if max_value == min_value else (line[i] - min_value) / (max_value - min_value)

            result.append(w_rank)

        df = pd.DataFrame({'RRank': result})
        return df


class Flip(Graph.Function):

    def compute(self, inputs):
        line1 = inputs[0].iloc[:, 0].tolist()
        line2 = inputs[1].iloc[:, 0].tolist()

        previous_state = 0
        result = []
        for i in range(len(line1)):
            delta = line1[i] - line2[i]
            current_state = 0 if delta == 0 else delta / abs(delta)
            if current_state == 0:
                result.append(0)
            elif previous_state == 0:
                result.append(current_state)
            elif current_state * previous_state == 1:
                result.append(0)
            else:
                result.append(current_state)
            previous_state = current_state

        df = pd.DataFrame({'Flip': result})
        return df


class Pearson(Graph.Function):

    def __init__(self, window, decimals=6):
        self.window = window
        self.decimals = decimals

    def compute(self, inputs):
        line = inputs[0].iloc[:, 0]
        line2 = inputs[1].iloc[:, 0]
        corr = line.rolling(self.window, min_periods=1).corr(line2)
        corr = corr.apply(lambda x: 0 if x > 1 or x < -1 else x)
        corr = corr.fillna(0).round(self.decimals)
        df = pd.DataFrame({'Pearson': corr})
        return df
