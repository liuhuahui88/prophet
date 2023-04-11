import pandas as pd
import numpy as np

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


class Keep(Graph.Function):

    def __init__(self, cond):
        self.cond = cond

    def compute(self, inputs):
        line = inputs[0].iloc[:, 0].tolist()

        cnt = 0
        result = []
        for i in range(len(line)):
            if self.cond(line[i]):
                cnt += 1
            else:
                cnt = 0
            result.append(cnt)

        df = pd.DataFrame({'Keep': result})
        return df


class Flip(Graph.Function):

    def compute(self, inputs):
        line1 = inputs[0].iloc[:, 0].tolist()
        line2 = inputs[1].iloc[:, 0].tolist()

        previous_state = 0
        result = []
        for i in range(len(line1)):
            current_state = np.sign(line1[i] - line2[i])
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
