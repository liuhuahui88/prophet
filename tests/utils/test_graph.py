from unittest import TestCase

from prophet.utils.graph import Graph


class TestGraph(TestCase):

    class Sum(Graph.Function):

        def __init__(self):
            self.counter = 0

        def compute(self, inputs):
            self.counter += 1
            return sum(inputs)

        def reset(self):
            self.counter = 0

    def test(self):
        g = Graph()

        g.register('a')
        g.register('b')

        self.assertRaises(KeyError, g.register, 'a')
        self.assertRaises(KeyError, g.register, 'b')

        self.assertRaises(KeyError, g.register, 'n', TestGraph.Sum(), ['c'])

        x_sum_function = TestGraph.Sum()
        y_sum_function = TestGraph.Sum()
        z_sum_function = TestGraph.Sum()
        sum_functions = [x_sum_function, y_sum_function, z_sum_function]

        g.register('x', x_sum_function, ['a', 'b'])
        g.register('y', y_sum_function, ['x', 'a'])
        g.register('z', z_sum_function, ['x', 'b'])

        ctx = dict(a=1, b=2)

        self.check(g, sum_functions, ctx, ['a'], dict(a=1), [0, 0, 0])
        self.check(g, sum_functions, ctx, ['b'], dict(b=2), [0, 0, 0])

        self.check(g, sum_functions, ctx, ['x'], dict(x=3), [1, 0, 0])
        self.check(g, sum_functions, ctx, ['y'], dict(y=4), [1, 1, 0])
        self.check(g, sum_functions, ctx, ['z'], dict(z=5), [1, 0, 1])
        self.check(g, sum_functions, ctx, ['x', 'y', 'z'], dict(x=3, y=4, z=5), [1, 1, 1])

    def check(self, graph, sum_functions, ctx, names, expected_output, expected_counters):
        for sum_function in sum_functions:
            sum_function.reset()
        expected_ctx = dict(ctx)

        output = graph.compute(names, ctx)

        self.assertEqual(expected_ctx, ctx)
        self.assertEqual(expected_output, output)
        self.assertEqual(expected_counters, [f.counter for f in sum_functions])
