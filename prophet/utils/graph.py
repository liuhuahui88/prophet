from abc import ABC, abstractmethod


class Graph:

    class Function(ABC):

        @abstractmethod
        def compute(self, inputs):
            pass

    class Node:

        def __init__(self, name, function, dependents):
            self.name = name
            self.dependents = dependents
            self.function = function

    def __init__(self):
        self.nodes = {}

    def register(self, name, function=None, dependent_names=None):
        if name in self.nodes:
            raise KeyError('{} has been registered'.format(name))
        dependents = []
        if dependent_names is not None:
            for dependent_name in dependent_names:
                if dependent_name not in self.nodes:
                    raise KeyError('{} depended by {} has not been registered'.format(dependent_name, name))
                dependents.append(self.nodes[dependent_name])
        self.nodes[name] = Graph.Node(name, function, dependents)

    def compute(self, names, ctx):
        ctx = dict(ctx)
        for name in names:
            if name not in self.nodes:
                raise KeyError('{} has not been registered'.format(name))
            self.compute_inner(self.nodes[name], ctx)
        return {k: v for k, v in ctx.items() if k in names}

    def compute_inner(self, node: Node, ctx):
        if node.name not in ctx:
            inputs = []
            for dependent in node.dependents:
                inputs.append(self.compute_inner(dependent, ctx))
            ctx[node.name] = node.function.compute(inputs)
        return ctx[node.name]
