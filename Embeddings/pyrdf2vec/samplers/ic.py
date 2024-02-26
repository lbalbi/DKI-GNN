from collections import defaultdict
from typing import Any, DefaultDict

from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler
import numpy as np

class ICSamplerV1(Sampler):
    """Defines the Object Frequency Weight sampling strategy.

    This sampling strategy is a node-centric object frequency approach. With
    this strategy, entities which have a high in degree get visisted more
    often.

    Attributes:
        inverse: True if Inverse Object Frequency Weight sampling strategy
            must be used, False otherwise. Default to False.
        split: True if Split Object Frequency Weight sampling strategy must
            be used, False otherwise. Default to False.

    """

    def __init__(self, dic_IC, inverse=False, split=False):
        super().__init__(inverse, split)
        self.dic_IC = dic_IC
        self.inverse = inverse

    def fit(self, kg: KG) -> None:
        """Fits the embedding network based on provided Knowledge Graph.

        Args:
            kg: The Knowledge Graph.

        """
        super().fit(kg)
        self.counts = {}
        for vertex in kg._vertices:
            if not vertex.predicate:

                if vertex.name in self.dic_IC:
                    self.counts[vertex.name] = self.dic_IC[vertex.name]

                else:
                    if self.inverse:
                        self.counts[vertex.name] = 1
                    else:
                        self.counts[vertex.name] = 0.00000001

    def get_weight(self, hop) -> int:
        """Gets the weights to the edge of the Knowledge Graph.

        Args:
            hop: The depth of the Knowledge Graph.

                A depth of eight means four hops in the graph, as each hop adds
                two elements to the sequence (i.e., the predicate and the
                object).

        Returns:
            The weights to the edge of the Knowledge Graph.

        """
        return self.counts[hop[1].name]



def naive_weighted_choices(weights, size=None):
    """
    Select indices at random, weighted by the iterator `weights` of
    arbitrary (non-negative) floats. That is, `x` will be returned
    with probability `weights[x]/sum(weights)`.
    For doing a single sample with arbitrary weights, this is much (5x
    or more) faster than numpy.random.choice, because the latter
    requires a lot of preprocessing (normalized probabilties), and
    does a lot of conversions/checks/preprocessing internally.
    """
    probs = np.cumsum(weights)
    total = probs[-1]
    if total == 0:
        # all weights were zero (probably), so we shouldn't choose anything
        return np.random.choice(range(len(weights)))

    thresholds = np.random.random() if size is None else np.random.random(size)
    idx = np.searchsorted(probs, thresholds * total, side="left")

    return idx



class ICSampler(Sampler):
    """Defines the Object Frequency Weight sampling strategy.

    This sampling strategy is a node-centric object frequency approach. With
    this strategy, entities which have a high in degree get visisted more
    often.

    Attributes:
        inverse: True if Inverse Object Frequency Weight sampling strategy
            must be used, False otherwise. Default to False.
        split: True if Split Object Frequency Weight sampling strategy must
            be used, False otherwise. Default to False.

    """

    def __init__(self, dic_IC, inverse=False, split=False):
        super().__init__(inverse, split)
        self.dic_IC = dic_IC
        self.inverse = inverse

    def fit(self, kg: KG) -> None:
        """Fits the embedding network based on provided Knowledge Graph.

        Args:
            kg: The Knowledge Graph.

        """
        super().fit(kg)
        self.counts = {}
        for vertex in kg._vertices:
            if not vertex.predicate:

                if vertex.name in self.dic_IC:
                    self.counts[vertex.name] = self.dic_IC[vertex.name]

                else:
                   self.counts[vertex.name] = 0



    def get_weight(self, hop) -> int:
        """Gets the weights to the edge of the Knowledge Graph.

        Args:
            hop: The depth of the Knowledge Graph.

                A depth of eight means four hops in the graph, as each hop adds
                two elements to the sequence (i.e., the predicate and the
                object).

        Returns:
            The weights to the edge of the Knowledge Graph.

        """
        return self.counts[hop[1].name]



    def sample_neighbor(self, kg: KG, walk, last):

        not_tag_neighbors = [
            x
            for x in kg.get_hops(walk[-1])
            if (x, len(walk)) not in self.visited
        ]

        # If there are no untagged neighbors, then tag
        # this vertex and return None
        if len(not_tag_neighbors) == 0:
            if len(walk) > 2:
                self.visited.add(((walk[-2], walk[-1]), len(walk) - 2))
            return None

        weights = [self.get_weight(hop) for hop in not_tag_neighbors]
        if self.inverse:
            weights = [max(weights) - (x - min(weights)) for x in weights]
        if self.split:
            weights = [
                w / self.degrees[v[1]]
                for w, v in zip(weights, not_tag_neighbors)
            ]

        # Sample a random neighbor and add them to visited if needed.
        rand_ix = naive_weighted_choices(weights)
        if last:
            self.visited.add((not_tag_neighbors[rand_ix], len(walk)))
        return not_tag_neighbors[rand_ix]






class PredICSampler(Sampler):
    """Defines the Object Frequency Weight sampling strategy.

    This sampling strategy is a node-centric object frequency approach. With
    this strategy, entities which have a high in degree get visisted more
    often.

    Attributes:
        inverse: True if Inverse Object Frequency Weight sampling strategy
            must be used, False otherwise. Default to False.
        split: True if Split Object Frequency Weight sampling strategy must
            be used, False otherwise. Default to False.

    """

    def __init__(self, dic_IC, inverse=False, split=False):
        super().__init__(inverse, split)
        self.dic_IC = dic_IC
        self.inverse = inverse

    def fit(self, kg: KG) -> None:
        """
        Fits the embedding network based on provided Knowledge Graph.
        Args:
            kg: The Knowledge Graph.

        """
        super().fit(kg)
        self.counts: DefaultDict[Any, Any] = defaultdict(int)
        for vertex in kg._vertices:
            if not vertex.predicate:

                if vertex.name in self.dic_IC:
                    self.counts[vertex.name] = self.dic_IC[vertex.name]

                else:
                   self.counts[vertex.name] = 0

            else:
                self.counts[vertex.name] += 1



    def get_weight(self, hop) -> int:
        """Gets the weights to the edge of the Knowledge Graph.

        Args:
            hop: The depth of the Knowledge Graph.

                A depth of eight means four hops in the graph, as each hop adds
                two elements to the sequence (i.e., the predicate and the
                object).

        Returns:
            The weights to the edge of the Knowledge Graph.

        """
        return self.counts[(hop[0].name, hop[1].name)]



    def sample_neighbor(self, kg: KG, walk, last):

        not_tag_neighbors = [
            x
            for x in kg.get_hops(walk[-1])
            if (x, len(walk)) not in self.visited
        ]

        # If there are no untagged neighbors, then tag
        # this vertex and return None
        if len(not_tag_neighbors) == 0:
            if len(walk) > 2:
                self.visited.add(((walk[-2], walk[-1]), len(walk) - 2))
            return None

        weights = [self.get_weight(hop) for hop in not_tag_neighbors]
        if self.inverse:
            weights = [max(weights) - (x - min(weights)) for x in weights]
        if self.split:
            weights = [
                w / self.degrees[v[1]]
                for w, v in zip(weights, not_tag_neighbors)
            ]

        # Sample a random neighbor and add them to visited if needed.
        rand_ix = naive_weighted_choices(weights)
        if last:
            self.visited.add((not_tag_neighbors[rand_ix], len(walk)))
        return not_tag_neighbors[rand_ix]
