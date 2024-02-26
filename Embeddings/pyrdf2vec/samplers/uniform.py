from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import Sampler
import numpy as np

class UniformSampler(Sampler):
    """Defines the Uniform Weight Weight sampling strategy.

    This sampling strategy is the most straight forward approach. With this
    strategy, strongly connected entities will have a higher influence on the
    resulting embeddings.

    Attributes:
        inverse: True if Inverse Uniform Weight sampling satrategy must be
            used, False otherwise. Default to False.

    """

    def __init__(self, inverse=False):
        super().__init__(inverse)
        self.remote_supported = True

    def fit(self, kg: KG) -> None:
        """Fits the embedding network based on provided Knowledge Graph.

        Args:
            kg: The Knowledge Graph.

        """
        pass

    def get_weight(self, hop):
        """Gets the weights to the edge of the Knowledge Graph.

        Args:
            hop: The depth of the Knowledge Graph.

                A depth of eight means four hops in the graph, as each hop adds
                two elements to the sequence (i.e., the predicate and the
                object).

        Returns:
            The weights to the edge of the Knowledge Graph.

        """
        return 1


class RandomSampler(Sampler):
    """Defines the Uniform Weight Weight sampling strategy.

    This sampling strategy is the most straight forward approach. With this
    strategy, strongly connected entities will have a higher influence on the
    resulting embeddings.

    Attributes:
        inverse: True if Inverse Uniform Weight sampling satrategy must be
            used, False otherwise. Default to False.

    """

    def __init__(self, inverse=False):
        super().__init__(inverse)
        self.remote_supported = True

    def fit(self, kg: KG) -> None:
        """Fits the embedding network based on provided Knowledge Graph.

        Args:
            kg: The Knowledge Graph.

        """
        pass

    def get_weight(self, hop):
        """Gets the weights to the edge of the Knowledge Graph.

        Args:
            hop: The depth of the Knowledge Graph.

                A depth of eight means four hops in the graph, as each hop adds
                two elements to the sequence (i.e., the predicate and the
                object).

        Returns:
            The weights to the edge of the Knowledge Graph.

        """
        return 1


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

        # Sample a random neighbor and add them to visited if needed.
        rand_ix = np.random.choice(range(len(not_tag_neighbors)))
        if last:
            self.visited.add((not_tag_neighbors[rand_ix], len(walk)))
        return not_tag_neighbors[rand_ix]