import numpy as np
import tensorflow as tf

from .utils import ns_with_same_type

class NegativeSampler:
    """ A base module for negative sampler.
    """
    def __init__(self):
        """ Initialized negative sampler
        """
        raise NotImplementedError("subclass of NegativeSampler should implement __init__() to init class")

    def __call__(self):
        """ Confuct negative sampling
        """
        raise NotImplementedError("subclass of NegativeSampler should implement __call__() to conduct negative sampling")


class UniformStrategy(NegativeSampler):
    """ An implementation of uniform negative sampling

    Uniform sampling is the most simple negative sampling strategy, usually is
    the default setting of knowledge graph embedding models. It sample entities
    from all entites with uniform distribution, and replaces either head or tail
    entity.
    """

    def __init__(self, sample_pool):
        """ Initialize UniformStrategy negative sampler.

        Parameters
        ----------
        sample_pool : tf.Tensor
            entities pool that used to sample.
        """
        self.sample_pool = sample_pool

    def __call__(self, X, negative_ratio, side):
        """ perform negative sampling

        Parameters
        ----------
        X : tf.Tensor
            positive triplets to be corrupt.
        negative_ratio : int
            number of negative sample.
        side : str
            corrup from which side, can be :code:`'h'` or :code:`'t'`

        Returns
        -------
        tf.Tensor
            sampling entities
        """

        self.sample_pool = tf.cast(self.sample_pool, X.dtype)
        sample_index = tf.random.uniform(
            shape=[X.shape[0] * negative_ratio, 1],
            minval=0, maxval=len(self.sample_pool), dtype=self.sample_pool.dtype
        )
        sample_entities = tf.gather_nd(self.sample_pool, sample_index)

        return sample_entities

class TypedStrategy(NegativeSampler):
    """ An implementation of typed negative sampling strategy.

    Typed negative sampling consider the entities' type, for example, for the
    positive triplet :math:`(MonaLisa, is\_in, Louvre)`, we may generate illogical
    negative triplet such as :math:`(MonaLis, is\_in, DaVinci)`. So Typed negative
    sampling strategy consider the type of entity to be corrupt, if we want
    to replace *Louvre*, we only sample the entities which have same type
    with *Louvre*.

    .. caution::
        When using :py:mod:`TypedStrategy <KGE.ns_strategy.TypedStrategy>`, :code:`metadata` should contains
        key :code:`'ind2type'` to indicate the entities' type when calling
        :py:func:`train() <KGE.models.base_model.BaseModel.KGEModel.train>`.
    """
    def __init__(self, pool, metadata):
        """ Initialize TypedStrategy negative sampler.

        Parameters
        ----------
        pool : :ref:`multiprocessing.pool.Pool <https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool>`
            multiprocessing pool for parallel.
        metadata : dict
            metadata that store the entities' type information.
        """
        self.pool = pool
        self.metadata = metadata

    def __call__(self, X, negative_ratio, side):
        """ perform negative sampling

        Parameters
        ----------
        X : tf.Tensor
            positive triplets to be corrupt.
        negative_ratio : int
            number of negative sample.
        side : str
            corrup from which side, can be :code:`'h'` or :code:`'t'`

        Returns
        -------
        tf.Tensor
            sampling entities
        """
        
        from itertools import repeat

        if side == "h":
            ref_type = X[:, 0].numpy()
        elif side == "t":
            ref_type = X[:, 2].numpy()

        if self.pool is not None:
            sample_entities = self.pool.starmap(
                ns_with_same_type,
                zip(ref_type, repeat(self.metadata), repeat(negative_ratio))
            )
        else:
            sample_entities = list(map(
                lambda x: ns_with_same_type(x, self.metadata, negative_ratio),
                ref_type
            ))

        sample_entities = tf.constant(np.concatenate(sample_entities), dtype=X.dtype)

        return sample_entities