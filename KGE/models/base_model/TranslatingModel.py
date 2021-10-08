"""Translating Model"""

from .BaseModel import KGEModel

class TranslatingModel(KGEModel):
    """A base module for Semantic Based Embedding Model.

    Subclass of :class:`SemanticModel` can have thier own interation model.

    Attributes
    ----------
    embedding_params : dict
        embedding dimension parameters
    model_weights : dict of tf.Tensor
        model weights
    metadata : dict
        metadata for kg data
    negative_ratio : int
        number of negaative sample
    corrupt_side : str
        corrupt from which side while trainging
    score_fn : function
        scoring function
    score_params : dict
        score parameters for score_fn
    loss_fn : function
        loss function
    loss_params : dict
        loss parameters for loss_fn
    constraint : bool
        apply constraint or not
    ns_strategy : function
        negative sampling strategy
    batch_size : int
        batch size
    seed : int
        seed for shuffling data & embedding initialzation
    log_path : str
        path of tensorboard logging
    best_step : int
        best iteration step, only has value if check_early_stop is not None
    ckpt_manager : tf.train.CheckpointManager
        checkpoint manager
    best_ckpt : tf.train.Checkpoint
        best checkoint
    """

    def __init__(self, embedding_params, negative_ratio, corrupt_side, 
                 score_fn, score_params, loss_fn, loss_params,
                 ns_strategy, constraint, n_workers):
        """Initialize TranslatingModel.

        Parameters
        ----------
        embedding_params : dict
            embedding dimension parameters
        negative_ratio : int
            number of negative sample
        corrupt_side : str
            corrupt from which side while trainging, can be "h", "r", or "h+t"
        score_fn : function
            scoring function
        score_params : dict
            score parameters for score_fn
        loss_fn : function
            loss function
        loss_params : dict
            loss paraneters for loss_fn
        ns_strategy : function
            negative sampling strategy
        constraint : bool
            conduct constraint or not
        n_workers : int
            number of workers for negative sampling
        """
        
        super(TranslatingModel, self).__init__(embedding_params, negative_ratio, corrupt_side, 
                                               loss_fn, loss_params, ns_strategy, constraint, n_workers)
        self.score_fn = score_fn
        self.score_params = score_params