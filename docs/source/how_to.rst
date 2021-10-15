How To Use
==========

Toy Example
-----------

Here is a toy example that demonstrates how to train a KGE model:

.. code-block:: python
    :linenos:
    
    import numpy as np
    from KGE.data_utils import index_kg, convert_kg_to_index
    from KGE.models.translating_based.TransE import TransE

    # load data
    train = np.loadtxt("./data/fb15k/train/train.csv", dtype=str, delimiter=',')
    valid = np.loadtxt("./data/fb15k/valid/valid.csv", dtype=str, delimiter=',')
    test = np.loadtxt("./data/fb15k/test/test.csv", dtype=str, delimiter=',')

    # index the kg data
    metadata = index_kg(train)
    
    # conver kg into index
    train = convert_kg_to_index(train, metadata["ent2ind"], metadata["rel2ind"])
    valid = convert_kg_to_index(valid, metadata["ent2ind"], metadata["rel2ind"])
    test = convert_kg_to_index(test, metadata["ent2ind"], metadata["rel2ind"])

    # initialized TransE model object
    model = TransE(
        embedding_params={"embedding_size": 32},
        negative_ratio=10,
        corrupt_side="h+t",
    )

    # train the model
    model.train(train_X=train, val_X=valid, metadata=metadata, epochs=10, batch_size=64,
                log_path="./tensorboard_logs", log_projector=True)

    # evaluate
    eval_result_filtered = model.evaluate(eval_X=test, corrupt_side="h", positive_X=np.concatenate((train, valid, test), axis=0))

First, you need to index the KG data. You can use :py:func:`KGE.data_utils.index_kg`
to index all entities and relation, this function return metadata of KG that mapping
all entities and relation to index. After creating the metadata, you can use
:py:func:`KGE.data_utils.convert_kg_to_index` to conver the string (h,r,t) into index.

After all preparation done for KG data, you can initialized the KGE model,
train the model, and evaluate model.

You can monitor the training and validation loss, distribution of model parameters on
TensorBoard using the command :code:`tensorboard --logdir=./tensorboard_logs`.

After the training procedure finished, the entities embedding are projected into lower
dimension and show on the Projector Tab in tensorboard
(if :code:`log_projector=True` is given when :code:`train()`).


Train KG from Disk File
------------------------

The toy example above demonstrates how to train KGE model from KG data stored in Numpy Array,
however, sometimes your KG may be too big that can not fit into memory, in this situation, you
can train the KG from the disk file without loading them into memory:

.. code-block:: python
    :linenos:

    from KGE.data_utils import index_kg, convert_kg_to_index
    from KGE.models.translating_based.TransE import TransE

    train = "./data/fb15k/train"
    valid = "./data/fb15k/valid"

    metadata = index_kg(train)

    convert_kg_to_index(train, metadata["ent2ind"], metadata["rel2ind"])
    convert_kg_to_index(valid, metadata["ent2ind"], metadata["rel2ind"])
    train = train + "_indexed"
    valid = valid + "_indexed"

    model = TransE(
        embedding_params={"embedding_size": 32},
        negative_ratio=10,
        corrupt_side="h+t"
    )

    model.train(train_X=train, val_X=valid, metadata=metadata, epochs=10, batch_size=64,
                log_path="./tensorboard_logs", log_projector=True)

We use the same function :py:func:`KGE.data_utils.index_kg` and
:py:func:`KGE.data_utils.convert_kg_to_index` to deal with KG data stored in disk.
If the input of :py:func:`KGE.data_utils.convert_kg_to_index` is a string path folder
but a numpy array, it won't return the indexed numpy array, instead it save the indexed KG
to the disk with suffix :code:`_indexed`.

Data folder can have multiple CSVs that store the different partitions of KG like that:

.. code-block::

    ./data/fb15k
    ├── test
    │   ├── test.csv
    │   ├── test1.csv
    │   └── test2.csv
    ├── train
    │   ├── train.csv
    │   ├── train1.csv
    │   └── train2.csv
    └── valid
        ├── valid.csv
        ├── valid1.csv
        └── valid2.csv


Train-Test Splitting KG Data
----------------------------

In the example above we use the benchmark dataset FB15K which is split
into the train, valid, test already, but when you bring your own KG data,
you should split data by yourself. Note that when splitting the KG data,
we need to guarantee that the entities in test data are also present in
the train data, otherwise, the entities not in the train would not have
embeddings being trained.

You can use :py:func:`KGE.data_utils.train_test_split_no_unseen` to split
the KG data that guarantee the entities in test data are also present in
the train data.

.. warning::
    :py:func:`KGE.data_utils.train_test_split_no_unseen` only support for numpy array.

.. code-block:: python

    import numpy as np
    from KGE.data_utils import train_test_split_no_unseen
    
    KG = np.array(
        [['DaVinci', 'painted', 'MonaLisa'],
         ['DaVinci', 'is_a', 'Person'],
         ['Lily', 'is_interested_in', 'DaVinci'],
         ['Lily', 'is_a', 'Person'],
         ['Lily', 'is_a_friend_of', 'James'],
         ['James', 'is_a', 'Person'],
         ['James', 'like', 'MonaLisa'],
         ['James', 'has_visited', 'Louvre'],
         ['James', 'has_lived_in', 'TourEiffel'],
         ['James', 'is_born_on', 'Jan,1,1984'],
         ['LaJocondeAWashinton', 'is_about', 'MonaLisa'],
         ['MonaLis', 'is_in', 'Louvre'],
         ['Louvre', 'is_located_in', 'Paris'],
         ['Paris', 'is_a', 'Place'],
         ['TourEiffel', 'is_located_in', 'Paris']]
    )

    train, test = train_test_split_no_unseen(KG, test_size=0.1, seed=12345)