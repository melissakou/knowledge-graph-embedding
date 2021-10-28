import os
import scipy
import unittest
import collections
import numpy as np
import pandas as pd
import tensorflow  as tf

from KGE.utils import check_path_exist_and_create
from KGE.data_utils import index_kg, convert_kg_to_index, set_tf_iterator
from KGE.data_utils import train_test_split_no_unseen, calculate_data_size

class TestDataUtils(unittest.TestCase):

    global train, val
    
    train = np.array(
        [['DaVinci', 'painted', 'MonaLisa'],
            ['Lily', 'is_interested_in', 'DaVinci'],
            ['Lily', 'is_a', 'Person'],
            ['Lily', 'is_a_friend_of', 'James'],
            ['James', 'like', 'MonaLisa'],
            ['James', 'has_visited', 'Louvre'],
            ['James', 'has_lived_in', 'TourEiffel'],
            ['James', 'is_born_on', 'Jan,1,1984'],
            ['LaJocondeAWashinton', 'is_about', 'MonaLisa'],
            ['MonaLis', 'is_in', 'Louvre'],
            ['Paris', 'is_a', 'Place'],
            ['TourEiffel', 'is_located_in', 'Paris']]
    )

    val = np.array(
        [['DaVinci', 'is_a', 'Person'],
        ['James', 'is_a', 'Person'],
        ['Louvre', 'is_located_in', 'Paris'],]
    )


    def test_index_kg(self):
        metadata = index_kg(train)
        [self.assertTrue(k in metadata.keys()) for k in ['ent2ind', 'ind2ent', 'rel2ind', 'ind2rel']]
        self.assertEqual(len(metadata["ent2ind"].keys()), len(metadata["ind2ent"]))
        self.assertEqual(len(metadata["rel2ind"].keys()), len(metadata["ind2rel"]))

    def test_calculate_data_size(self):
        check_path_exist_and_create("./tmp/train")
        pd.DataFrame(train).to_csv("./tmp/train/train.csv", index=False, header=False)
        self.assertEqual(calculate_data_size("./tmp/train"), len(train))
    
    def test_convert_kg_to_index(self):
        metadata = index_kg(train)
        
        # test np.array
        indexed_kg = convert_kg_to_index(train, metadata["ent2ind"], metadata["rel2ind"])
        self.assertEqual(train.shape, indexed_kg.shape)
        self.assertEqual(indexed_kg.dtype, int)

        # test csv
        check_path_exist_and_create("./tmp/train")
        pd.DataFrame(train).to_csv("./tmp/train/train.csv", index=False, header=False)
        convert_kg_to_index("./tmp/train", metadata["ent2ind"], metadata["rel2ind"])
        self.assertTrue(os.path.exists("./tmp/train_indexed"))
        self.assertEqual(os.listdir("./tmp/train"), os.listdir("./tmp/train_indexed"))
        self.assertEqual(calculate_data_size("./tmp/train"), calculate_data_size("./tmp/train_indexed"))

        indexed_kg = pd.read_csv("./tmp/train_indexed/train.csv", header=None)
        self.assertEqual(train.shape, indexed_kg.shape)
        [self.assertTrue(pd.api.types.is_integer_dtype(indexed_kg[i])) for i in range(indexed_kg.shape[1])]


    def test_set_tf_iterator(self):
        global train
        batch_size = 5
        metadata = index_kg(train)
        train_indexed = convert_kg_to_index(train, metadata["ent2ind"], metadata["rel2ind"])

        # test np.array
        data_iter = set_tf_iterator(data=train_indexed, batch_size=batch_size, shuffle=False)
        self.assertIsInstance(data_iter, collections.Iterator)
        batch = next(data_iter)
        self.assertEqual(len(batch), batch_size)
        self.assertEqual(batch.dtype, train_indexed.dtype)

        # test csv
        check_path_exist_and_create("./tmp")
        pd.DataFrame(train_indexed).to_csv("./tmp/train.csv", index=False, header=False)
        data_iter = set_tf_iterator(data="./tmp", batch_size=batch_size, shuffle=False)
        batch = next(data_iter)
        self.assertEqual(len(batch), batch_size)
        self.assertEqual(batch.dtype, tf.int32)

    def test_train_test_split_no_unseen(self):
        global train
        tot_sample_size = len(train) + len(val)
        test_size_prop = 0.1
        train_data, test_data = train_test_split_no_unseen(np.concatenate((train, val), axis=0),
                                                           test_size=test_size_prop, seed=1234)
        self.assertEqual(len(train_data), tot_sample_size - int(tot_sample_size*test_size_prop))
        self.assertEqual(len(test_data), int(tot_sample_size*test_size_prop))
        self.assertEqual(train_data.dtype, train.dtype)
        self.assertEqual(test_data.dtype, train.dtype)


if __name__ == "__main__":
    unittest.main()