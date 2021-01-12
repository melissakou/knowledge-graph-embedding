# KGE

## Knowledge Graph Embedding - TransE
This is an implementation of knowledge graph embedding model - TransE.

## Requirement
* tensorflow >= 2.0

## How To Use
### Example

```python
import numpy as np
from KGE import TransE

X = np.array([['a', 'y', 'b'],
              ['b', 'y', 'a'],
              ['a', 'y', 'c'],
              ['c', 'y', 'a'],
              ['a', 'y', 'd'],
              ['c', 'y', 'd'],
              ['b', 'y', 'c'],
              ['f', 'y', 'e']])

model = TransE(embedding_size = 16, negative_ratio = 2, corrupt_side = 'h+t', margin = 0.5)
model.fit(train_X = X, val_X = None, epochs = 20, learning_rate = 0.001, batch_count = 1, early_stopping = None, log_path = 'tmp/TransE_logs')

## get entity or relation embedding:
model.get_embeddings(['a', 'b'], embedding_type = 'entity')
model.get_embeddings(['y'], embedding_type = 'relation')

## get all entities and relations embeddings in knowledge graph:
model.ent_emb.numpy()
model.rel_emb.numpy()
```

You can start tensorboard through the command line with the command:
```
tensorboard --logdir=tmp/TransE_logs --host=localhost --port=XXXX
```
You can monitor training and validation loss on tensorboard, also the distribution of embeddings.  
After finish the training procdure, you can view the entities and relations embedding projected into low dimensions in the Projector tab.



## API Documentation
## TransE
```python
class KGE.TransE.TransE(self, embedding_size, negative_ratio, corrupt_side, margin, norm = 2, batch_corrupt = False, norm_emb = False)
```
Described in [Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)


### Methods
| Method | Descrption |
| ------ | ---------- |
| ```__init__()``` | Initialize Embedding Model |
| ```fit()``` | Train an Embedding Model |
| ```get_embeddings()``` | Get the embeddings of entities or relations |
| ```get_rank()``` | Get the rank of triplets |
| ```convert_to_index()``` | Convert triplet to corresponding index |
| ```entity_to_index()``` | Convert entity to corresponding index|
| ```rel_to_index()``` | Convert relation to corresponding index|



### Attributes
| Attribute | Descrption |
| --------- | ---------- |
| ```entities``` | unique entities in ```train_X``` |
| ```relations``` | unique relations in ```train_X``` |
| ```dict_ent2id``` | a dictionary to map entity to corresponding index |
| ```dict_id2ent``` | a list to map index to corresponding entity |
| ```dict_rel2id``` | a dictionary to map relation to corresponding index |
| ```dict_id2rel``` | a list to map index to corresponding relation |
| ```ent_emb``` | ```tf.Tensor``` all entities embeddings, ordered by the entity index |
| ```rel_emb``` | ```tf.Tensor``` all relations embeddings, ordered by the relation index |
| ```loss_history``` | pairwise loss history of ```train_X``` during iteration |
| ```loss_history_val``` | pairwise loss history of ```val_X``` during iteration (if ```val_X``` is not None) |


<br>
<br>
<br>

> \_\___init__\_\_(embedding_size, negative_ratio, corrupt_side, margin, norm = 2, batch_corrupt = False, norm_emb = False)
Initialze the Embedding model.
>> __Parameters__
>> * __embedding_size__ (_int_): embedding dimension
>> * __negative_ratio__ (_int_): number of negative sample to be generated per positive triplet during training
>> * __corrupt_side__ (_string_): which side to be corrupted while generate negative sample. __'h'__ to corrupt head, __'t'__ to corrupt tail, __'h+t'__ to corrupt both side
>> * __margin__ (_float_): hyperparameter for pairwise loss $\gamma$, $\gamma>0$
>> * __norm__ (_int_): norm of scoring function (default: 2)
>> * __batch_corrupt__ (_bool_): whether to use entities only in current batch while generate negative sample (default: False)
>> * __norm_emb__ (_bool_): whether to normalize relations embeddings before training and normalize entities embedding every iteration


<br>


> __fit__(train_X, val_X, epochs, learning_rate, batch_count, early_stopping, log_path = None, log_step = 1)
Train an Embedding Model on ```train_X```
>> __Parameters__
>> * __train_X__ (_np.array_ or _string_):  
>>training triplets.  
>>If ```np.array```, shape should be (n, 3), store (h,r,t) in first, second, third column repectively.  
>>If ```string```, training triplets should be save under this __train_X__ folder with csv format, every csv files should have 3 columns for (h,r,t) respectively.  
>> * __val_X__ (_np.array_ or _string_):  
>>validation triplets.  
>>If ```np.array```, shape should be (n, 3), store (h,r,t) in first, second, third column repectively.  
>>If ```string```, validation triplets should be save under this __val_X__ folder with csv format, every csv files should have 3 columns for (h,r,t) respectively.  
>> * __epochs__ (_int_): number of epochs.
>> * __learninh_rate__ (_float_): learning_rate.
>> * __batch_count__ (_int_): number of batch.
>> * __early_stopping__ (_dict_): dictionary for early stopping setting, support followinh keys:
>>> * __patience__ (_int_): stop if validation loss doesn't improve over __patience__ iteration.
>>>  * __min_delta__ (_int_): 
>> * __log_path__ (_string_): path to log tensorboard summary.
>> * __log_step__ (_int_): interval to log tensorboard summary.  
