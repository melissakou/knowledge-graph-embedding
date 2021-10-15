What's Knowledge Graph Embedding?
=================================

Knowledge Graph (KG) is a directed, multi-relational, heterogeneous graph. It is composed of two components:
**entity** and **relation**. Figure 1 depicts an exemplary KG.

In the graph, each node is an entity, and each edge is the relation.

From the vivid example of KG in Fig.1, We can understand the obscure definition of KG clearly:

- **directed**: It is a directed graph obviously.
- **multi-relational**: There are many different relations on the graph such as *painted*, *is in*,
  *is interested in*, etc.
- **heterogeneous**: There are also have different types of entities, *MONA LISA* is an artwork,
  *DA VINCI* is a person, *PARIS* is a place, and so forth. An entity can be a concrete object like
  *MONA LISA* and *LOUVRE*, it can also be an abstract concept like *Person* and *Place*.

.. figure:: kg.png
    
    Figure 1: An exemplary Knowledge Graph

In general, we formulate a knowledge graph
:math:`\mathcal{K} \subseteq \mathbb{K}=\mathcal{E} \times \mathcal{R} \times \mathcal{E}`,
where :math:`\mathcal{E}` and :math:`\mathcal{R}` are set of entities and relations.
:math:`\mathcal{K}` comprise many trplets :math:`(h,r,t) \in \mathcal{K}` in which :math:`h,t \in \mathcal{E}`
represent a triplets' head and tail respectively, and :math:`r \in \mathcal{R}` represents its relationship.

For instance, the triplet :math:`(\it{DA~VINCI},~\it{painted},~\it{MONA~LISA})` in Fig.1,
:math:`\it{DA~VINCI}` is the head and :math:`\it{MONA~LISA}` is the tail entity, there has a relation
:math:`\it{painted}` from head to tail.

Knowledge Graph Embedding Modle learn the latent representation of entities :math:`e \in \mathcal{E}` and 
relations :math:`r \in \mathcal{R}` in a Knowledge Graph :math:`\mathcal{K}` that these laten representations
preseve the structural infomation in KG.

Here we taxnomize the Knowledge Graph Embedding Models into two:

- Translating-based Model
- Semantic-based Model


Translating-based Model
-----------------------

In translating-based knowledge graph embedding models, the head entity is translated by relation to the tail entity:

.. math::
    trans_r(h) \approx predicate(t)

Here we repesent the lhs :math:`trans_r(h)` as :math:`translation`,
the rhs :math:`predicate(t)` as :math:`predicate`.

Translating-based models use :py:mod:`scoring function <KGE.score.Score>` to measure the plausibility of a triplet,
the scoring function can be distance-based such as :py:mod:`Lp-distance <KGE.score.LpDistance>` or similarity-based
such as :py:mod:`dot prouct <KGE.score.Dot>` that measure the distance bewtween :math:`translation` and :math:`predicate`.

Every translating-based model can be formulated in this frame, for example:

+---------------------------------------------------------+------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
| Model                                                   | :math:`translation`                                                                      | :math:`predicate`                                                                         |
+=========================================================+==========================================================================================+===========================================================================================+
| :py:mod:`TransE <KGE.models.translating_based.TransE>`  | :math:`\textbf{e}_h + \textbf{r}_r`                                                      | :math:`\textbf{e}_t`                                                                      |
+---------------------------------------------------------+------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
|| :py:mod:`TransH <KGE.models.translating_based.TransH>` || :math:`{\textbf{e}_h}_{\perp} + \textbf{r}_r`                                           || :math:`{\textbf{e}_t}_{\perp}`                                                           |
||                                                        || :math:`{\textbf{e}_h}_{\perp} = \textbf{e}_h - \textbf{w}_r^T \textbf{e}_h\textbf{w}_r` || :math:`{\textbf{e}_t}_{\perp} = \textbf{e}_t - \textbf{w}_r^T \textbf{e}_t \textbf{w}_r` |
+---------------------------------------------------------+------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+
|| :py:mod:`TransR <KGE.models.translating_based.TransR>` || :math:`{\textbf{e}_h}_{\perp} + \textbf{r}_r`                                           || :math:`{\textbf{e}_t}_{\perp}`                                                           |
||                                                        || :math:`{\textbf{e}_h}_{\perp} = \textbf{e}_h \textbf{M}_r`                              || :math:`{\textbf{e}_t}_{\perp} = \textbf{e}_t \textbf{M}_r`                               |
+---------------------------------------------------------+------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------+


Semantic-based Model
--------------------

Semantic-based Model measures plausibility of triplets by matching latent semantics of entities and relations
in their laten vector representations. Each model difines how it measures the plausibility of triplets
:math:`f(h,r,t)`, for example:

+---------------------------------------------------------+------------------------------------------------------------+
| Model                                                   | :math:`f(h,r,t)`                                           |
+=========================================================+============================================================+
| :py:mod:`RESCAL <KGE.models.semantic_based.RESCAL>`     | :math:`\textbf{e}_h^{T} \textbf{R}_{r} \textbf{e}_t`       |
+---------------------------------------------------------+------------------------------------------------------------+
| :py:mod:`DistMult <KGE.models.semantic_based.DistMult>` | :math:`\textbf{e}_h^{T} diag(\textbf{R}_{r}) \textbf{e}_t` |
+---------------------------------------------------------+------------------------------------------------------------+

