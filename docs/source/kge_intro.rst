What's Knowledge Graph Embedding?
=================================

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

