import numpy as np

from KGE.data_utils import index_kg, convert_kg_to_index

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

metadata = index_kg(train)
train = convert_kg_to_index(train, metadata["ent2ind"], metadata["rel2ind"])
val = convert_kg_to_index(val, metadata["ent2ind"], metadata["rel2ind"])