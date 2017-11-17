from data import DATA_PATH
from math import sin, cos, atan2, radians, sqrt
from json import JSONDecoder

def load_states():
    """Load the coordinates of all the state outlines and return them
    in a dictionary, from names to shapes lists.

    >>> len(load_states()['HI'])  # Hawaii has 5 islands
    5
    """
    json_data_file = open(DATA_PATH + 'states.json', encoding='utf8')
    states = JSONDecoder().decode(json_data_file.read())
    for state, shapes in states.items():
        for index, shape in enumerate(shapes):
            if type(shape[0][0]) == list:  # the shape is a single polygon
                assert len(shape) == 1, 'Multi-polygon shape'
                shape = shape[0]
            shapes[index] = [make_position(*reversed(pos)) for pos in shape]
    return states
