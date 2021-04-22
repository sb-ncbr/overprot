from typing import Generic, TypeVar, List, Sized, Container


K = TypeVar('K')  # Type of keys
V = TypeVar('V')  # Type of values
X = TypeVar('X')  # Input type of a function
Y = TypeVar('Y')  # Output type of a function


class AbstractSimilarityTree(Generic[K, V], Sized, Container):
    '''Abstract class defining basic methods for similarity-searching trees'''

    def insert(self, key: K, value: V) -> None:
        '''Insert a new element with given key and value into the tree. The key must be unique.'''
        raise NotImplementedError

    def delete(self, key: K) -> None:
        '''Delete an existing element from the tree.'''
        raise NotImplementedError
    
    def kNN_query(self, query: K, k: int, include_query: bool = False) -> List[K]:
        '''Perform classical kNN query, return the k nearest neighbors to query (including itself iff include_query==True). query must be present in the tree.'''
        raise NotImplementedError
    
    def get_distance(self, key1: K, key2: K) -> float:
        '''Get the distance between two elements present in the tree'''
        raise NotImplementedError
    
    def get_statistics(self) -> str:
        '''Get some basic statistics about the tree, like number of nodes, number of calculated distances etc.'''
        class_name = type(self).__name__
        return f'Tree statistics not available for {class_name}. Implement {class_name}.get_statistics().'
