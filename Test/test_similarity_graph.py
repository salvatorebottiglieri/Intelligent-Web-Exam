import math
import numpy as np
import pytest
from src.Model import SimilarityMatrix
from src.WeightsComputation import compute_similarity

class TestSimilarityGraph:
    
    @pytest.fixture(scope = "class")
    def similarity_graph(self):
        sm =  SimilarityMatrix(5, "ml-latest-small")
        sm.add_edge(1, 2, 0.9)
        sm.add_edge(1, 4, 0.2)
        sm.add_edge(2, 3, 0.5)
        sm.add_edge(3, 4, 0.6)
        sm.add_edge(3, 5, 0.3)
        yield sm

        del sm
    
    def test_should_are_connected_return_true(self, similarity_graph):
        assert similarity_graph.are_connected(1, 2) == True

    def test_should_are_connected_return_false(self, similarity_graph):
        assert similarity_graph.are_connected(1, 5) == False

    

