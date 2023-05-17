import pytest
from src.Model import SimilarityMatrix

class TestSimilarityGraph:
    
    @pytest.fixture(scope = "class")
    def similarity_graph(self):
        sm =  SimilarityMatrix(5)
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

    
    def test_should_get_neighbors_return_correct_elements(self, similarity_graph):

        assert similarity_graph.get_neighbors(1).sort() == [0.9,0.2].sort() and \
                similarity_graph.get_neighbors(2).sort() == [0.5,0.9].sort()    and \
                similarity_graph.get_neighbors(3).sort() == [0.6,0.3,0.5].sort() and \
                similarity_graph.get_neighbors(4).sort() == [0.2,0.6].sort() and \
                similarity_graph.get_neighbors(5).sort() == [0.3].sort()


    