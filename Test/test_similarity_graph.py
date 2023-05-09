import pytest
from src.Model import SimilarityMatrix

class TestSimilarityGraph:
    
    @pytest.fixture(scope = "class")
    def similarity_graph(self):
        return SimilarityMatrix(10)
    
    def test_should_are_connected_return_true(self, similarity_graph):
        assert similarity_graph.are_connected(4, 8) == True

    def test_should_are_connected_return_false(self, similarity_graph):
        assert similarity_graph.are_connected(1, 2) == False


    def test_should_get_neighbors_return_empy_list(self, similarity_graph):
        assert similarity_graph.get_neighbors(4) == []

    
    def test_should_get_neighbors_return_list_with_one_element(self, similarity_graph):
        similarity_graph.add_edge(4, 8, 0.5)

        print(similarity_graph.graph)

        assert similarity_graph.get_neighbors(4) == [0.5] and \
                similarity_graph.get_neighbors(8) == [0.5]


    