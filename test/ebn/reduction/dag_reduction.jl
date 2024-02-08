@testset "DAG reduction" begin
    @testset "Links Inversion" begin

        badjlist2 = Vector{Vector{Int}}([[2], [], [1, 2]])
        fadjlist2 = Vector{Vector{Int}}([[3], [1, 3], []])
        resulting_dag = DiGraph(3, fadjlist2, badjlist2)

        badjlist = Vector{Vector{Int}}([[], [1], [1, 2]])
        fadjlist = Vector{Vector{Int}}([[2, 3], [3], []])
        dag = DiGraph(3, fadjlist, badjlist)
        @test EnhancedBayesianNetworks._invert_link_simple(dag, 1, 2) == resulting_dag

        @test_throws ErrorException("Invalid dag-link to invert") EnhancedBayesianNetworks._invert_link_simple(deepcopy(dag), 2, 1)

        @test_throws ErrorException("Cyclic dag error") EnhancedBayesianNetworks._invert_link_simple(dag, 1, 3)

        badjlist = Vector{Vector{Int}}([[], [1], [1], [2, 3]])
        fadjlist = Vector{Vector{Int}}([[2, 3], [4], [4], []])
        dag = DiGraph(4, fadjlist, badjlist)

        badjlist2 = Vector{Vector{Int}}([[], [1], [1, 2, 4], [1, 2]])
        fadjlist2 = Vector{Vector{Int}}([[2, 3, 4], [3, 4], [], [3]])
        resulting_dag = DiGraph(6, fadjlist2, badjlist2)
        @test EnhancedBayesianNetworks._invert_link(dag, 3, 4) == resulting_dag

        badjlist = Vector{Vector{Int}}([[], [1], [1, 2, 4], [1, 2]])
        fadjlist = Vector{Vector{Int}}([[2, 3, 4], [3, 4], [], [3]])
        dag = DiGraph(6, fadjlist, badjlist)
    end

    badjlist2 = Vector{Vector{Int}}([[], [1], [1, 2]])
    fadjlist2 = Vector{Vector{Int}}([[2, 3], [3], []])
    resulting_dag = DiGraph(3, fadjlist2, badjlist2)
    @test_throws ErrorException("Cannot eliminate a not-barren node") EnhancedBayesianNetworks._remove_barren_node(dag, 2)
    @test EnhancedBayesianNetworks._remove_barren_node(dag, 3) == resulting_dag

    badjlist = Vector{Vector{Int}}([[], [1], [1], [2, 3]])
    fadjlist = Vector{Vector{Int}}([[2, 3], [4], [4], []])
    dag = DiGraph(4, fadjlist, badjlist)

    badjlist = Vector{Vector{Int}}([[], [1], [1, 2]])
    fadjlist = Vector{Vector{Int}}([[2, 3], [3], []])
    resulting_dag = DiGraph(3, fadjlist, badjlist)
    @test EnhancedBayesianNetworks._reduce_dag_single(dag, 3) == resulting_dag

    badjlist = Vector{Vector{Int}}([[], [1], [1, 2], [2, 3]])
    fadjlist = Vector{Vector{Int}}([[2, 3], [3, 4], [4], []])
    dag = DiGraph(4, fadjlist, badjlist)
    @test EnhancedBayesianNetworks._is_reducible(dag, [3]) == true
    @test EnhancedBayesianNetworks._is_reducible(dag, [3, 4]) == false

end