@testset "EBN reduction" begin
    @testset "Node Elimination" begin
        badjlist = Vector{Vector{Int}}([[], [1], [1, 2]])
        fadjlist = Vector{Vector{Int}}([[2, 3], [3], []])
        dag = DiGraph(3, fadjlist, badjlist)

        badjlist2 = Vector{Vector{Int}}([[2], [], [1, 2]])
        fadjlist2 = Vector{Vector{Int}}([[3], [1, 3], []])
        resulting_dag = DiGraph(3, fadjlist2, badjlist2)

        @test_throws ErrorException("Invalid dag-link to be inverted") EnhancedBayesianNetworks._invert_link_dag(copy(dag), 2, 1)
        @test_throws ErrorException("Cyclic dag error") EnhancedBayesianNetworks._invert_link_dag(copy(dag), 1, 3)
        @test EnhancedBayesianNetworks._invert_link_dag(deepcopy(dag), 1, 2) == resulting_dag

        badjlist = Vector{Vector{Int}}([[], [1], [1], [2, 3]])
        fadjlist = Vector{Vector{Int}}([[2, 3], [4], [4], []])
        dag = DiGraph(4, fadjlist, badjlist)

        badjlist2 = Vector{Vector{Int}}([[], [1], [1, 2, 4], [1, 2]])
        fadjlist2 = Vector{Vector{Int}}([[2, 3, 4], [3, 4], [], [3]])
        resulting_dag = DiGraph(6, fadjlist2, badjlist2)

        @test EnhancedBayesianNetworks._invert_link_nodes(deepcopy(dag), 3, 4) == resulting_dag

        badjlist = Vector{Vector{Int}}([[], [1], [1, 2]])
        fadjlist = Vector{Vector{Int}}([[2, 3], [3], []])
        resulting_dag = DiGraph(3, fadjlist, badjlist)
        @test EnhancedBayesianNetworks._reduce_continuousnode(deepcopy(dag), 3) == resulting_dag

        badjlist = Vector{Vector{Int}}([[], [1], [1, 2, 4], [1, 2]])
        fadjlist = Vector{Vector{Int}}([[2, 3, 4], [3, 4], [], [3]])
        dag = DiGraph(6, fadjlist, badjlist)

        badjlist2 = Vector{Vector{Int}}([[], [1], [1, 2]])
        fadjlist2 = Vector{Vector{Int}}([[2, 3], [3], []])
        resulting_dag = DiGraph(3, fadjlist2, badjlist2)

        @test_throws ErrorException("node to be eliminated must be a barren node") EnhancedBayesianNetworks._remove_barren_node(copy(dag), 2)
        @test EnhancedBayesianNetworks._remove_barren_node(copy(dag), 3) == resulting_dag
    end

    @testset "EBN Reduction" begin

        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
        root2 = DiscreteRootNode(:z, Dict(:y => 0.2, :n => 0.8), Dict(:y => [Parameter(1, :z)], :n => [Parameter(0, :z)]))

        states_child1 = Dict([:yes] => Dict(:a => 0.5, :b => 0.5), [:no] => Dict(:a => 0.5, :b => 0.5))
        child1 = DiscreteChildNode(:child1, [root1], states_child1, Dict(:a => [Parameter(1, :child1)], :b => [Parameter(0, :child1)]))

        distributions_child2 = Dict([:a] => Normal(), [:b] => Normal(2, 2))
        child2 = ContinuousChildNode(:child2, [child1], distributions_child2)

        model = Model(df -> sqrt.(df.child1 .^ 2 + df.child2 .- df.z .^ 2), :value1)
        simulation = MonteCarlo(400)
        performance = df -> 1 .- 2 .* df.value1
        functional = DiscreteFunctionalNode(:functional, [child1, child2, root2], [model], performance, simulation)

        ebn = EnhancedBayesianNetwork([root1, root2, child1, child2, functional])

        badjlist = Vector{Vector{Int}}([[], [], [2], [1, 3]])
        fadjlist = Vector{Vector{Int}}([[4], [3], [4], []])
        resulting_dag = DiGraph(3, fadjlist, badjlist)

        @test EnhancedBayesianNetworks._reduce_continuousnode(ebn.dag, ebn.name_to_index[:child2]) == resulting_dag

        rbn = reduce!(ebn)

        @test issetequal(get_children(rbn, root1), [child1])
        @test issetequal(get_parents(rbn, child1), [root1])

        functional_r = DiscreteFunctionalNode(:functional, [child1, root2], [model], performance, simulation)
        @test issetequal(get_neighbors(rbn, child1), [root1, functional_r])

        badjlist = Vector{Vector{Int}}([[], [1, 4], [], [2, 3]])
        fadjlist = Vector{Vector{Int}}([[2], [4], [4], []])
        resulting_dag = DiGraph(3, fadjlist, badjlist)

        functional_r = DiscreteFunctionalNode(:functional, [child1, root2], [model], performance, simulation)
        @test rbn.dag == resulting_dag
        @test issetequal(rbn.nodes, [root2, root1, child1, functional_r])
        @test rbn.name_to_index == Dict(:z => 3, :x => 1, :child1 => 2, :functional => 4)

        @test EnhancedBayesianNetworks._is_reducible(ebn) == true

        ## TODO missing test for "_is_reducible == false" 
    end
end