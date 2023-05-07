@testset "Enhanced Bayesian Networks" begin
    @testset "DiGraphFunctions" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
        root2 = DiscreteRootNode(:y, Dict(:y => 0.4, :n => 0.6))
        root3 = ContinuousRootNode(RandomVariable(Normal(), :z))

        name = :child
        parents = [root1, root2, root3]
        distribution = OrderedDict([:yes, :y] => Normal(), [:no, :y] => Normal(1, 1), [:yes, :n] => Normal(2, 1), [:no, :n] => Normal(3, 1))
        child_node = ContinuousStandardNode(name, parents, distribution)

        nodes = [root1, root2, root3, child_node]

        @test EnhancedBayesianNetworks._build_digraph(nodes) == SimpleDiGraph{Int64}(3, [[4], [4], [4], Int64[]], [Int64[], Int64[], Int64[], [1, 2, 3]])

        @test EnhancedBayesianNetworks._topological_ordered_dag(nodes)[1] == SimpleDiGraph{Int64}(3, [[4], [4], [4], Int64[]], [Int64[], Int64[], Int64[], [1, 2, 3]])

        @test EnhancedBayesianNetworks._topological_ordered_dag(nodes)[2] == [root3, root2, root1, child_node]

        @test EnhancedBayesianNetworks._topological_ordered_dag(nodes)[3] == Dict(:z => 1, :y => 2, :x => 3, :child => 4)
    end

    @testset "EnhancedBayesianNetwork" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))

        states_child1 = OrderedDict([:yes] => Dict(:a => 0.5, :b => 0.5), [:no] => Dict(:a => 0.5, :b => 0.5))
        child1 = DiscreteStandardNode(:child1, [root1], states_child1, Dict(:a => [Parameter(3, :child1)], :b => [Parameter(0, :child1)]))

        distributions_child2 = OrderedDict([:a] => Normal(), [:b] => Normal(2, 2))
        child2 = ContinuousStandardNode(:child2, [child1], distributions_child2)

        model = Model(df -> sqrt.(df.child1 .^ 2 + df.child2 .^ 2), :value1)
        prf = Model(df -> 1 .- 2 .* df.value1, :value2)
        models = OrderedDict([:a] => [model, prf], [:b] => [model, prf])
        simulations = OrderedDict([:a] => MonteCarlo(100), [:b] => MonteCarlo(200))
        functional = DiscreteFunctionalNode(:functional, [child1, child2], models, simulations)

        badjlist = Vector{Vector{Int}}([[], [1], [2], [2, 3]])
        fadjlist = Vector{Vector{Int}}([[2], [3, 4], [4], []])

        nodes = [root1, child1, child2, functional]

        @test EnhancedBayesianNetwork(nodes).dag == EnhancedBayesianNetwork(DiGraph(4, fadjlist, badjlist), nodes, Dict(:x => 1, :child1 => 2, :child2 => 3, :functional => 4)).dag

        @test EnhancedBayesianNetwork(nodes).nodes == EnhancedBayesianNetwork(DiGraph(4, fadjlist, badjlist), nodes, Dict(:x => 1, :child1 => 2, :child2 => 3, :functional => 4)).nodes

        @test EnhancedBayesianNetwork(nodes).name_to_index == EnhancedBayesianNetwork(DiGraph(4, fadjlist, badjlist), nodes, Dict(:x => 1, :child1 => 2, :child2 => 3, :functional => 4)).name_to_index

        ebn = EnhancedBayesianNetwork(nodes)
        envelope = markov_envelope(ebn)[1]

        @test EnhancedBayesianNetworks._create_ebn_from_envelope(ebn, envelope).dag == ebn.dag

        @test EnhancedBayesianNetworks._create_ebn_from_envelope(ebn, envelope).nodes == ebn.nodes

        @test EnhancedBayesianNetworks._create_ebn_from_envelope(ebn, envelope).name_to_index == ebn.name_to_index
    end

    @testset "Nodes Operation" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
        root2 = DiscreteRootNode(:z, Dict(:yes => 0.2, :no => 0.8), Dict(:yes => [Parameter(3, :z)], :no => [Parameter(0, :z)]))

        states_child1 = OrderedDict([:yes] => Dict(:a => 0.5, :b => 0.5), [:no] => Dict(:a => 0.5, :b => 0.5))
        child1 = DiscreteStandardNode(:child1, [root1], states_child1, Dict(:a => [Parameter(3, :child1)], :b => [Parameter(0, :child1)]))

        distributions_child2 = OrderedDict([:a] => Normal(), [:b] => Normal(2, 2))
        child2 = ContinuousStandardNode(:child2, [child1], distributions_child2)

        model = Model(df -> sqrt.(df.child1 .^ 2 + df.child2 .- df.z .^ 2), :value1)
        prf = Model(df -> 1 .- 2 .* df.value1, :value2)
        models = OrderedDict([:a, :yes] => [model, prf], [:b, :yes] => [model, prf], [:a, :no] => [model, prf], [:b, :no] => [model, prf])
        simulations = OrderedDict([:a, :yes] => MonteCarlo(100), [:b, :yes] => MonteCarlo(200), [:a, :no] => MonteCarlo(300), [:b, :no] => MonteCarlo(400))
        functional = DiscreteFunctionalNode(:functional, [child1, child2, root2], models, simulations)

        @test_throws ErrorException("nodes state must have different symbols") EnhancedBayesianNetwork([root1, root2, child1, child2, functional])

        root1 = DiscreteRootNode(:x, Dict(:y => 0.5, :n => 0.5))

        states_child1 = OrderedDict([:y] => Dict(:a => 0.5, :b => 0.5), [:n] => Dict(:a => 0.5, :b => 0.5))
        child1 = DiscreteStandardNode(:child1, [root1], states_child1, Dict(:a => [Parameter(3, :child1)], :b => [Parameter(0, :child1)]))

        ebn = EnhancedBayesianNetwork([root1, root2, child1, child2, functional])

        @test Set(get_parents(ebn, child1)) == Set([root1])

        @test Set(get_children(ebn, child2)) == Set([functional])

        @test Set(get_neighbors(ebn, child2)) == Set([child1, functional])

        @test Set(markov_blanket(ebn, child2)) == Set([root2, functional, child1])

        @test Set.(markov_envelope(ebn)) == [Set([child2, root2, functional, child1])]
    end
end
