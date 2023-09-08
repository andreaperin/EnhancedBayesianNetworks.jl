@testset "Enhanced Bayesian Networks" begin
    @testset "DiGraphFunctions" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
        root2 = DiscreteRootNode(:y, Dict(:y => 0.4, :n => 0.6))
        root3 = ContinuousRootNode(:z, Normal())

        name = :child
        parents = [root1, root2]
        distribution = Dict([:yes, :y] => Normal(), [:no, :y] => Normal(1, 1), [:yes, :n] => Normal(2, 1), [:no, :n] => Normal(3, 1))
        child_node = ContinuousChildNode(name, parents, distribution)

        nodes = [root1, root2, child_node]

        @test EnhancedBayesianNetworks._build_digraph(nodes) == SimpleDiGraph{Int64}(2, [[3], [3], Int64[]], [Int64[], Int64[], [1, 2]])

        dag, nodes, name_to_index = EnhancedBayesianNetworks._topological_ordered_dag(nodes)

        @test dag == SimpleDiGraph{Int64}(2, [[3], [3], Int64[]], [Int64[], Int64[], [1, 2]])

        @test issetequal(nodes, [root2, root1, child_node])

        @test name_to_index == Dict(:y => 1, :x => 2, :child => 3)
    end

    @testset "EnhancedBayesianNetwork" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
        root1_1 = DiscreteRootNode(:x, Dict(:y => 0.5, :n => 0.5))
        root1_2 = DiscreteRootNode(:p, Dict(:yes => 0.5, :no => 0.5))

        states_child1 = Dict([:yes] => Dict(:a => 0.5, :b => 0.5), [:no] => Dict(:a => 0.5, :b => 0.5))
        child1 = DiscreteChildNode(:child1, [root1], states_child1, Dict(:a => [Parameter(3, :child1)], :b => [Parameter(0, :child1)]))

        distributions_child2 = Dict([:a] => Normal(), [:b] => Normal(2, 2))
        child2 = ContinuousChildNode(:child2, [child1], distributions_child2)

        model = Model(df -> sqrt.(df.child1 .^ 2 + df.child2 .^ 2), :value1)
        df -> 1 .- 2 .* df.v
        models = Dict([:a] => [model], [:b] => [model])
        simulations = Dict([:a] => MonteCarlo(100), [:b] => MonteCarlo(200))
        performances = Dict([:a] => df -> 1 .- 2 .* df.v, [:b] => df -> 1 .- 2 .* df.v)
        functional = DiscreteFunctionalNode(:functional, [child1, child2], models, performances, simulations)

        badjlist = Vector{Vector{Int}}([[], [1], [2], [2, 3]])
        fadjlist = Vector{Vector{Int}}([[2], [3, 4], [4], []])

        nodes = [root1, child1, child2, functional]
        @test_throws ErrorException("nodes must have different names") EnhancedBayesianNetwork([root1, root1_1, child1, child2, functional])

        @test_throws ErrorException("nodes state must have different symbols") EnhancedBayesianNetwork([root1, root1_2, child1, child2, functional])

        ebn = EnhancedBayesianNetwork(nodes)

        @test ebn.dag == EnhancedBayesianNetwork(DiGraph(4, fadjlist, badjlist), nodes, Dict(:x => 1, :child1 => 2, :child2 => 3, :functional => 4)).dag

        @test issetequal(ebn.nodes, EnhancedBayesianNetwork(DiGraph(4, fadjlist, badjlist), nodes, Dict(:x => 1, :child1 => 2, :child2 => 3, :functional => 4)).nodes)

        @test ebn.name_to_index == EnhancedBayesianNetwork(DiGraph(4, fadjlist, badjlist), nodes, Dict(:x => 1, :child1 => 2, :child2 => 3, :functional => 4)).name_to_index

        envelope = markov_envelope(ebn)[1]

        @test EnhancedBayesianNetworks._create_ebn_from_envelope(ebn, envelope).dag == ebn.dag

        @test issetequal(EnhancedBayesianNetworks._create_ebn_from_envelope(ebn, envelope).nodes, ebn.nodes)

        @test EnhancedBayesianNetworks._create_ebn_from_envelope(ebn, envelope).name_to_index == ebn.name_to_index
    end

    @testset "Nodes Operation" begin
        root1 = DiscreteRootNode(:x, Dict(:y => 0.5, :n => 0.5))
        root2 = DiscreteRootNode(:z, Dict(:yes => 0.2, :no => 0.8), Dict(:yes => [Parameter(3, :z)], :no => [Parameter(0, :z)]))

        states_child1 = Dict([:y] => Dict(:a => 0.5, :b => 0.5), [:n] => Dict(:a => 0.5, :b => 0.5))
        child1 = DiscreteChildNode(:child1, [root1], states_child1, Dict(:a => [Parameter(3, :child1)], :b => [Parameter(0, :child1)]))

        distributions_child2 = Dict([:a] => Normal(), [:b] => Normal(2, 2))
        child2 = ContinuousChildNode(:child2, [child1], distributions_child2)

        model = Model(df -> sqrt.(df.child1 .^ 2 + df.child2 .- df.z .^ 2), :value1)

        models = Dict([:a, :yes] => [model], [:b, :yes] => [model], [:a, :no] => [model], [:b, :no] => [model])
        performances = Dict(
            [:a, :yes] => df -> 1 .- 2 .* df.value1,
            [:b, :yes] => df -> 1 .- 2 .* df.value1,
            [:a, :no] => df -> 1 .- 2 .* df.value1,
            [:b, :no] => df -> 1 .- 2 .* df.value1
        )
        simulations = Dict([:a, :yes] => MonteCarlo(100), [:b, :yes] => MonteCarlo(200), [:a, :no] => MonteCarlo(300), [:b, :no] => MonteCarlo(400))
        functional = DiscreteFunctionalNode(:functional, [child1, child2, root2], models, performances, simulations)

        ebn = EnhancedBayesianNetwork([root1, root2, child1, child2, functional])

        @test issetequal(get_parents(ebn, child1), [root1])

        @test issetequal(get_children(ebn, child2), [functional])

        @test issetequal(get_neighbors(ebn, child2), [child1, functional])

        @test issetequal(markov_blanket(ebn, child2), [child1, root2, functional])

        @test issetequal(markov_envelope(ebn)[1], [child1, root1, root2, functional, child2])

        @test isequal(EnhancedBayesianNetworks._get_node_given_state(ebn, :a), child1)
    end
end
