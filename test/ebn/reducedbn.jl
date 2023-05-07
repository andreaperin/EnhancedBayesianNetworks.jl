@testset "Reduced Bayesian Networks" begin

    @testset "Node Elimination" begin
        badjlist = Vector{Vector{Int}}([[], [1], [1, 2]])
        fadjlist = Vector{Vector{Int}}([[2, 3], [3], []])
        dag = DiGraph(3, fadjlist, badjlist)

        badjlist2 = Vector{Vector{Int}}([[2], [], [1, 2]])
        fadjlist2 = Vector{Vector{Int}}([[3], [1, 3], []])
        resulting_dag = DiGraph(3, fadjlist2, badjlist2)

        @test_throws ErrorException("Invalid dag-link to be inverted") EnhancedBayesianNetworks._invert_link_dag(copy(dag), 2, 1)
        @test_throws ErrorException("Cyclic dag error") EnhancedBayesianNetworks._invert_link_dag(copy(dag), 1, 3)
        @test EnhancedBayesianNetworks._invert_link_dag(copy(dag), 1, 2) == resulting_dag

        badjlist = Vector{Vector{Int}}([[], [1], [1], [2, 3]])
        fadjlist = Vector{Vector{Int}}([[2, 3], [4], [4], []])
        dag = DiGraph(4, fadjlist, badjlist)

        badjlist2 = Vector{Vector{Int}}([[], [1], [1, 2, 4], [1, 2]])
        fadjlist2 = Vector{Vector{Int}}([[2, 3, 4], [3, 4], [], [3]])
        resulting_dag = DiGraph(6, fadjlist2, badjlist2)

        @test EnhancedBayesianNetworks._invert_link_nodes(copy(dag), 3, 4) == resulting_dag

        badjlist = Vector{Vector{Int}}([[], [1], [1, 2, 4], [1, 2]])
        fadjlist = Vector{Vector{Int}}([[2, 3, 4], [3, 4], [], [3]])
        dag = DiGraph(6, fadjlist, badjlist)

        badjlist2 = Vector{Vector{Int}}([[], [1], [1, 2]])
        fadjlist2 = Vector{Vector{Int}}([[2, 3], [3], []])
        resulting_dag = DiGraph(3, fadjlist2, badjlist2)

        @test_throws ErrorException("node to be eliminated must be a barren node") EnhancedBayesianNetworks._remove_barren_node(copy(dag), 2)
        @test EnhancedBayesianNetworks._remove_barren_node(copy(dag), 3) == resulting_dag
    end

    @testset "Reduction of EnhancedBayesianNetworks" begin

        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
        root2 = DiscreteRootNode(:z, Dict(:y => 0.2, :n => 0.8), Dict(:y => [Parameter(1, :z)], :n => [Parameter(0, :z)]))

        states_child1 = OrderedDict([:yes] => Dict(:a => 0.5, :b => 0.5), [:no] => Dict(:a => 0.5, :b => 0.5))
        child1 = DiscreteStandardNode(:child1, [root1], states_child1, Dict(:a => [Parameter(1, :child1)], :b => [Parameter(0, :child1)]))

        distributions_child2 = OrderedDict([:a] => Normal(), [:b] => Normal(2, 2))
        child2 = ContinuousStandardNode(:child2, [child1], distributions_child2)

        model = Model(df -> sqrt.(df.child1 .^ 2 + df.child2 .- df.z .^ 2), :value1)

        models = OrderedDict([:a, :y] => [model], [:b, :y] => [model], [:a, :n] => [model], [:b, :n] => [model])
        performances = OrderedDict(
            [:a, :y] => df -> 1 .- 2 .* df.value1,
            [:b, :y] => df -> 1 .- 2 .* df.value1,
            [:a, :n] => df -> 1 .- 2 .* df.value1,
            [:b, :n] => df -> 1 .- 2 .* df.value1
        )
        simulations = OrderedDict([:a, :y] => MonteCarlo(100), [:b, :y] => MonteCarlo(200), [:a, :n] => MonteCarlo(300), [:b, :n] => MonteCarlo(400))
        functional = DiscreteFunctionalNode(:functional, [child1, child2, root2], models, performances, simulations)

        ebn = EnhancedBayesianNetwork([root1, root2, child1, child2, functional])

        badjlist = Vector{Vector{Int}}([[], [], [2], [1, 3]])
        fadjlist = Vector{Vector{Int}}([[4], [3], [4], []])
        resulting_dag = DiGraph(3, fadjlist, badjlist)

        @test EnhancedBayesianNetworks._reduce_continuousnode(ebn.dag, ebn.name_to_index[:child2]) == resulting_dag

        @test reduce_ebn_standard(ebn).dag == resulting_dag
        @test reduce_ebn_standard(ebn).nodes == [root2, root1, child1, functional]
        @test reduce_ebn_standard(ebn).name_to_index == Dict(:z => 1, :x => 2, :child1 => 3, :functional => 4)

        root1 = DiscreteRootNode(:x, Dict(:yes => 0.2, :no => 0.8))
        root2 = DiscreteRootNode(:alpha, Dict(:y => 0.4, :n => 0.6), Dict(:y => [Parameter(1, :alpha)], :n => [Parameter(0, :alpha)]))
        root3 = ContinuousRootNode(RandomVariable(Normal(), :z))

        standard1_states = OrderedDict([:yes, :y] => Dict(:a => 0.2, :b => 0.8), [:no, :y] => Dict(:a => 0.3, :b => 0.7), [:yes, :n] => Dict(:a => 0.4, :b => 0.6), [:no, :n] => Dict(:a => 0.5, :b => 0.5))
        standard1_node = DiscreteStandardNode(:α, [root1, root2], standard1_states, Dict(:a => [Parameter(1, :α)], :b => [Parameter(0, :α)]))

        standard2_states = OrderedDict([:yes] => Normal(), [:no] => Normal(2, 2))
        standard2_node = ContinuousStandardNode(:β, [root1], standard2_states)

        functional1_model = Model(df -> sqrt.(df.x .^ 2 + df.β .^ 2), :value1)
        functional1_models = OrderedDict([:y] => [functional1_model], [:n] => [functional1_model])
        functional1_performances = OrderedDict([:y] => df -> 1 .- 2 .* df.value1, [:n] => df -> 1 .- 2 .* df.value1)
        functional1_simulations = OrderedDict([:y] => MonteCarlo(100), [:n] => MonteCarlo(200))
        functional1_node = DiscreteFunctionalNode(:f1, [root2, standard2_node], functional1_models, functional1_performances, functional1_simulations)


        functional2_model = Model(df -> sqrt.(df.α .^ 2 + df.z .^ 2), :value1)
        functional2_models = OrderedDict([:a] => [functional2_model], [:b] => [functional2_model])
        functional2_performances = OrderedDict([:a] => df -> 1 .- 2 .* df.value1, [:b] => df -> 1 .- 2 .* df.value1)
        functional2_simulations = OrderedDict([:a] => MonteCarlo(150), [:b] => MonteCarlo(250))
        functional2_node = DiscreteFunctionalNode(:f2, [standard1_node, root3], functional2_models, functional2_performances, functional2_simulations)

        nodes = [standard1_node, root1, root3, root2, functional1_node, functional2_node, standard2_node]

        ebn = EnhancedBayesianNetwork(nodes)

        badjlist1 = Vector{Vector{Int}}([[], [], [1, 2], [3]])
        fadjlist1 = Vector{Vector{Int}}([[3], [3], [4], []])
        resulting_dag1 = DiGraph(3, fadjlist1, badjlist1)


        badjlist2 = Vector{Vector{Int}}([[], [], [1, 2]])
        fadjlist2 = Vector{Vector{Int}}([[3], [3], []])
        resulting_dag2 = DiGraph(2, fadjlist2, badjlist2)

        @test reduce_ebn_markov_envelopes(ebn)[1].dag == resulting_dag1
        @test Set(reduce_ebn_markov_envelopes(ebn)[1].nodes) == Set([root1, root2, standard1_node, functional2_node])
        @test reduce_ebn_markov_envelopes(ebn)[1].name_to_index ∈ [Dict(:x => 1, :alpha => 2, :α => 3, :f2 => 4), Dict(:x => 2, :alpha => 1, :α => 3, :f2 => 4)]

        @test reduce_ebn_markov_envelopes(ebn)[2].dag == resulting_dag2
        @test Set(reduce_ebn_markov_envelopes(ebn)[2].nodes) == Set([root2, root1, functional1_node])
        @test reduce_ebn_markov_envelopes(ebn)[2].name_to_index ∈ [Dict(:alpha => 1, :x => 2, :f1 => 3), Dict(:alpha => 2, :x => 1, :f1 => 3)]
    end
end
