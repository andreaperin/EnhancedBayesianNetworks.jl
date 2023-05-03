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
        root2 = DiscreteRootNode(:z, Dict(:yes => 0.2, :no => 0.8))

        states_child1 = OrderedDict([:yes] => Dict(:a => 0.5, :b => 0.5), [:no] => Dict(:a => 0.5, :b => 0.5))
        child1 = DiscreteStandardNode(:child1, [root1], states_child1)

        distributions_child2 = OrderedDict([:a] => Normal(), [:b] => Normal(2, 2))
        child2 = ContinuousStandardNode(:child2, [child1], distributions_child2)

        model = Model(df -> sqrt.(df.child1 .^ 2 + df.child2 .- df.z .^ 2), :value1)
        prf = Model(df -> 1 .- 2 .* df.value1, :value2)
        models = OrderedDict([:a, :yes] => [model, prf], [:b, :yes] => [model, prf], [:a, :no] => [model, prf], [:b, :no] => [model, prf])

        functional = DiscreteFunctionalNode(:functional, [child1, child2, root2], models)

        ebn = EnhancedBayesianNetwork([root1, root2, child1, child2, functional])

        badjlist = Vector{Vector{Int}}([[], [], [2], [1, 3]])
        fadjlist = Vector{Vector{Int}}([[4], [3], [4], []])
        resulting_dag = DiGraph(3, fadjlist, badjlist)

        @test EnhancedBayesianNetworks._reduce_continuousnode(ebn.dag, ebn.name_to_index[:child2]) == resulting_dag

        @test reduce_ebn_standard(ebn).dag == resulting_dag
        @test reduce_ebn_standard(ebn).nodes == [root2, root1, child1, functional]
        @test reduce_ebn_standard(ebn).name_to_index == Dict(:z => 1, :x => 2, :child1 => 3, :functional => 4)

        root1 = DiscreteRootNode(:x, Dict(:yes => 0.2, :no => 0.8))
        root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6))
        root3 = ContinuousRootNode(RandomVariable(Normal(), :z))

        standard1_states = OrderedDict([:yes, :yes] => Dict(:a => 0.2, :b => 0.8), [:no, :yes] => Dict(:a => 0.3, :b => 0.7), [:yes, :no] => Dict(:a => 0.4, :b => 0.6), [:no, :no] => Dict(:a => 0.5, :b => 0.5))
        standard1_node = DiscreteStandardNode(:α, [root1, root2], standard1_states)

        standard2_states = OrderedDict([:yes] => Normal(), [:no] => Normal(2, 2))
        standard2_node = ContinuousStandardNode(:β, [root1], standard2_states)

        functional1_model = Model(df -> sqrt.(df.x .^ 2 + df.β .^ 2), :value1)
        functional1_performance = Model(df -> 1 .- 2 .* df.value1, :value2)
        functional1_models = OrderedDict([:yes] => [functional1_model, functional1_performance], [:no] => [functional1_model, functional1_performance])
        functional1_node = DiscreteFunctionalNode(:f1, [root2, standard2_node], functional1_models)


        functional2_model = Model(df -> sqrt.(df.α .^ 2 + df.z .^ 2), :value1)
        functional2_performance = Model(df -> 1 .- 2 .* df.value1, :value2)
        functional2_models = OrderedDict([:a] => [functional2_model, functional2_performance], [:b] => [functional2_model, functional2_performance])
        functional2_node = DiscreteFunctionalNode(:f2, [standard1_node, root3], functional2_models)

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
        @test reduce_ebn_markov_envelopes(ebn)[1].name_to_index ∈ [Dict(:x => 1, :y => 2, :α => 3, :f2 => 4), Dict(:x => 2, :y => 1, :α => 3, :f2 => 4)]

        @test reduce_ebn_markov_envelopes(ebn)[2].dag == resulting_dag2
        @test Set(reduce_ebn_markov_envelopes(ebn)[2].nodes) == Set([root2, root1, functional1_node])
        @test reduce_ebn_markov_envelopes(ebn)[2].name_to_index ∈ [Dict(:y => 1, :x => 2, :f1 => 3), Dict(:y => 2, :x => 1, :f1 => 3)]
    end
end
