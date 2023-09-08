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

    @testset "Reduction of EnhancedBayesianNetworks" begin

        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
        root2 = DiscreteRootNode(:z, Dict(:y => 0.2, :n => 0.8), Dict(:y => [Parameter(1, :z)], :n => [Parameter(0, :z)]))

        states_child1 = Dict([:yes] => Dict(:a => 0.5, :b => 0.5), [:no] => Dict(:a => 0.5, :b => 0.5))
        child1 = DiscreteChildNode(:child1, [root1], states_child1, Dict(:a => [Parameter(1, :child1)], :b => [Parameter(0, :child1)]))

        distributions_child2 = Dict([:a] => Normal(), [:b] => Normal(2, 2))
        child2 = ContinuousChildNode(:child2, [child1], distributions_child2)

        model = Model(df -> sqrt.(df.child1 .^ 2 + df.child2 .- df.z .^ 2), :value1)

        models = Dict([:a, :y] => [model], [:b, :y] => [model], [:a, :n] => [model], [:b, :n] => [model])
        performances = Dict(
            [:a, :y] => df -> 1 .- 2 .* df.value1,
            [:b, :y] => df -> 1 .- 2 .* df.value1,
            [:a, :n] => df -> 1 .- 2 .* df.value1,
            [:b, :n] => df -> 1 .- 2 .* df.value1
        )
        simulations = Dict([:a, :y] => MonteCarlo(100), [:b, :y] => MonteCarlo(200), [:a, :n] => MonteCarlo(300), [:b, :n] => MonteCarlo(400))
        functional = DiscreteFunctionalNode(:functional, [child1, child2, root2], models, performances, simulations)

        ebn = EnhancedBayesianNetwork([root1, root2, child1, child2, functional])

        badjlist = Vector{Vector{Int}}([[], [], [2], [1, 3]])
        fadjlist = Vector{Vector{Int}}([[4], [3], [4], []])
        resulting_dag = DiGraph(3, fadjlist, badjlist)

        @test EnhancedBayesianNetworks._reduce_continuousnode(ebn.dag, ebn.name_to_index[:child2]) == resulting_dag

        rbn = reduce_ebn_standard(ebn)

        @test issetequal(get_children(rbn, root1), [child1])
        @test issetequal(get_parents(rbn, child1), [root1])

        functional_r = DiscreteFunctionalNode(:functional, [child1, root2], models, performances, simulations)
        @test issetequal(get_neighbors(rbn, child1), [root1, functional_r])

        badjlist = Vector{Vector{Int}}([[], [1, 4], [], [2, 3]])
        fadjlist = Vector{Vector{Int}}([[2], [4], [4], []])
        resulting_dag = DiGraph(3, fadjlist, badjlist)

        functional_r = DiscreteFunctionalNode(:functional, [child1, root2], models, performances, simulations)
        @test rbn.dag == resulting_dag
        @test issetequal(rbn.nodes, [root2, root1, child1, functional_r])
        @test rbn.name_to_index == Dict(:z => 3, :x => 1, :child1 => 2, :functional => 4)

        root1 = DiscreteRootNode(:x, Dict(:yes => 0.2, :no => 0.8))
        root2 = DiscreteRootNode(:alpha, Dict(:y => 0.4, :n => 0.6), Dict(:y => [Parameter(1, :alpha)], :n => [Parameter(0, :alpha)]))
        root3 = ContinuousRootNode(:z, Normal())

        standard1_states = Dict([:yes, :y] => Dict(:a => 0.2, :b => 0.8), [:no, :y] => Dict(:a => 0.3, :b => 0.7), [:yes, :n] => Dict(:a => 0.4, :b => 0.6), [:no, :n] => Dict(:a => 0.5, :b => 0.5))
        standard1_node = DiscreteChildNode(:α, [root1, root2], standard1_states, Dict(:a => [Parameter(1, :α)], :b => [Parameter(0, :α)]))

        standard2_states = Dict([:yes] => Normal(), [:no] => Normal(2, 2))
        standard2_node = ContinuousChildNode(:β, [root1], standard2_states)

        functional1_model = Model(df -> sqrt.(df.x .^ 2 + df.β .^ 2), :value1)
        functional1_models = Dict([:y] => [functional1_model], [:n] => [functional1_model])
        functional1_performances = Dict([:y] => df -> 1 .- 2 .* df.value1, [:n] => df -> 1 .- 2 .* df.value1)
        functional1_simulations = Dict([:y] => MonteCarlo(100), [:n] => MonteCarlo(200))
        functional1_node = DiscreteFunctionalNode(:f1, [root2, standard2_node], functional1_models, functional1_performances, functional1_simulations)


        functional2_model = Model(df -> sqrt.(df.α .^ 2 + df.z .^ 2), :value1)
        functional2_models = Dict([:a] => [functional2_model], [:b] => [functional2_model])
        functional2_performances = Dict([:a] => df -> 1 .- 2 .* df.value1, [:b] => df -> 1 .- 2 .* df.value1)
        functional2_simulations = Dict([:a] => MonteCarlo(150), [:b] => MonteCarlo(250))
        functional2_node = DiscreteFunctionalNode(:f2, [standard1_node, root3], functional2_models, functional2_performances, functional2_simulations)

        nodes = [standard1_node, root1, root3, root2, functional1_node, functional2_node, standard2_node]

        ebn = EnhancedBayesianNetwork(nodes)

        badjlist1 = Vector{Vector{Int}}([[], [], [1, 2], [3]])
        fadjlist1 = Vector{Vector{Int}}([[3], [3], [4], []])
        resulting_dag1 = DiGraph(3, fadjlist1, badjlist1)


        badjlist2 = Vector{Vector{Int}}([[], [], [1, 2]])
        fadjlist2 = Vector{Vector{Int}}([[3], [3], []])
        resulting_dag2 = DiGraph(2, fadjlist2, badjlist2)

        rbn1, rbn2 = reduce_ebn_markov_envelopes(ebn)

        functional2_node_r = DiscreteFunctionalNode(:f2, [standard1_node], functional2_models, functional2_performances, functional2_simulations)

        @test rbn1.dag == resulting_dag1
        @test issetequal(rbn1.nodes, [root1, root2, standard1_node, functional2_node_r])
        @test rbn1.name_to_index ∈ [Dict(:x => 1, :alpha => 2, :α => 3, :f2 => 4), Dict(:x => 2, :alpha => 1, :α => 3, :f2 => 4)]

        functional1_node_r = deepcopy(functional1_node)
        deleteat!(functional1_node_r.parents, findall(x -> x.name == :β, functional1_node_r.parents))
        push!(functional1_node_r.parents, root1)

        @test rbn2.dag == resulting_dag2
        @test issetequal(rbn2.nodes, [root2, root1, functional1_node_r])
        @test rbn2.name_to_index ∈ [Dict(:alpha => 1, :x => 2, :f1 => 3), Dict(:alpha => 2, :x => 1, :f1 => 3)]
    end

    @testset "Structural Reliability Problem" begin

        root1 = DiscreteRootNode(:x, Dict(:y => 0.2, :n => 0.8), Dict(:y => [Parameter(1, :x)], :n => [Parameter(0, :x), Parameter(5.6, :x1)]))
        root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6), Dict(:yes => [Parameter(2.2, :y)], :no => [Parameter(5.5, :y)]))
        root3 = ContinuousRootNode(:z, Normal())

        standard1_name = :α
        standard1_parents = [root1, root2]
        standard1_states = Dict(
            [:y, :yes] => Dict(:a => 0.2, :b => 0.8),
            [:n, :yes] => Dict(:a => 0.3, :b => 0.7),
            [:y, :no] => Dict(:a => 0.4, :b => 0.6),
            [:n, :no] => Dict(:a => 0.5, :b => 0.5)
        )
        standard1_parameters = Dict(:a => [Parameter(3, :α)], :b => [Parameter(10, :α)])
        standard1_node = DiscreteChildNode(standard1_name, standard1_parents, standard1_states, standard1_parameters)

        standard2_name = :β
        standard2_parents = [root1]
        standard2_states = Dict(
            [:y] => Normal(),
            [:n] => Normal(2, 2)
        )
        standard2_states = Dict(
            [:y] => Normal(),
            [:n] => Normal(2, 2)
        )
        standard2_node = ContinuousChildNode(standard2_name, standard2_parents, standard2_states)

        functional1_name = :f1
        functional1_parents = [root2, standard2_node]
        functional1_model1 = Model(df -> (df.y .^ 2 + df.β .^ 2) ./ 2, :value1)
        functional1_model2 = Model(df -> (df.y .^ 2 - df.β .^ 2) ./ 2, :value1)
        functional1_models = Dict(
            [:yes] => [functional1_model1],
            [:no] => [functional1_model2],
        )
        functional1_simulations = Dict(
            [:yes] => MonteCarlo(200),
            [:no] => MonteCarlo(300),
        )
        functional1_performances = Dict(
            [:yes] => df -> 1 .- 2 .* df.value1,
            [:no] => df -> 1 .- 2 .* df.value1,
        )
        functional1_node = DiscreteFunctionalNode(functional1_name, functional1_parents, functional1_models, functional1_performances, functional1_simulations)

        functional2_name = :f2
        functional2_parents = [standard1_node, root3]
        functional2_model = Model(df -> (df.α .^ 2 + df.z .^ 2) ./ 2, :value1)
        functional2_models = Dict(
            [:a] => [functional2_model],
            [:b] => [functional2_model]
        )
        functional2_simulations = Dict(
            [:a] => MonteCarlo(600),
            [:b] => MonteCarlo(800)
        )
        functional2_performances = Dict(
            [:a] => df -> 1 .- 2 .* df.value1,
            [:b] => df -> 1 .- 2 .* df.value1
        )

        functional2_node = DiscreteFunctionalNode(functional2_name, functional2_parents, functional2_models, functional2_performances, functional2_simulations)

        nodes = [standard1_node, root1, root3, root2, functional1_node, functional2_node, standard2_node]
        ebn = EnhancedBayesianNetwork(nodes)
        rbn1, rbn2 = reduce_ebn_markov_envelopes(ebn)

        srp_node = EnhancedBayesianNetworks._build_structuralreliabilityproblem_node(rbn1, ebn, rbn1.nodes[4])

        @test srp_node.name == :f2
        @test issetequal(srp_node.parents, [standard1_node])

        srps = Dict(
            [:a] => EnhancedBayesianNetworks.StructuralReliabilityProblem(functional2_node.models[[:a]], [standard1_node.parameters[:a][1], get_randomvariable(root3, [Symbol()])], functional2_node.performances[[:a]], functional2_node.simulations[[:a]]),
            [:b] => EnhancedBayesianNetworks.StructuralReliabilityProblem(functional2_node.models[[:b]], [standard1_node.parameters[:b][1], get_randomvariable(root3, [Symbol()])], functional2_node.performances[[:b]], functional2_node.simulations[[:b]]))
        @test srp_node.srps[[:a]].inputs == srps[[:a]].inputs
        @test srp_node.srps[[:a]].models == srps[[:a]].models
        @test srp_node.srps[[:a]].simulation == srps[[:a]].simulation
        @test srp_node.srps[[:a]].performance == srps[[:a]].performance
        @test srp_node.srps[[:b]].inputs == srps[[:b]].inputs
        @test srp_node.srps[[:b]].models == srps[[:b]].models
        @test srp_node.srps[[:b]].simulation == srps[[:b]].simulation
        @test srp_node.srps[[:b]].performance == srps[[:b]].performance
    end


end
