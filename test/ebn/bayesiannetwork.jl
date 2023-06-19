@testset "Bayesian Networks" begin
    @testset "DiscreteStandardNode from Functional" begin
        root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6), Dict(:yes => [Parameter(2.2, :y)], :no => [Parameter(5.5, :y)]))
        root3 = ContinuousRootNode(:z, Normal())

        functional_model = Model(df -> (df.y .^ 2 + df.z .^ 2) ./ 2, :value1)
        functional_models = Dict(
            [:yes] => [functional_model],
            [:no] => [functional_model],
        )
        functional_simulations = Dict(
            [:yes] => MonteCarlo(200),
            [:no] => MonteCarlo(300),
        )
        functional_performances = Dict(
            [:yes] => df -> 1 .- 2 .* df.value1,
            [:no] => df -> 1 .- 2 .* df.value1,
        )
        pf = Dict([:yes] => 1.0, [:no] => 1.0)
        functional_node = DiscreteFunctionalNode(:f1, [root2, root3], functional_models, functional_performances, functional_simulations)
        functional_node.pf = pf

        discrete_standard_node = EnhancedBayesianNetworks._get_discretestandardnode(functional_node)

        @test discrete_standard_node.name == functional_node.name
        @test discrete_standard_node.parameters == Dict{Symbol,Vector{Parameter}}()
        @test discrete_standard_node.parents == functional_node.parents
        @test discrete_standard_node.states == Dict([:yes] => Dict(:f => 1, :s => 0), [:no] => Dict(:f => 1, :s => 0))
    end

    @testset "Beyesian Network" begin
        r = ContinuousRootNode(:R, Normal())
        v = DiscreteRootNode(:V, Dict(:yesV => 0.01, :noV => 0.99))
        s = DiscreteRootNode(:S, Dict(:yesS => 0.5, :noS => 0.5))
        t = DiscreteStandardNode(:T, [v], Dict(
            [:yesV] => Dict(:yesT => 0.05, :noT => 0.95),
            [:noV] => Dict(:yesT => 0.01, :noT => 0.99))
        )
        l = DiscreteStandardNode(:L, [s], Dict(
            [:yesS] => Dict(:yesL => 0.1, :noL => 0.9),
            [:noS] => Dict(:yesL => 0.01, :noL => 0.99))
        )
        nodes = [r, v, s, t]
        @test_throws ErrorException("Bayesian Network allows discrete node only!") BayesianNetwork(nodes)

        bn = BayesianNetwork([v, s, t, l])
        dag = SimpleDiGraph{Int64}(2, [[2], Int64[], [4], Int64[]], [Int64[], [1], Int64[], [3]])

        @test bn.dag == dag
        @test bn.name_to_index == Dict(:T => 4, :L => 2, :S => 1, :V => 3)
        @test issetequal(bn.nodes, [v, s, t, l])

        root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6), Dict(:yes => [Parameter(2.2, :y)], :no => [Parameter(5.5, :y)]))
        root3 = ContinuousRootNode(:z, Normal())

        functional_model = Model(df -> (df.y .^ 2 + df.z .^ 2) ./ 2, :value1)
        functional_models = Dict(
            [:yes] => [functional_model],
            [:no] => [functional_model],
        )
        functional_simulations = Dict(
            [:yes] => MonteCarlo(200),
            [:no] => MonteCarlo(300),
        )
        functional_performances = Dict(
            [:yes] => df -> 1 .- 2 .* df.value1,
            [:no] => df -> 1 .- 2 .* df.value1,
        )
        functional = DiscreteFunctionalNode(:f1, [root2, root3], functional_models, functional_performances, functional_simulations)

        rbn = reduce_ebn_standard(EnhancedBayesianNetwork([root2, root3, functional]))
        @test_throws ErrorException("rbn needs to evaluated!") BayesianNetwork(rbn)

        rbn = evaluate_ebn(EnhancedBayesianNetwork([root2, root3, functional]), true)[1]
        bn = BayesianNetwork(rbn)
        dag = SimpleDiGraph{Int64}(1, [[2], Int64[]], [Int64[], [1]])
        functional_evaluated = EnhancedBayesianNetworks._get_discretestandardnode(rbn.nodes[2])
        @test bn.dag == dag
        @test bn.name_to_index == Dict(:f1 => 2, :y => 1)
        @test issetequal(bn.nodes, [root2, functional_evaluated])
    end

    @testset "Conditional Probability Distribiution" begin
        v = DiscreteRootNode(:V, Dict(:yesV => 0.01, :noV => 0.99))
        s = DiscreteRootNode(:S, Dict(:yesS => 0.5, :noS => 0.5))
        t = DiscreteStandardNode(:T, [v], Dict(
            [:yesV] => Dict(:yesT => 0.05, :noT => 0.95),
            [:noV] => Dict(:yesT => 0.01, :noT => 0.99))
        )
        l = DiscreteStandardNode(:L, [s], Dict(
            [:yesS] => Dict(:yesL => 0.1, :noL => 0.9),
            [:noS] => Dict(:yesL => 0.01, :noL => 0.99))
        )
        bn = BayesianNetwork([v, s, t, l])
        cpd_s = get_cpd(bn, :S)
        cpd_t = get_cpd(bn, :T)

        @test cpd_s.distributions == Dict(Symbol[] => Dict(:noS => 0.5, :yesS => 0.5))
        @test cpd_t.distributions == Dict([:yesV] => Dict(:noT => 0.95, :yesT => 0.05),
            [:noV] => Dict(:noT => 0.99, :yesT => 0.01))

        @test isempty(cpd_s.parental_ncategories)
        @test cpd_t.parental_ncategories == [2]

        @test isempty(cpd_s.parents)
        @test cpd_t.parents == [:V]

        @test isempty(cpd_s.parents_states_mapping_dict)
        @test cpd_t.parents_states_mapping_dict == Dict(:V => Dict(:yesV => 1, :noV => 2))

        @test Set(cpd_s.states) == Set([:yesS, :noS])
        @test Set(cpd_t.states) == Set([:yesT, :noT])

        @test cpd_s.target == :S
        @test cpd_t.target == :T
    end
end

