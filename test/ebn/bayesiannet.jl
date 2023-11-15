@testset "Bayesian Networks" begin
    @testset "Beyesian Network" begin
        r = ContinuousRootNode(:R, Normal())
        v = DiscreteRootNode(:V, Dict(:yesV => 0.01, :noV => 0.99), Dict(:yesV => [Parameter(0, :v1)], :noV => [Parameter(1, :v1)]))
        s = DiscreteRootNode(:S, Dict(:yesS => 0.5, :noS => 0.5))
        t = DiscreteChildNode(:T, [v], Dict(
            [:yesV] => Dict(:yesT => 0.05, :noT => 0.95),
            [:noV] => Dict(:yesT => 0.01, :noT => 0.99))
        )
        l = DiscreteChildNode(:L, [s], Dict(
            [:yesS] => Dict(:yesL => 0.1, :noL => 0.9),
            [:noS] => Dict(:yesL => 0.01, :noL => 0.99))
        )
        f1 = DiscreteFunctionalNode(
            :F1, [r, v], [Model(df -> df.v1 .+ df.R, :f1)], df -> 0.8 .- df.f1, MonteCarlo(200)
        )
        nodes = [r, v, s, t, f1]
        @test_throws ErrorException("Network needs to be evaluated first") BayesianNetwork(nodes)

        nodes = [r, v, s, t]
        @test_throws ErrorException("Bayesian Network allows discrete node only!") BayesianNetwork(nodes)

        bn = BayesianNetwork([v, s, t, l])
        dag = SimpleDiGraph{Int64}(2, [[2], Int64[], [4], Int64[]], [Int64[], [1], Int64[], [3]])

        @test bn.dag == dag
        @test bn.name_to_index == Dict(:T => 4, :L => 2, :S => 1, :V => 3)
        @test issetequal(bn.nodes, [v, s, t, l])

        root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6), Dict(:yes => [Parameter(2.2, :y)], :no => [Parameter(5.5, :y)]))
        root3 = ContinuousRootNode(:z, Normal())

        functional_model = [Model(df -> (df.y .^ 2 + df.z .^ 2) ./ 2, :value1)]
        functional_simulation = MonteCarlo(300)
        functional_performance = df -> 1 .- 2 .* df.value1
        functional = DiscreteFunctionalNode(:f1, [root2, root3], functional_model, functional_performance, functional_simulation)

        ebn = EnhancedBayesianNetwork([root2, root3, functional])
        e_ebn = evaluate!(ebn)
        bn = BayesianNetwork(e_ebn)

        dag = SimpleDiGraph{Int64}(1, [[2], Int64[]], [Int64[], [1]])

        @test bn.dag == dag
        @test bn.name_to_index == Dict(:f1 => 2, :y => 1)
        @test issetequal(bn.nodes, e_ebn.nodes)
    end

    @testset "Additional CPDs" begin
        v = DiscreteRootNode(:V, Dict(:yesV => 0.01, :noV => 0.99))
        s = DiscreteRootNode(:S, Dict(:yesS => 0.5, :noS => 0.5))
        t = DiscreteChildNode(:T, [v], Dict(
            [:yesV] => Dict(:yesT => 0.05, :noT => 0.95),
            [:noV] => Dict(:yesT => 0.01, :noT => 0.99))
        )
        l = DiscreteChildNode(:L, [s], Dict(
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

