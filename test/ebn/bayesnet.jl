@testset "Bayesian Networks" begin
    @testset "Beyesian Network" begin
        r = ContinuousNode{UnivariateDistribution}(:R, DataFrame(:Prob => Normal()))
        v = DiscreteNode(:V, DataFrame(:V => [:yesV, :noV], :Prob => [0.01, 0.99]), Dict(:yesV => [Parameter(0, :v1)], :noV => [Parameter(1, :v1)]))
        s = DiscreteNode(:S, DataFrame(:S => [:yesS, :nos], :Prob => [0.5, 0.5]))
        t = DiscreteNode(:T, DataFrame(:V => [:yesV, :yesV, :noV, :noV], :T => [:yesT, :noT, :yesT, :noT], :Prob => [0.05, 0.95, 0.01, 0.99]))
        l = DiscreteNode(:L, DataFrame(:S => [:yesS, :yesS, :noS, :noS], :L => [:yesL, :noL, :yesL, :noL], :Prob => [0.1, 0.9, 0.01, 0.99]))
        f1 = DiscreteFunctionalNode(
            :F1, [Model(df -> df.v1 .+ df.R, :f1)], df -> 0.8 .- df.f1, MonteCarlo(200)
        )
        nodes = [r, v, s, t, f1]
        @test_throws ErrorException("node/s [:F1] are functional nodes. evaluate the related EnhancedBayesianNetwork structure before!") BayesianNetwork(nodes)

        nodes = [r, v, s, t]
        @test_throws ErrorException("node/s [:R] are continuous. Use EnhancedBayesianNetwork structure!") BayesianNetwork(nodes)

        r = ContinuousNode{UnivariateDistribution}(:V, DataFrame(:Prob => Normal()))
        nodes = [r, v, s, t, f1]
        @test_throws ErrorException("network nodes names must be unique") BayesianNetwork(nodes)

        r = ContinuousNode{UnivariateDistribution}(:R, DataFrame(:Prob => Normal()))
        v = DiscreteNode(:V, DataFrame(:V => [:yesT, :noT], :Prob => [0.01, 0.99]), Dict(:yesT => [Parameter(0, :v1)], :noT => [Parameter(1, :v1)]))
        nodes = [r, v, s, t, f1]
        @test_throws ErrorException("network nodes states must be unique") BayesianNetwork(nodes)

        # v_imp = DiscreteRootNode(:V, Dict(:yesV => [0.01, 0.3], :noV => [0.7, 0.99]), Dict(:yesV => [Parameter(0, :v1)], :noV => [Parameter(1, :v1)]))
        # nodes = [v_imp, s, t]
        # @test_throws ErrorException("node/s [:V] are imprecise. Use CrealNetwork structure!") BayesianNetwork(nodes)

        # v = DiscreteRootNode(:V, Dict(:yesV => 0.01, :noV => 0.99), Dict(:yesV => [Parameter(0, :v1)], :noV => [Parameter(1, :v1)]))
        # s = DiscreteRootNode(:S, Dict(:yesS => 0.5, :noS => 0.5))
        # t = DiscreteChildNode(:T, Dict(
        #     [:yesV] => Dict(:yesT => 0.05, :noT => 0.95),
        #     [:noV] => Dict(:yesT => 0.01, :noT => 0.99))
        # )
        # l = DiscreteChildNode(:L, Dict(
        #     [:yesS] => Dict(:yesL => 0.1, :noL => 0.9),
        #     [:noS] => Dict(:yesL => 0.01, :noL => 0.99))
        # )
        # bn = BayesianNetwork([v, s, t, l])
        # add_child!(bn, v, t)
        # add_child!(bn, s, l)
        # order!(bn)

        # @test bn.adj_matrix == sparse([
        #     0.0 0.0 1.0 0.0;
        #     0.0 0.0 0.0 1.0;
        #     0.0 0.0 0.0 0.0;
        #     0.0 0.0 0.0 0.0])
        # @test bn.topology_dict == Dict(:T => 3, :L => 4, :V => 1, :S => 2,)
        # @test issetequal(bn.nodes, [v, s, t, l])


        # ebn = EnhancedBayesianNetwork([v, s, t, l])
        # add_child!(ebn, v, t)
        # add_child!(ebn, s, l)
        # order!(ebn)
        # reduce!(ebn)
        # bn2 = BayesianNetwork(ebn)

        # @test bn2 == bn
    end

    # @testset "get_cpd" begin
    #     v = DiscreteRootNode(:V, Dict(:yesV => 0.01, :noV => 0.99))
    #     s = DiscreteRootNode(:S, Dict(:yesS => 0.5, :noS => 0.5))
    #     t = DiscreteChildNode(:T, Dict(
    #         [:yesV] => Dict(:yesT => 0.05, :noT => 0.95),
    #         [:noV] => Dict(:yesT => 0.01, :noT => 0.99))
    #     )
    #     l = DiscreteChildNode(:L, Dict(
    #         [:yesS] => Dict(:yesL => 0.1, :noL => 0.9),
    #         [:noS] => Dict(:yesL => 0.01, :noL => 0.99))
    #     )
    #     bn = BayesianNetwork([v, s, t, l])
    #     add_child!(bn, v, t)
    #     add_child!(bn, s, l)
    #     order!(bn)
    #     cpd_s = get_cpd(bn, :S)
    #     cpd_t = get_cpd(bn, :T)

    #     @test cpd_s.probabilities == Dict(Symbol[] => Dict(:noS => 0.5, :yesS => 0.5))
    #     @test cpd_t.probabilities == Dict([:yesV] => Dict(:noT => 0.95, :yesT => 0.05),
    #         [:noV] => Dict(:noT => 0.99, :yesT => 0.01))

    #     @test isempty(cpd_s.parental_ncategories)
    #     @test cpd_t.parental_ncategories == [2]

    #     @test isempty(cpd_s.parents)
    #     @test cpd_t.parents == [:V]

    #     @test isempty(cpd_s.parents_states_mapping_dict)
    #     @test cpd_t.parents_states_mapping_dict == Dict(:V => Dict(:yesV => 1, :noV => 2))

    #     @test Set(cpd_s.states) == Set([:yesS, :noS])
    #     @test Set(cpd_t.states) == Set([:yesT, :noT])

    #     @test cpd_s.target == :S
    #     @test cpd_t.target == :T

    #     @test get_cpd(bn, 2) == cpd_s
    #     @test get_cpd(bn, s) == cpd_s
    # end
end

