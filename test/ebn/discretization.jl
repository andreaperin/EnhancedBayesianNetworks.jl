@testset "Enhanced Bayesian Networks" begin
    @testset "Node Discretization" begin

        root1 = DiscreteRootNode(:x, Dict(:y => 0.2, :n => 0.8), Dict(:y => [Parameter(1, :x)], :n => [Parameter(0, :x), Parameter(5.6, :x1)]))
        root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6), Dict(:yes => [Parameter(2.2, :y)], :no => [Parameter(5.5, :y)]))
        root3 = ContinuousRootNode(:z, Normal())

        standard1_name = :α
        standard1_parents = [root1, root2]
        standard1_states = OrderedDict(
            [:y, :yes] => Dict(:a => 0.2, :b => 0.8),
            [:n, :yes] => Dict(:a => 0.3, :b => 0.7),
            [:y, :no] => Dict(:a => 0.4, :b => 0.6),
            [:n, :no] => Dict(:a => 0.5, :b => 0.5)
        )
        standard1_parameters = Dict(:a => [Parameter(3, :α)], :b => [Parameter(10, :α)])
        standard1_node = DiscreteStandardNode(standard1_name, standard1_parents, standard1_states, standard1_parameters)

        standard2_name = :β
        standard2_parents = [root1]
        standard2_states = OrderedDict(
            [:y] => Normal(),
            [:n] => Normal(2, 2)
        )
        standard2_states = OrderedDict(
            [:y] => Normal(),
            [:n] => Normal(2, 2)
        )
        standard2_node = ContinuousStandardNode(standard2_name, standard2_parents, standard2_states)

        functional2_name = :f2
        functional2_parents = [standard1_node, root3]
        functional2_model = Model(df -> (df.α .^ 2 + df.z .^ 2) ./ 2, :value1)
        functional2_models = OrderedDict(
            [:a] => [functional2_model],
            [:b] => [functional2_model]
        )
        functional2_simulations = OrderedDict(
            [:a] => MonteCarlo(600),
            [:b] => MonteCarlo(800)
        )
        functional2_performances = OrderedDict(
            [:a] => df -> 1 .- 2 .* df.value1,
            [:b] => df -> 1 .- 2 .* df.value1
        )

        functional2_node = DiscreteFunctionalNode(functional2_name, functional2_parents, functional2_models, functional2_performances, functional2_simulations)


        nodes = [standard1_node, root1, root3, root2, standard2_node, functional2_node]
        ebn = EnhancedBayesianNetwork(nodes)

        @test_throws ErrorException("non continuous range of intervals") EnhancedBayesianNetworks._discretize_node(ebn, root3, [[-1.1, 0.1], [0.2, 1]])
        @test_throws ErrorException("overlapping intervals") EnhancedBayesianNetworks._discretize_node(ebn, root3, [[-1.1, 0.1], [0.0, 1]])

        e_ebn = EnhancedBayesianNetworks._discretize_node(ebn, root3, [[-1.1, 0.1], [0.1, 1]])

        z_d_node = DiscreteRootNode(:z_d,
            Dict(
                Symbol("[-1.1, 0.1]") => 0.40416177633064637,
                Symbol("[-Inf, -1.1]") => 0.13566606094638262,
                Symbol("[1.0, Inf]") => 0.15865525393145696,
                Symbol("[0.1, 1.0]") => 0.30151690879151405
            )
        )

        z_c_node = ContinuousStandardNode(:z, [z_d_node],
            OrderedDict(
                [Symbol("[-1.1, 0.1]")] => truncated(Normal(0.0, 1.0), -1.1, 0.1),
                [Symbol("[0.1, 1.0]")] => truncated(Normal(0.0, 1.0), 0.1, 1.0),
                [Symbol("[-Inf, -1.1]")] => truncated(Normal(0.0, 1.0), -Inf, -1.1),
                [Symbol("[1.0, Inf]")] => truncated(Normal(0.0, 1.0), 1.0, Inf)
            )
        )

        functional2_r_node = DiscreteFunctionalNode(functional2_name, [standard1_node, z_c_node], functional2_models, functional2_performances, functional2_simulations)

        @test all(is_equal.(e_ebn, [root2, root1, standard2_node, standard1_node, functional2_r_node, z_c_node, z_d_node]))


        e_ebn = EnhancedBayesianNetworks._discretize_node(ebn, standard2_node, [[-1.1, 0], [0, 0.11]], 2)

        β_d_node = DiscreteStandardNode(:β_d, [root1], OrderedDict(
            [:y] => Dict(
                Symbol("[-Inf, -1.1]") => 0.13566606094638262,
                Symbol("[0.11, Inf]") => 0.45620468745768317,
                Symbol("[0.0, 0.11]") => 0.04379531254231683,
                Symbol("[-1.1, 0.0]") => 0.3643339390536174),
            [:n] => Dict(
                Symbol("[-Inf, -1.1]") => 0.060570758002059,
                Symbol("[0.11, Inf]") => 0.8276705619871281,
                Symbol("[0.0, 0.11]") => 0.013674184081414853,
                Symbol("[-1.1, 0.0]") => 0.09808449592939802)
        )
        )

        β_c_node = ContinuousStandardNode(:β, [β_d_node], OrderedDict(
            [Symbol("[-1.1, 0.0]")] => Uniform(-1.1, 0.0),
            [Symbol("[0.0, 0.11]")] => Uniform(0.0, 0.11),
            [Symbol("[-Inf, -1.1]")] => truncated(Normal(-1.1, 2.0), -Inf, -1.1),
            [Symbol("[0.11, Inf]")] => truncated(Normal(0.11, 2.0), 0.11, Inf)
        )
        )

        @test all(is_equal.(e_ebn, [root2, root3, root1, standard1_node, functional2_node, β_c_node, β_d_node]))
    end
end
