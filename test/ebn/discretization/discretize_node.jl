@testset "Node Discretization" begin

    @testset "Format Intervals" begin
        discretization = ExactDiscretization([-Inf, 0, Inf])
        root = ContinuousRootNode(:z1, Normal(), discretization)

        formatted_intervals = EnhancedBayesianNetworks._format_interval(root)
        @test formatted_intervals == [[-Inf, 0.0], [0.0, Inf]]

        discretization = ExactDiscretization([0, Inf])
        root = ContinuousRootNode(:z1, Normal(), discretization)

        @test_logs (:warn, "Minimum intervals value 0.0 >= support lower bound -Inf. Lower bound will be used as intervals start.") EnhancedBayesianNetworks._format_interval(root)

        discretization = ExactDiscretization([-Inf, 0.0])
        root = ContinuousRootNode(:z1, Normal(), discretization)

        @test_logs (:warn, "Maximum intervals value 0.0 <= support upper bound Inf. Upper bound will be used as intervals end.") EnhancedBayesianNetworks._format_interval(root)
    end

    @testset "Root node" begin
        discretization = ExactDiscretization([0, Inf])
        root = ContinuousRootNode(:z1, Normal(), discretization)

        continuous_node, discretized_root = @suppress EnhancedBayesianNetworks._discretize(root)

        @test discretized_root.name == :z1_d

        @test continuous_node.distributions[[Symbol("[-Inf, 0.0]")]] == truncated(Normal(), -Inf, 0.0)
        @test continuous_node.distributions[[Symbol("[0.0, Inf]")]] == truncated(Normal(), 0.0, Inf)

        @test discretized_root.states[Symbol("[-Inf, 0.0]")] == 0.5
        @test discretized_root.states[Symbol("[0.0, Inf]")] == 0.5

        discretization = ExactDiscretization([-Inf, 0, Inf])

        continuous_node, discretized_root = @suppress EnhancedBayesianNetworks._discretize(root)

        @test discretized_root.name == :z1_d

        @test continuous_node.distributions[[Symbol("[-Inf, 0.0]")]] == truncated(Normal(), -Inf, 0.0)
        @test continuous_node.distributions[[Symbol("[0.0, Inf]")]] == truncated(Normal(), 0.0, Inf)

        @test discretized_root.states[Symbol("[-Inf, 0.0]")] == 0.5
        @test discretized_root.states[Symbol("[0.0, Inf]")] == 0.5
    end

    @testset "Child node" begin
        discretization = ApproximatedDiscretization([-Inf, -1, 0, 1, Inf], 1.5)

        root = DiscreteRootNode(:x, Dict(:y => 0.2, :n => 0.8))

        states = Dict(
            [:y] => Normal(),
            [:n] => Normal(2, 2)
        )

        child = ContinuousChildNode(:β, [root], states, discretization)

        approximated_node, discretized_child = @suppress EnhancedBayesianNetworks._discretize(child)

        @test approximated_node.distributions[[Symbol("[-Inf, -1.0]")]] == truncated(Normal(-1, 1.5), -Inf, -1)
        @test approximated_node.distributions[[Symbol("[-1.0, 0.0]")]] == Uniform(-1, 0)
        @test approximated_node.distributions[[Symbol("[0.0, 1.0]")]] == Uniform(0, 1.0)
        @test approximated_node.distributions[[Symbol("[1.0, Inf]")]] == truncated(Normal(1, 1.5), 1, Inf)

    end
end