@testset "Node Discretization" begin

    @testset "Format Intervals" begin
        discretization = ExactDiscretization([-10, 0, 10])
        root = ContinuousRootNode(:z1, truncated(Normal(0, 1); lower=-10, upper=10), discretization)

        formatted_intervals = EnhancedBayesianNetworks._format_interval(root)
        @test formatted_intervals == [[-10, 0.0], [0.0, 10]]

        discretization = ExactDiscretization([-9, 10])
        root = ContinuousRootNode(:z1, truncated(Normal(0, 1); lower=-10, upper=10), discretization)

        @test_logs (:warn, "Minimum intervals value -9 > support lower bound -10.0. Lower bound will be used as intervals start.") EnhancedBayesianNetworks._format_interval(root)

        discretization = ExactDiscretization([-11, 10])
        root = ContinuousRootNode(:z1, truncated(Normal(0, 1); lower=-10, upper=10), discretization)

        @test_logs (:warn, "Minimum intervals value -11 < support lower bound -10.0. Lower bound will be used as intervals start.") EnhancedBayesianNetworks._format_interval(root)

        discretization = ExactDiscretization([-10, 9])
        root = ContinuousRootNode(:z1, truncated(Normal(0, 1); lower=-10, upper=10), discretization)

        @test_logs (:warn, "Maximum intervals value 9 < support upper bound 10.0. Upper bound will be used as intervals end.") EnhancedBayesianNetworks._format_interval(root)

        discretization = ExactDiscretization([-10, 11])
        root = ContinuousRootNode(:z1, truncated(Normal(0, 1); lower=-10, upper=10), discretization)

        @test_logs (:warn, "Maximum intervals value 11 > support upper bound 10.0. Upper bound will be used as intervals end.") EnhancedBayesianNetworks._format_interval(root)

        intervals = [[-Inf, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, Inf]]
        σ = 2

        approx = [
            truncated(Normal(-1.0, 2.0); lower=-Inf, upper=-1.0),
            Uniform(-1.0, 0.0),
            Uniform(0.0, 1.0),
            truncated(Normal(1.0, 2.0); lower=1.0, upper=Inf)
        ]

        @test approx == EnhancedBayesianNetworks._approximate(intervals, σ)

        dist = Normal()
        probs = [0.15865525393145702, 0.341344746068543, 0.34134474606854304, 0.15865525393145696]

        @test all(isapprox.(EnhancedBayesianNetworks._discretize(dist, intervals), probs))
    end

    @testset "Root node" begin
        discretization = ExactDiscretization([0, Inf])
        root = ContinuousRootNode(:z1, Normal(), discretization)

        continuous_node, discretized_root = @suppress EnhancedBayesianNetworks._discretize(root)

        @test discretized_root.name == :z1_d

        @test continuous_node.distribution[[Symbol("[-Inf, 0.0]")]] == truncated(Normal(), -Inf, 0.0)
        @test continuous_node.distribution[[Symbol("[0.0, Inf]")]] == truncated(Normal(), 0.0, Inf)

        @test discretized_root.states[Symbol("[-Inf, 0.0]")] == 0.5
        @test discretized_root.states[Symbol("[0.0, Inf]")] == 0.5

        discretization = ExactDiscretization([-Inf, 0, Inf])

        continuous_node, discretized_root = @suppress EnhancedBayesianNetworks._discretize(root)

        @test discretized_root.name == :z1_d

        @test continuous_node.distribution[[Symbol("[-Inf, 0.0]")]] == truncated(Normal(), -Inf, 0.0)
        @test continuous_node.distribution[[Symbol("[0.0, Inf]")]] == truncated(Normal(), 0.0, Inf)

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

        @test approximated_node.distribution[[Symbol("[-Inf, -1.0]")]] == truncated(Normal(-1, 1.5), -Inf, -1)
        @test approximated_node.distribution[[Symbol("[-1.0, 0.0]")]] == Uniform(-1, 0)
        @test approximated_node.distribution[[Symbol("[0.0, 1.0]")]] == Uniform(0, 1.0)
        @test approximated_node.distribution[[Symbol("[1.0, Inf]")]] == truncated(Normal(1, 1.5), 1, Inf)

    end

    root = DiscreteRootNode(:x, Dict(:y => 0.2, :n => 0.8))
    discretization1 = ExactDiscretization([0, Inf])
    root1 = ContinuousRootNode(:z1, Normal(), discretization1)
    discretization2 = ApproximatedDiscretization([-Inf, -1, 0, 1, Inf], 1.5)
    states = Dict(
        [:y] => Normal(),
        [:n] => Normal(2, 2)
    )
    child = ContinuousChildNode(:β, [root], states, discretization2)

    nodes = [root, root1, child]

    continuous_root, discrete_root = @suppress EnhancedBayesianNetworks._discretize(root1)
    continuous_child, discrete_child = EnhancedBayesianNetworks._discretize(child)

    @test @suppress issetequal(EnhancedBayesianNetworks._discretize!(nodes), [root, continuous_root, discrete_root, continuous_child, discrete_child])
end