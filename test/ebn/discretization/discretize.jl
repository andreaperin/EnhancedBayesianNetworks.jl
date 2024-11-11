@testset "Node Discretization" begin

    @testset "Format Intervals" begin
        discretization = ExactDiscretization([-10, 0, 10])
        root = ContinuousRootNode(:z1, truncated(Normal(0, 1); lower=-10, upper=10), discretization)

        formatted_intervals = EnhancedBayesianNetworks._format_interval(root)
        @test formatted_intervals == [[-10, 0.0], [0.0, 10]]

        discretization = ExactDiscretization([-9, 10])
        root = ContinuousRootNode(:z1, truncated(Normal(0, 1); lower=-10, upper=10), discretization)

        @test_logs (:warn, "node z1 has minimum intervals value -9 > support lower bound -10.0. Lower bound will be used as intervals start") EnhancedBayesianNetworks._format_interval(root)

        discretization = ExactDiscretization([-11, 10])
        root = ContinuousRootNode(:z1, truncated(Normal(0, 1); lower=-10, upper=10), discretization)

        @test_logs (:warn, "node z1 has minimum intervals value -11 < support lower bound -10.0. Lower bound will be used as intervals start") EnhancedBayesianNetworks._format_interval(root)

        discretization = ExactDiscretization([-10, 9])
        root = ContinuousRootNode(:z1, truncated(Normal(0, 1); lower=-10, upper=10), discretization)

        @test_logs (:warn, "node z1 has maximum intervals value 9 < support upper bound 10.0. Upper bound will be used as intervals end") EnhancedBayesianNetworks._format_interval(root)

        discretization = ExactDiscretization([-10, 11])
        root = ContinuousRootNode(:z1, truncated(Normal(0, 1); lower=-10, upper=10), discretization)

        @test_logs (:warn, "node z1 has maximum intervals value 11 > support upper bound 10.0. Upper bound will be used as intervals end") EnhancedBayesianNetworks._format_interval(root)

        intervals = [[-Inf, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, Inf]]
        λ = 2

        exp1 = -Exponential(2) - 1
        exp2 = Exponential(2) + 1
        approx = [
            exp1,
            Uniform(-1, 0),
            Uniform(0.0, 1.0),
            exp2
        ]

        @test approx == EnhancedBayesianNetworks._approximate(intervals, λ)

        dist = Normal()
        probs = [0.15865525393145702, 0.341344746068543, 0.34134474606854304, 0.15865525393145696]

        @test all(isapprox.(EnhancedBayesianNetworks._discretize(dist, intervals), probs))
    end

    @testset "Root node" begin
        discretization = ExactDiscretization([0, Inf])
        root = ContinuousRootNode(:z1, Normal(), discretization)

        disc_root, cont_root = @suppress EnhancedBayesianNetworks._discretize(root)

        @test disc_root.name == :z1_d
        @test disc_root.states[Symbol("[-Inf, 0.0]")] == 0.5
        @test disc_root.states[Symbol("[0.0, Inf]")] == 0.5

        @test cont_root.name == :z1
        @test cont_root.distribution[[Symbol("[-Inf, 0.0]")]] == truncated(Normal(), -Inf, 0.0)
        @test cont_root.distribution[[Symbol("[0.0, Inf]")]] == truncated(Normal(), 0.0, Inf)

        discretization = ExactDiscretization([-Inf, 0, Inf])

        disc_root, cont_root = @suppress EnhancedBayesianNetworks._discretize(root)

        @test disc_root.name == :z1_d
        @test disc_root.states[Symbol("[-Inf, 0.0]")] == 0.5
        @test disc_root.states[Symbol("[0.0, Inf]")] == 0.5

        @test cont_root.distribution[[Symbol("[-Inf, 0.0]")]] == truncated(Normal(), -Inf, 0.0)
        @test cont_root.distribution[[Symbol("[0.0, Inf]")]] == truncated(Normal(), 0.0, Inf)

        discretization = ExactDiscretization([-1, 0, 1])
        root = ContinuousRootNode(:z1, UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)]), discretization)

        disc_root, cont_root = @suppress EnhancedBayesianNetworks._discretize(root)

        @test isapprox(disc_root.states[Symbol("[-Inf, -1.0]")], [0.0668072, 0.401294]; atol=0.001)
        @test isapprox(disc_root.states[Symbol("[-1.0, 0.0]")], [0.24173, 0.290169]; atol=0.001)
        @test isapprox(disc_root.states[Symbol("[0.0, 1.0]")], [0.24173, 0.290169]; atol=0.001)
        @test isapprox(disc_root.states[Symbol("[1.0, Inf]")], [0.0668072, 0.401294]; atol=0.001)

        dict = Dict(
            [Symbol("[-1.0, 0.0]")] => UnamedProbabilityBox{Normal}(Interval[Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)], -1.0, 0.0),
            [Symbol("[-Inf, -1.0]")] => UnamedProbabilityBox{Normal}(Interval[Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)], -Inf, -1.0),
            [Symbol("[1.0, Inf]")] => UnamedProbabilityBox{Normal}(Interval[Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)], 1.0, Inf),
            [Symbol("[0.0, 1.0]")] => UnamedProbabilityBox{Normal}(Interval[Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)], 0.0, 1.0)
        )

        @test cont_root.distribution == dict

        discretization = ExactDiscretization([-1, 0, 1])
        root = ContinuousRootNode(:z1, (-1, 1), discretization)

        disc_root, cont_root = @suppress EnhancedBayesianNetworks._discretize(root)

        @test isapprox(disc_root.states[Symbol("[-1.0, 0.0]")], [0, 1]; atol=0.001)
        @test isapprox(disc_root.states[Symbol("[0.0, 1.0]")], [0, 1]; atol=0.001)

        dict = Dict(
            [Symbol("[-1.0, 0.0]")] => (-1, 0),
            [Symbol("[0.0, 1.0]")] => (0, 1)
        )
        @test cont_root.distribution == dict
    end

    @testset "Child node" begin
        λ = 1.5
        discretization = ApproximatedDiscretization([-Inf, -1, 0, 1, Inf], λ)

        root = DiscreteRootNode(:x, Dict(:y => 0.2, :n => 0.8))

        states = Dict(
            [:y] => Normal(),
            [:n] => Normal(2, 2)
        )

        child = ContinuousChildNode(:β, states, discretization)

        disc_child, cont_child = EnhancedBayesianNetworks._discretize(child)

        exp1 = -Exponential(λ) - 1
        exp2 = Exponential(λ) + 1
        @test cont_child.distribution[[Symbol("[-Inf, -1.0]")]] == exp1
        @test cont_child.distribution[[Symbol("[-1.0, 0.0]")]] == Uniform(-1, 0)
        @test cont_child.distribution[[Symbol("[0.0, 1.0]")]] == Uniform(0, 1.0)
        @test cont_child.distribution[[Symbol("[1.0, Inf]")]] == exp2

        @test isapprox(disc_child.states[[:y]][Symbol("[-Inf, -1.0]")], 0.158655; atol=0.01)
        @test isapprox(disc_child.states[[:y]][Symbol("[0.0, 1.0]")], 0.341345; atol=0.01)
        @test isapprox(disc_child.states[[:y]][Symbol("[1.0, Inf]")], 0.158655; atol=0.01)
        @test isapprox(disc_child.states[[:y]][Symbol("[-1.0, 0.0]")], 0.341345; atol=0.01)

        @test isapprox(disc_child.states[[:n]][Symbol("[-Inf, -1.0]")], 0.0668072; atol=0.01)
        @test isapprox(disc_child.states[[:n]][Symbol("[0.0, 1.0]")], 0.149882; atol=0.01)
        @test isapprox(disc_child.states[[:n]][Symbol("[1.0, Inf]")], 0.691462; atol=0.01)
        @test isapprox(disc_child.states[[:n]][Symbol("[-1.0, 0.0]")], 0.0918481; atol=0.01)

        states = Dict(
            [:y] => UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)]),
            [:n] => UnamedProbabilityBox{Normal}([Interval(1, 2, :μ), Interval(1, 2, :σ)])
        )

        child = ContinuousChildNode(:β, states, discretization)

        disc_child, cont_child = EnhancedBayesianNetworks._discretize(child)

        @test cont_child.distribution[[Symbol("[-Inf, -1.0]")]] == exp1
        @test cont_child.distribution[[Symbol("[-1.0, 0.0]")]] == Uniform(-1, 0)
        @test cont_child.distribution[[Symbol("[0.0, 1.0]")]] == Uniform(0, 1.0)
        @test cont_child.distribution[[Symbol("[1.0, Inf]")]] == exp2

        @test isapprox(disc_child.states[[:y]][Symbol("[-Inf, -1.0]")], [0.0668072, 0.401294]; atol=0.01)
        @test isapprox(disc_child.states[[:y]][Symbol("[0.0, 1.0]")], [0.24173, 0.290169]; atol=0.01)
        @test isapprox(disc_child.states[[:y]][Symbol("[1.0, Inf]")], [0.0668072, 0.401294]; atol=0.01)
        @test isapprox(disc_child.states[[:y]][Symbol("[-1.0, 0.0]")], [0.24173, 0.290169]; atol=0.01)

        @test isapprox(disc_child.states[[:n]][Symbol("[-Inf, -1.0]")], [0.0013499, 0.158655]; atol=0.01)
        @test isapprox(disc_child.states[[:n]][Symbol("[0.0, 1.0]")], [0.135905, 0.191462]; atol=0.01)
        @test isapprox(disc_child.states[[:n]][Symbol("[1.0, Inf]")], [0.5, 0.841345]; atol=0.01)
        @test isapprox(disc_child.states[[:n]][Symbol("[-1.0, 0.0]")], [0.0214002, 0.149882]; atol=0.01)

        states = Dict(
            [:y] => (-1, 0),
            [:n] => (0, 1)
        )
        child = ContinuousChildNode(:β, states, discretization)

        disc_child, cont_child = @suppress EnhancedBayesianNetworks._discretize(child)

        @test cont_child.distribution[[Symbol("[-1.0, 0.0]")]] == Uniform(-1, 0)
        @test cont_child.distribution[[Symbol("[0.0, 1.0]")]] == Uniform(0, 1)

        @test disc_child.states[[:y]][Symbol("[0.0, 1.0]")] == [0, 1]
        @test disc_child.states[[:y]][Symbol("[-1.0, 0.0]")] == [0, 1]
        @test disc_child.states[[:n]][Symbol("[0.0, 1.0]")] == [0, 1]
        @test disc_child.states[[:n]][Symbol("[-1.0, 0.0]")] == [0, 1]
    end

    @testset "Network" begin

        root1 = DiscreteRootNode(:x, Dict(:y => 0.2, :n => 0.8), Dict(:y => [Parameter(1, :x)], :n => [Parameter(0, :x), Parameter(5.6, :x1)]))
        root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6), Dict(:yes => [Parameter(2.2, :y)], :no => [Parameter(5.5, :y)]))
        discretization_root3 = ExactDiscretization([-Inf, 0, Inf])
        root3 = ContinuousRootNode(:z1, Normal(), discretization_root3)

        standard1_name = :α
        standard1_states = Dict(
            [:y, :yes] => Dict(:a => 0.2, :b => 0.8),
            [:n, :yes] => Dict(:a => 0.3, :b => 0.7),
            [:y, :no] => Dict(:a => 0.4, :b => 0.6),
            [:n, :no] => Dict(:a => 0.5, :b => 0.5)
        )
        standard1_parameters = Dict(:a => [Parameter(3, :α)], :b => [Parameter(10, :α)])
        standard1_node = DiscreteChildNode(standard1_name, standard1_states, standard1_parameters)

        standard2_name = :β
        standard2_states = Dict(
            [:y] => Normal(),
            [:n] => Normal(2, 2)
        )
        standard2_states = Dict(
            [:y] => Normal(),
            [:n] => Normal(2, 2)
        )
        discretization_standard2 = ApproximatedDiscretization([-Inf, 0.1, Inf], 1.5)
        standard2_node = ContinuousChildNode(standard2_name, standard2_states, discretization_standard2)

        functional2_name = :f2
        functional2_model = Model(df -> (df.α .^ 2 + df.z .^ 2) ./ 2, :value1)
        functional2_simulation = MonteCarlo(800)
        functional2_performance = df -> 1 .- 2 .* df.value1
        functional2_node = DiscreteFunctionalNode(functional2_name, [functional2_model], functional2_performance, functional2_simulation)

        nodes = [standard1_node, root1, root3, root2, standard2_node, functional2_node]
        net = EnhancedBayesianNetwork(nodes)

        add_child!(net, :x, :α)
        add_child!(net, :y, :α)
        add_child!(net, :x, :β)
        add_child!(net, :α, :f2)
        add_child!(net, :z1, :f2)
        order!(net)

        EnhancedBayesianNetworks._discretize!(net)
        d1, c1 = EnhancedBayesianNetworks._discretize(root3)
        d2, c2 = EnhancedBayesianNetworks._discretize(standard2_node)

        @test net.adj_matrix == sparse([
            0.0 0.0 0.0 1.0 0.0 1.0 0.0 0.0;
            0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
            0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
        ])
        @test net.topology_dict == Dict(:α => 4, :β_d => 6, :z1_d => 3, :y => 2, :z1 => 5, :f2 => 7, :β => 8, :x => 1)
        @test issetequal(net.nodes, [standard1_node, root1, d1, c1, root2, d2, c2, functional2_node])
    end
end

