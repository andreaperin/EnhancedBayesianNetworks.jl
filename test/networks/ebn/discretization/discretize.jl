@testset "Node Discretization" begin

    @testset "Format Intervals" begin
        discretization = ExactDiscretization([-10, 0, 10])
        root = ContinuousNode{UnivariateDistribution}(:z1, DataFrame(:Π => truncated(Normal(0, 1); lower=-10, upper=10)), discretization)

        formatted_intervals = EnhancedBayesianNetworks._format_interval(root)
        @test formatted_intervals == [[-10, 0.0], [0.0, 10]]

        discretization = ExactDiscretization([-9, 10])
        root = ContinuousNode{UnivariateDistribution}(:z1, DataFrame(:Π => truncated(Normal(0, 1); lower=-10, upper=10)), discretization)

        @test_logs (:warn, "node z1 has minimum intervals value -9 > support lower bound -10.0. Lower bound will be used as intervals start") EnhancedBayesianNetworks._format_interval(root)

        discretization = ExactDiscretization([-11, 10])
        root = ContinuousNode{UnivariateDistribution}(:z1, DataFrame(:Π => truncated(Normal(0, 1); lower=-10, upper=10)), discretization)

        @test_logs (:warn, "node z1 has minimum intervals value -11 < support lower bound -10.0. Lower bound will be used as intervals start") EnhancedBayesianNetworks._format_interval(root)

        discretization = ExactDiscretization([-10, 9])
        root = ContinuousNode{UnivariateDistribution}(:z1, DataFrame(:Π => truncated(Normal(0, 1); lower=-10, upper=10)), discretization)

        @test_logs (:warn, "node z1 has maximum intervals value 9 < support upper bound 10.0. Upper bound will be used as intervals end") EnhancedBayesianNetworks._format_interval(root)

        discretization = ExactDiscretization([-10, 11])
        root = ContinuousNode{UnivariateDistribution}(:z1, DataFrame(:Π => truncated(Normal(0, 1); lower=-10, upper=10)), discretization)

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

        @test approx == EnhancedBayesianNetworks._approximate.(intervals, λ)

        dist = Normal()
        probs = [0.15865525393145702, 0.341344746068543, 0.34134474606854304, 0.15865525393145696]

        @test all(isapprox.(EnhancedBayesianNetworks._discretize(dist, intervals), probs))
    end

    @testset "Root node" begin
        discretization = ExactDiscretization([0, Inf])
        root = ContinuousNode{UnivariateDistribution}(:z1, DataFrame(:Π => Normal()), discretization)

        disc_root, cont_root = @suppress EnhancedBayesianNetworks._discretize(root)

        @test disc_root.name == :z1_d
        @test disc_root.cpt[!, :z1_d] == [Symbol("[-Inf, 0.0]"), Symbol("[0.0, Inf]")]
        @test disc_root.cpt[!, :Π] == [0.5, 0.5]

        @test cont_root.name == :z1
        @test cont_root.cpt[!, :z1_d] == [Symbol("[-Inf, 0.0]"), Symbol("[0.0, Inf]")]
        @test cont_root.cpt[!, :Π] == [truncated(Normal(), -Inf, 0.0), truncated(Normal(), 0.0, Inf)]

        discretization = ExactDiscretization([-Inf, 0, Inf])

        disc_root, cont_root = @suppress EnhancedBayesianNetworks._discretize(root)

        @test disc_root.name == :z1_d
        @test disc_root.cpt[!, :z1_d] == [Symbol("[-Inf, 0.0]"), Symbol("[0.0, Inf]")]
        @test disc_root.cpt[!, :Π] == [0.5, 0.5]

        @test cont_root.name == :z1
        @test cont_root.cpt[!, :z1_d] == [Symbol("[-Inf, 0.0]"), Symbol("[0.0, Inf]")]
        @test cont_root.cpt[!, :Π] == [truncated(Normal(), -Inf, 0.0), truncated(Normal(), 0.0, Inf)]

        discretization = ExactDiscretization([-1, 0, 1])
        root = ContinuousNode{UnamedProbabilityBox}(:z1, DataFrame(:Π => [UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)])]), discretization)

        disc_root, cont_root = @suppress EnhancedBayesianNetworks._discretize(root)

        @test disc_root.cpt[!, :z1_d] == [Symbol("[-1.0, 0.0]"), Symbol("[-Inf, -1.0]"), Symbol("[0.0, 1.0]"), Symbol("[1.0, Inf]")]
        @test all(isapprox.(disc_root.cpt[!, :Π], [
                [0.24173033745712885, 0.2901687869569368],
                [0.06680720126885804, 0.4012936743170763],
                [0.2417303374571288, 0.2901687869569368],
                [0.06680720126885809, 0.4012936743170763]
            ]; atol=0.001))

        dists = [
            UnamedProbabilityBox{Normal}(Interval[Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)], -Inf, -1.0),
            UnamedProbabilityBox{Normal}(Interval[Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)], -1.0, 0.0), UnamedProbabilityBox{Normal}(Interval[Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)], 0.0, 1.0),
            UnamedProbabilityBox{Normal}(Interval[Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)], 1.0, Inf)
        ]

        @test cont_root.cpt[!, :z1_d] == [Symbol("[-1.0, 0.0]"), Symbol("[-Inf, -1.0]"), Symbol("[0.0, 1.0]"), Symbol("[1.0, Inf]")]

        cont_root.cpt[!, :Π] == dists

        discretization = ExactDiscretization([-1, 0, 1])
        root = ContinuousNode{Tuple{<:Real,<:Real}}(:z1, DataFrame(:Π => (-1, 1)), discretization)

        disc_root, cont_root = @suppress EnhancedBayesianNetworks._discretize(root)

        @test disc_root.cpt[!, :z1_d] == [Symbol("[-1.0, 0.0]"), Symbol("[0.0, 1.0]")]
        @test all(isapprox.(disc_root.cpt[!, :Π], [
                [0, 1],
                [0, 1],
            ]; atol=0.001))

        dists = [
            (-1.0, 0.0),
            (0.0, 1.0)
        ]
        @test cont_root.cpt[!, :z1_d] == [Symbol("[-1.0, 0.0]"), Symbol("[0.0, 1.0]")]
        @test cont_root.cpt[!, :Π] == dists
    end

    @testset "Child node" begin
        λ = 1.5
        discretization = ApproximatedDiscretization([-Inf, -1, 0, 1, Inf], λ)

        root = DiscreteNode(:x, DataFrame(:x => [:y, :n], :Π => [0.2, 0.8]))

        states = DataFrame(:x => [:y, :n], :Π => [Normal(), Normal(2, 2)])

        child = ContinuousNode{UnivariateDistribution}(:β, states, discretization)

        disc_child, cont_child = EnhancedBayesianNetworks._discretize(child)

        exp1 = -Exponential(λ) - 1
        exp2 = Exponential(λ) + 1
        @test cont_child.cpt[!, :β_d] == [
            Symbol("[-1.0, 0.0]"),
            Symbol("[-Inf, -1.0]"),
            Symbol("[0.0, 1.0]"),
            Symbol("[1.0, Inf]")
        ]
        @test cont_child.cpt[!, :Π] == [Uniform(-1, 0), exp1, Uniform(0, 1.0), exp2]

        @test disc_child.cpt[!, :x] == [:n, :n, :n, :n, :y, :y, :y, :y]
        @test disc_child.cpt[!, :β_d] == [
            Symbol("[-1.0, 0.0]"),
            Symbol("[-Inf, -1.0]"),
            Symbol("[0.0, 1.0]"),
            Symbol("[1.0, Inf]"),
            Symbol("[-1.0, 0.0]"),
            Symbol("[-Inf, -1.0]"),
            Symbol("[0.0, 1.0]"),
            Symbol("[1.0, Inf]")
        ]

        @test all(isapprox.(disc_child.cpt[!, :Π], [
                0.09184805266259898,
                0.06680720126885804,
                0.14988228479452986,
                0.6914624612740131,
                0.341344746068543,
                0.15865525393145702,
                0.34134474606854304,
                0.15865525393145696,
            ]; atol=0.01))

        states = DataFrame(:β => [:y, :n], :Π => [UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)]), UnamedProbabilityBox{Normal}([Interval(1, 2, :μ), Interval(1, 2, :σ)])])

        child = ContinuousNode{UnamedProbabilityBox}(:β, states, discretization)

        disc_child, cont_child = EnhancedBayesianNetworks._discretize(child)

        @test cont_child.cpt[!, :β_d] == [
            Symbol("[-1.0, 0.0]"),
            Symbol("[-Inf, -1.0]"),
            Symbol("[0.0, 1.0]"),
            Symbol("[1.0, Inf]")
        ]
        @test cont_child.cpt[!, :Π] == [Uniform(-1, 0), exp1, Uniform(0, 1), exp2]

        @test disc_child.cpt[!, :β] == [:n, :n, :n, :n, :y, :y, :y, :y]
        @test disc_child.cpt[!, :β_d] == [
            Symbol("[-1.0, 0.0]"),
            Symbol("[-Inf, -1.0]"),
            Symbol("[0.0, 1.0]"),
            Symbol("[1.0, Inf]"),
            Symbol("[-1.0, 0.0]"),
            Symbol("[-Inf, -1.0]"),
            Symbol("[0.0, 1.0]"),
            Symbol("[1.0, Inf]")
        ]
        @test isapprox(disc_child.cpt[!, :Π], [
                [0.021400233916549112, 0.14988228479452986],
                [0.001349898031630093, 0.15865525393145702],
                [0.1359051219832778, 0.19146246127401312],
                [0.5, 0.841344746068543],
                [0.24173033745712885, 0.2901687869569368],
                [0.06680720126885804, 0.4012936743170763],
                [0.2417303374571288, 0.2901687869569368],
                [0.06680720126885809, 0.4012936743170763]
            ]; atol=0.01)

        states = DataFrame(:β => [:y, :n], :Π => [(-1, 0), (0, 1)])

        child = ContinuousNode{Tuple{<:Real,<:Real}}(:β, states, discretization)

        disc_child, cont_child = @suppress EnhancedBayesianNetworks._discretize(child)

        @test cont_child.cpt[!, :β_d] == [Symbol("[-1.0, 0.0]"), Symbol("[0.0, 1.0]")]
        @test cont_child.cpt[!, :Π] == [Uniform(-1, 0), Uniform(0, 1)]

        @test disc_child.cpt[!, :β] == [:n, :n, :y, :y]
        @test disc_child.cpt[!, :β_d] == [
            Symbol("[-1.0, 0.0]"),
            Symbol("[0.0, 1.0]"),
            Symbol("[-1.0, 0.0]"),
            Symbol("[0.0, 1.0]"),
        ]
        @test disc_child.cpt[!, :Π] == [[0, 1], [0, 1], [0, 1], [0, 1]]
    end

    @testset "Network" begin

        root1 = DiscreteNode(:x, DataFrame(:x => [:y, :n], :Π => [0.2, 0.8]), Dict(:y => [Parameter(1, :x)], :n => [Parameter(0, :x), Parameter(5.6, :x1)]))
        root2 = DiscreteNode(:y, DataFrame(:y => [:yes, :no], :Π => [0.4, 0.6]), Dict(:yes => [Parameter(2.2, :y)], :no => [Parameter(5.5, :y)]))
        discretization_root3 = ExactDiscretization([-Inf, 0, Inf])
        root3 = ContinuousNode{UnivariateDistribution}(:z1, DataFrame(:Π => Normal()), discretization_root3)

        standard1_name = :α
        standard1_states = DataFrame(:x => [:y, :y, :y, :y, :n, :n, :n, :n], :y => [:yes, :yes, :no, :no, :yes, :yes, :no, :no], :α => [:a, :b, :a, :b, :a, :b, :a, :b], :Π => [0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5]
        )
        standard1_parameters = Dict(:a => [Parameter(3, :α)], :b => [Parameter(10, :α)])
        standard1_node = DiscreteNode(standard1_name, standard1_states, standard1_parameters)

        standard2_name = :β
        standard2_states = DataFrame(:x => [:y, :n], :Π => [Normal(), Normal(2, 2)])

        discretization_standard2 = ApproximatedDiscretization([-Inf, 0.1, Inf], 1.5)
        standard2_node = ContinuousNode{UnivariateDistribution}(standard2_name, standard2_states, discretization_standard2)

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