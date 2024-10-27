@testset "Child Nodes" begin
    @testset "ContinuousChildNode" begin
        name = :A
        distribution = Dict(
            [:yes, :y] => Normal(),
            [:no, :y] => Normal(1, 1),
            [:yes, :n] => Normal(2, 1),
            [:no, :n] => Normal(3, 1)
        )
        node = ContinuousChildNode(name, distribution)

        @test node.name == name
        @test node.distribution == distribution
        @test isequal(node.discretization, ApproximatedDiscretization())
        @test node.additional_info == Dict{Vector{Symbol},Dict}()

        node = ContinuousChildNode(name, distribution, Dict{Vector{Symbol},Dict}())
        @test node.name == name
        @test node.distribution == distribution
        @test isequal(node.discretization, ApproximatedDiscretization())
        @test node.additional_info == Dict{Vector{Symbol},Dict}()

        node = ContinuousChildNode(name, Dict([:yes] => Normal(), [:no] => Normal(2, 2)))
        evidence = [:a]

        @test_throws ErrorException("evidence [:a] does not contain all the parents of the ContinuousChildNode A") EnhancedBayesianNetworks._get_continuous_input(node, evidence)

        evidence = [:yes]
        @test EnhancedBayesianNetworks._get_continuous_input(node, evidence) == RandomVariable(Normal(), node.name)
        @test EnhancedBayesianNetworks._get_node_distribution_bounds(node) == (-Inf, Inf)
        @test EnhancedBayesianNetworks._is_imprecise(node) == false

        @testset "Imprecise Child - Interval" begin
            name = :child
            states = Dict(
                [:yes] => (0.1, 0.3),
                [:no] => (0.6, 0.7)
            )
            child = ContinuousChildNode(name, states)
            @test child.name == name
            @test child.distribution == states
            @test isequal(child.discretization, ApproximatedDiscretization())
            @test child.additional_info == Dict{Vector{Symbol},Dict}()

            @test EnhancedBayesianNetworks._get_continuous_input(child, [:yes]) == Interval(0.1, 0.3, :child)
            @test EnhancedBayesianNetworks._get_node_distribution_bounds(child) == (0.1, 0.7)
            @test EnhancedBayesianNetworks._is_imprecise(child)
        end

        @testset "Imprecise Child - Interval" begin
            normal_pbox = UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)])
            uniform_pbox = UnamedProbabilityBox{Uniform}([Interval(1, 3, :a), Interval(4, 5, :b)])
            states = Dict(
                [:yes] => normal_pbox,
                [:no] => uniform_pbox
            )
            child = ContinuousChildNode(:child, states)

            yes_input = EnhancedBayesianNetworks._get_continuous_input(child, [:yes])
            @test typeof(yes_input) == ProbabilityBox{Normal}
            @test yes_input.lb == normal_pbox.lb
            @test yes_input.ub == normal_pbox.ub
            @test yes_input.name == :child
            @test yes_input.parameters == normal_pbox.parameters

            # ! TODO uncomment this test when issue 204 in UncertaintyQuantification.jl is closed
            # no_input = _get_continuous_input(child, [:no])
            # @test typeof(no_input) == ProbabilityBox{Uniform}
            # @test no_input.lb == normal_pbox.lb
            # @test no_input.ub == normal_pbox.ub
            # @test no_input.name == :child
            # @test no_input.parameters == normal_pbox.parameters
        end
    end

    @testset "DiscreteChildNode" begin
        name = :child
        states = Dict(
            [:yes] => Dict(:yes => [0.1, 0.2], :no => 0.9),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )

        @test_throws ErrorException("mixed interval and single value states probabilities are not allowed") DiscreteChildNode(name, states)

        states = Dict(
            [:yes] => Dict(:yes => 0.1, :no => 0.9),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )
        covs = Dict(
            [:yes] => "a",
            [:no] => 0.1
        )
        additional_info = Dict{Vector{Symbol},Dict}()
        states = Dict(
            [:yes] => Dict(:yes => -0.1, :no => 0.9),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )

        @test_throws ErrorException("probabilities must be non-negative") DiscreteChildNode(name, states)

        states = Dict(
            [:yes] => Dict(:yes => 1.1, :no => 0.9),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("probabilities must be lower or equal than 1") DiscreteChildNode(name, states)

        states = Dict(
            [:yes] => Dict(:yes => 0.3, :no => 0.9),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("states [:yes, :no] are exhaustives and mutually exclusive. Their probabilities [0.3, 0.9] does not sum up to 1") DiscreteChildNode(name, states)

        states = Dict(
            [:yes] => Dict(:yes => 0.4999, :no => 0.4999),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_logs (:warn, "total probaility should be one, but the evaluated value is 0.9998 , and will be normalized") DiscreteChildNode(name, states)

        states = Dict(
            [:yes, :yes] => Dict(:yep => 0.2, :no => 0.8),
            [:no, :yes] => Dict(:yes => 0.2, :no => 0.8),
            [:yes, :no] => Dict(:yes => 0.2, :no => 0.8),
            [:no, :no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("non coherent definition of states over scenarios: [([:yes, :no], [0.2, 0.8]), ([:yep, :no], [0.2, 0.8]), ([:yes, :no], [0.2, 0.8]), ([:yes, :no], [0.2, 0.8])]") DiscreteChildNode(name, states)

        states = Dict(
            [:yes, :yes] => Dict(:yes => 0.2, :no => 0.8),
            [:no, :yes] => Dict(:yes => 0.2, :no => 0.8),
            [:yes, :no] => Dict(:yes => 0.2, :no => 0.8)
        )
        node = DiscreteChildNode(name, states)
        @test node.name == name
        @test node.states == states
        @test node.parameters == Dict{Symbol,Vector{Parameter}}()
        @test node.additional_info == Dict{Vector{Symbol},Dict}()

        node = DiscreteChildNode(name, states, Dict{Vector{Symbol},Dict}())
        @test node.name == name
        @test node.states == states
        @test node.parameters == Dict{Symbol,Vector{Parameter}}()
        @test node.additional_info == Dict{Vector{Symbol},Dict}()

        @test EnhancedBayesianNetworks._get_states(node) == [:yes, :no]

        node = DiscreteChildNode(name, states, Dict(:yes => [Parameter(1.1, :g)], :no => [Parameter(1.2, :g)]))
        evidence = [:a]
        @test_throws ErrorException("evidence [:a] does not contain child") EnhancedBayesianNetworks._get_parameters(node, evidence)

        node_ = DiscreteChildNode(name, states)
        @test_throws ErrorException("node child has an empty parameters vector") EnhancedBayesianNetworks._get_parameters(node_, evidence)

        evidence = [:a, :yes]
        @test EnhancedBayesianNetworks._get_parameters(node, evidence) == [Parameter(1.1, :g)]
        @test EnhancedBayesianNetworks._is_imprecise(node) == false

        @testset "Imprecise Child - Interval" begin
            states = Dict(
                [:yes, :yes] => Dict(:a => [0.4, 0.5], :b => [0.5, 0.6]),
                [:no, :yes] => Dict(:a => [0.4, 0.5], :b => [0.5, 0.6]),
                [:yes, :no] => Dict(:a => [0.4, 0.5], :b => [0.5, 0.6]),
                [:no, :no] => Dict(:a => [0.4, 0.5], :b => [0.5, 0.6])
            )
            parameters = Dict(:a => [Parameter(1, :a1)], :b => [Parameter(2, :a1)])
            child = DiscreteChildNode(name, states, parameters)

            @test child.name == name
            @test child.states == states
            @test child.parameters == parameters
            @test node.additional_info == Dict{Vector{Symbol},Dict}()

            @test EnhancedBayesianNetworks._get_parameters(child, [:a]) == [Parameter(1, :a1)]
            @test EnhancedBayesianNetworks._get_states(child) == [:a, :b]
            @test EnhancedBayesianNetworks._is_imprecise(child)

            child = DiscreteChildNode(name, Dict(
                [:yes] => Dict(:y => [0.4, 0.5], :n => [0.5, 0.6]),
                [:no] => Dict(:y => [0.5, 0.6], :n => [0.4, 0.5])
            ))
            extreme_points = EnhancedBayesianNetworks._extreme_points(child)

            n1 = DiscreteChildNode(name, Dict(
                [:yes] => Dict(:y => 0.4, :n => 0.6),
                [:no] => Dict(:y => 0.5, :n => 0.5)
            ))

            n2 = DiscreteChildNode(name, Dict(
                [:yes] => Dict(:y => 0.5, :n => 0.5),
                [:no] => Dict(:y => 0.5, :n => 0.5)
            ))

            n3 = DiscreteChildNode(name, Dict(
                [:yes] => Dict(:y => 0.5, :n => 0.5),
                [:no] => Dict(:y => 0.6, :n => 0.4)
            ))

            n4 = DiscreteChildNode(name, Dict(
                [:yes] => Dict(:y => 0.4, :n => 0.6),
                [:no] => Dict(:y => 0.6, :n => 0.4)
            ))

            @test issetequal(extreme_points, [n1, n2, n3, n4])

            child = DiscreteChildNode(name, Dict(
                [:yes] => Dict(:y => [0.1, 0.3], :n => [0.5, 0.6], :m => [0.1, 0.3]),
                [:no] => Dict(:y => [0.5, 0.6], :n => [0.4, 0.5], :m => [0.1, 0.3])
            ))
            extreme_points = @suppress EnhancedBayesianNetworks._extreme_points(child)

            @test isapprox(collect(values(extreme_points[1].states[[:yes]])), [0.1, 0.6, 0.3])
            @test isapprox(collect(values(extreme_points[2].states[[:yes]])), [0.2, 0.5, 0.3])
            @test isapprox(collect(values(extreme_points[3].states[[:yes]])), [0.3, 0.6, 0.1])
            @test isapprox(collect(values(extreme_points[4].states[[:yes]])), [0.3, 0.5, 0.2])
        end
    end
end
