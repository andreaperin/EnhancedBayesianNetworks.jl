@testset "Root Nodes" begin
    @testset "ContinuesRootNode" begin

        node1 = ContinuousRootNode(:x1, Normal())
        node2 = ContinuousRootNode(:x2, Normal(), ExactDiscretization([-1, 0, 1]))

        @test node1.name == :x1
        @test node1.distribution == Normal()
        @test isequal(node1.discretization, ExactDiscretization())
        @test node2.name == :x2
        @test node2.distribution == Normal()
        @test isequal(node2.discretization, ExactDiscretization([-1, 0, 1]))

        @test get_randomvariable(node1) == RandomVariable(node1.distribution, node1.name)
        @test EnhancedBayesianNetworks._get_node_distribution_bounds(node1) == (-Inf, Inf)
        @test EnhancedBayesianNetworks._is_imprecise(node1) == false

        @testset "Imprecise Root - Interval" begin
            node1 = ContinuousRootNode(:x1, (0.1, 0.3))
            @test node1.name == :x1
            @test node1.distribution == (0.1, 0.3)
            @test isequal(node1.discretization, ExactDiscretization())
            @test get_randomvariable(node1) == Interval(0.1, 0.3, :x1)
            @test EnhancedBayesianNetworks._get_node_distribution_bounds(node1) == (0.1, 0.3)
            @test EnhancedBayesianNetworks._is_imprecise(node1)
        end
    end

    @testset "DiscreteRootNode" begin
        name = :x
        parameters = Dict(:yes => [Parameter(2, :d)], :no => [Parameter(0, :d)])

        states = Dict(:yes => -0.5, :no => 0.5)
        @test_throws ErrorException("Probabilites must be nonnegative") DiscreteRootNode(name, states, parameters)

        states = Dict(:yes => 0.5, :no => 1.5)
        @test_throws ErrorException("Probabilites must be less or equal to 1.0") DiscreteRootNode(name, states, parameters)

        states = Dict(:yes => 0.8, :no => 0.8)
        @test_throws ErrorException("defined states probabilities Dict(:yes => 0.8, :no => 0.8) are wrong") DiscreteRootNode(name, states, parameters)

        states = Dict(:yes => 0.4999, :no => 0.4999)
        @test_logs (:warn, "total probaility should be one, but the evaluated value is 0.9998 , and will be normalized") DiscreteRootNode(name, states, parameters)

        states = Dict(:yes => 0.2, :no => 0.8)
        node1 = DiscreteRootNode(name, states, parameters)
        node2 = DiscreteRootNode(name, states)

        @test node1.name == name
        @test node1.parameters == parameters
        @test node1.states == states

        @test EnhancedBayesianNetworks._get_states(node1) == [:yes, :no]

        @test_throws ErrorException("evidence [:y] does not contain x") get_parameters(node1, [:y])
        @test_throws ErrorException("node x has an empty parameters vector") get_parameters(node2, [:y, :yes])

        @test get_parameters(node1, [:yes]) == [Parameter(2, :d)]

        @testset "Imprecise Root - Interval" begin
            name = :x
            parameters = Dict(:yes => [Parameter(2, :d)], :no => [Parameter(0, :d)])

            states = Dict(:yes => [0.1, 0.15], :no => 0.8)
            @test_throws ErrorException("node x has mixed interval and single value states probabilities!") DiscreteRootNode(name, states, parameters)

            states = Dict(:yes => [-0.5, 0.6], :no => [0.7, 0.9])
            @test_throws ErrorException("Probabilites must be nonnegative") DiscreteRootNode(name, states, parameters)

            states = Dict(:yes => [1.5, 0.6], :no => [0.7, 0.9])
            @test_throws ErrorException("Probabilites must be less or equal to 1.0") DiscreteRootNode(name, states, parameters)

            states = Dict(:yes => [0.5, 0.6], :no => [0.7, 0.9])
            @test_throws ErrorException("sum of intervals lower bounds is bigger than 1 in Dict{Symbol, AbstractVector{Real}}(:yes => [0.5, 0.6], :no => [0.7, 0.9])") DiscreteRootNode(name, states, parameters)

            states = Dict(:yes => [0.1, 0.15], :no => [0.7, 0.8])
            @test_throws ErrorException("sum of intervals upper bounds is smaller than 1 in Dict{Symbol, AbstractVector{Real}}(:yes => [0.1, 0.15], :no => [0.7, 0.8])") DiscreteRootNode(name, states, parameters)

            states = Dict(:yes => [0.1, 0.15], :no => "a")
            @test_throws ErrorException("node x must have real valued states probabilities") DiscreteRootNode(name, states, parameters)
        end
    end
end