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

        @test get_continuous_input(node1) == RandomVariable(node1.distribution, node1.name)
        @test isequal(node1, ContinuousRootNode(:x1, Normal()))
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

        @test isequal(node1, DiscreteRootNode(name, states, parameters))
    end
end