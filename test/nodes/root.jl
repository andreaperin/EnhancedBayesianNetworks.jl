@testset "Root Nodes" begin
    @testset "ContinuesRootNode" begin
        root1 = DiscreteRootNode(:d, Dict(:yes => 0.5, :no => 0.5))
        evidence = [(:yes, root1)]

        name = :x
        distribution = Normal()

        @test ContinuousRootNode(name, distribution).name == name

        @test ContinuousRootNode(name, distribution).distribution == distribution

        @test get_randomvariable(ContinuousRootNode(name, distribution), evidence) == RandomVariable(distribution, name)

        node = ContinuousRootNode(name, distribution)

        @test node.name == name
        @test node.distribution == distribution
        @test node.intervals == Vector{Vector{Float64}}()
    end

    @testset "DiscreteRootNode" begin
        root1 = DiscreteRootNode(:d, Dict(:yes => 0.5, :no => 0.5))
        evidence = [(:yes, root1)]

        name = :x
        states = Dict(:yes => -0.5, :no => 0.5)
        parameters = Dict(:yes => [Parameter(2, :d)], :no => [Parameter(0, :d)])

        @test_throws ErrorException("Probabilites must be nonnegative") DiscreteRootNode(name, states, parameters)

        states = Dict(:yes => 0.5, :no => 1.5)
        @test_throws ErrorException("Probabilites must be less or equal to 1.0") DiscreteRootNode(name, states, parameters)

        states = Dict(:yes => 0.5, :no => 0.7)
        @test_throws ErrorException("Probabilites must sum up to 1.0") DiscreteRootNode(name, states, parameters)

        states = Dict(:yes => 0.2, :no => 0.8)
        node = DiscreteRootNode(name, states, parameters)

        @test node.name == name
        @test node.parameters == parameters
        @test node.states == states

        @test EnhancedBayesianNetworks._get_states(node) == [:yes, :no]

        @test_throws ErrorException("evidence does not contain DiscreteRootNode") get_parameters(node, [(:yes, root1)])

        @test get_parameters(node, [(:yes, node)]) == [Parameter(2, :d)]

    end
end