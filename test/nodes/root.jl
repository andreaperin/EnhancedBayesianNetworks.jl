@testset "Root Nodes" begin
    @testset "ContinuesRootNode" begin
        rv = RandomVariable(Normal(), :x)
        node = ContinuousRootNode(rv)
        @test EnhancedBayesianNetworks._get_states(node) == Normal()
    end

    @testset "DiscreteRootNode" begin
        name = :x

        states = Dict(:yes => -0.5, :no => 0.5)
        @test_throws ErrorException("Probabilites must be nonnegative") DiscreteRootNode(name, states)

        states = Dict(:yes => 0.5, :no => 1.5)
        @test_throws ErrorException("Probabilites must be less or equal to 1.0") DiscreteRootNode(name, states)

        states = Dict(:yes => 0.5, :no => 0.7)
        @test_throws ErrorException("Probabilites must sum up to 1.0") DiscreteRootNode(name, states)

        states = Dict(:yes => 0.2, :no => 0.8)
        node = DiscreteRootNode(name, states)
        @test EnhancedBayesianNetworks._get_states(node) == [:yes, :no]
    end
end