@testset "Functional Nodes" begin
    @testset "ContinuousFunctionalNode" begin

        root1 = DiscreteRootNode(:x, Dict(:yes => 0.2, :no => 0.8), Dict(:yes => [Parameter(2.2, :x)], :no => [Parameter(5.5, :x)]))
        root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :n => 0.6), Dict(:yes => [Parameter(2.2, :y)], :n => [Parameter(5.5, :y)]))
        root3 = ContinuousRootNode(:z, Normal())
        root4 = DiscreteRootNode(:k, Dict(:a => 0.2, :b => 0.8))

        name = :child
        parents = [root1, root2, root3]
        model = Model(df -> sqrt.(df.z .^ 2 + df.z .^ 2), :value1)
        models = [model]
        simulation = MonteCarlo(200)

        @test_throws ErrorException("all discrete parents of a functional node must have different named states") ContinuousFunctionalNode(name, parents, models, simulation)

        root2 = DiscreteRootNode(:y, Dict(:y => 0.4, :n => 0.6))

        @test_throws ErrorException("all discrete parents of a functional node must have a non-empty parameters dictionary") ContinuousFunctionalNode(name, [root1, root2, root3], models, simulation)

        root2 = DiscreteRootNode(:y, Dict(:y => 0.4, :n => 0.6), Dict(:y => [Parameter(2.2, :y)], :n => [Parameter(5.5, :y)]))

        node = ContinuousFunctionalNode(name, [root1, root2, root3], models, simulation)

        @test node.name == name
        @test issetequal(node.parents, [root1, root2, root3])
        @test node.models == models
        @test node.simulation == simulation
        @test isequal(node.discretization, ApproximatedDiscretization())
    end

    @testset "DiscreteFunctionalNode" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.2, :no => 0.8), Dict(:yes => [Parameter(2.2, :x)], :no => [Parameter(5.5, :x)]))
        root2 = DiscreteRootNode(:y, Dict(:y => 0.4, :n => 0.6), Dict(:y => [Parameter(2.2, :y)], :n => [Parameter(5.5, :y)]))
        root3 = ContinuousRootNode(:z, Normal())
        root4 = DiscreteRootNode(:k, Dict(:yes => 0.2, :no => 0.8))

        name = :child
        parents = [root1, root2, root3]
        model = Model(df -> sqrt.(df.z .^ 2 + df.z .^ 2), :value1)
        performance = df -> 1 .- 2 .* df.value1
        models = [model]
        simulation = MonteCarlo(400)
        performances = df -> 1 .- 2 .* df.value1

        node = DiscreteFunctionalNode(name, [root1, root2, root3], models, performances, simulation)

        @test node.models == models
        @test node.performance == performances
        @test node.simulation == simulation
        @test node.name == :child
        @test issetequal(node.parents, [root1, root2, root3])
        @test node.parameters == Dict{Symbol,Vector{Parameter}}()
    end
end
