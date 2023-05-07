@testset "Functional Nodes" begin
    @testset "ContinuousFunctionalNode" begin end

    @testset "DiscreteFunctionalNode" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.2, :no => 0.8), Dict(:yes => [Parameter(2.2, :x)], :no => [Parameter(5.5, :x)]))
        root2 = DiscreteRootNode(:y, Dict(:y => 0.4, :n => 0.6), Dict(:y => [Parameter(2.2, :y)], :n => [Parameter(5.5, :y)]))
        root3 = ContinuousRootNode(RandomVariable(Normal(), :z))
        root4 = DiscreteRootNode(:x, Dict(:yes => 0.2, :no => 0.8))

        name = :child
        parents = [root1, root2, root3]
        model = Model(df -> sqrt.(df.z .^ 2 + df.z .^ 2), :value1)
        performance = Model(df -> 1 .- 2 .* df.value1, :value2)
        models = OrderedDict(
            [:yes] => [model, performance],
            [:yes] => [model, performance],
            [:no] => [model, performance],
            [:no] => [model, performance]
        )
        simulations = OrderedDict(
            [:yes, :n] => MonteCarlo(100),
            [:yes, :y] => MonteCarlo(200),
            [:no, :n] => MonteCarlo(300),
            [:no, :y] => MonteCarlo(400)
        )
        @test_throws ErrorException("defined parent nodes states must be equal to the number of discrete parent nodes") DiscreteFunctionalNode(name, parents, models, simulations)

        models = OrderedDict(
            [:yes, :n] => [model, performance],
            [:yes, :y] => [model, performance],
            [:no, :y] => [model, performance]
        )
        simulations = OrderedDict(
            [:yes, :n] => MonteCarlo(100),
            [:yes, :y] => MonteCarlo(200),
            [:no, :n] => MonteCarlo(300),
            [:no, :y] => MonteCarlo(400)
        )
        @test_throws ErrorException("defined combinations must be equal to the discrete parents combinations") DiscreteFunctionalNode(name, parents, models, simulations)

        models = OrderedDict(
            [:yes, :maybe] => [model, performance],
            [:yes, :y] => [model, performance],
            [:no, :y] => [model, performance],
            [:no, :n] => [model, performance]
        )
        simulations = OrderedDict(
            [:yes, :n] => MonteCarlo(100),
            [:yes, :y] => MonteCarlo(200),
            [:no, :n] => MonteCarlo(300),
            [:no, :y] => MonteCarlo(400)
        )
        @test_throws ErrorException("StandardNode state's keys must contain state from parent and the order of the parents states must be coherent with the order of the parents defined in node.parents") DiscreteFunctionalNode(name, parents, models, simulations)

        models = OrderedDict(
            [:yes, :n] => [model, performance],
            [:yes, :y] => [model, performance],
            [:no, :y] => [model, performance],
            [:no, :n] => [model, performance]
        )
        simulations = OrderedDict(
            [:yes, :n] => MonteCarlo(100),
            [:yes, :y] => MonteCarlo(200),
            [:no, :n] => MonteCarlo(300),
            [:no, :y] => MonteCarlo(400)
        )
        @test_throws ErrorException("all discrete parents of a functional node must have a non-empty parameters dictionary") DiscreteFunctionalNode(name, [root1, root3, root4], models, simulations)

        @test DiscreteFunctionalNode(name, [root1, root2, root3], models, simulations).simulations == simulations

        @test DiscreteFunctionalNode(name, [root1, root2, root3], models, simulations).models == models

        @test DiscreteFunctionalNode(name, [root1, root2, root3], models, simulations).name == :child

        @test Set(DiscreteFunctionalNode(name, [root1, root2, root3], models, simulations).parents) == Set([root1, root2, root3])
    end
end
