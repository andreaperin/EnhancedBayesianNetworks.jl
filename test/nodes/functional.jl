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
        performance = df -> 1 .- 2 .* df.value1
        models = OrderedDict(
            [:yes] => [model],
            [:yes] => [model],
            [:no] => [model],
            [:no] => [model]
        )
        simulations = OrderedDict(
            [:yes, :n] => MonteCarlo(100),
            [:yes, :y] => MonteCarlo(200),
            [:no, :n] => MonteCarlo(300),
            [:no, :y] => MonteCarlo(400)
        )
        performances = OrderedDict(
            [:yes, :n] => df -> 1 .- 2 .* df.value1,
            [:yes, :y] => df -> 1 .- 2 .* df.value1,
            [:no, :n] => df -> 1 .- 2 .* df.value1,
            [:no, :y] => df -> 1 .- 2 .* df.value1
        )
        @test_throws ErrorException("defined parent nodes states must be equal to the number of discrete parent nodes") DiscreteFunctionalNode(name, parents, models, performances, simulations)

        models = OrderedDict(
            [:yes, :n] => [model],
            [:yes, :y] => [model],
            [:no, :y] => [model]
        )
        @test_throws ErrorException("defined combinations must be equal to the discrete parents combinations") DiscreteFunctionalNode(name, parents, models, performances, simulations)

        models = OrderedDict(
            [:yes, :maybe] => [model],
            [:yes, :y] => [model],
            [:no, :y] => [model],
            [:no, :n] => [model]
        )
        @test_throws ErrorException("StandardNode state's keys must contain state from parent and the order of the parents states must be coherent with the order of the parents defined in node.parents") DiscreteFunctionalNode(name, parents, models, performances, simulations)

        models = OrderedDict(
            [:yes, :n] => [model],
            [:yes, :y] => [model],
            [:no, :y] => [model],
            [:no, :n] => [model]
        )
        @test_throws ErrorException("all discrete parents of a functional node must have a non-empty parameters dictionary") DiscreteFunctionalNode(name, [root1, root3, root4], models, performances, simulations)

        @test DiscreteFunctionalNode(name, [root1, root2, root3], models, performances, simulations).models == models

        @test DiscreteFunctionalNode(name, [root1, root2, root3], models, performances, simulations).performances == performances

        @test DiscreteFunctionalNode(name, [root1, root2, root3], models, performances, simulations).simulations == simulations

        @test DiscreteFunctionalNode(name, [root1, root2, root3], models, performances, simulations).name == :child

        @test Set(DiscreteFunctionalNode(name, [root1, root2, root3], models, performances, simulations).parents) == Set([root1, root2, root3])
    end
end
