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
        @test_throws ErrorException("defined parent nodes states must be equal to the number of discrete parent nodes") DiscreteFunctionalNode(name, parents, models)

        models = OrderedDict(
            [:yes, :n] => [model, performance],
            [:yes, :y] => [model, performance],
            [:no, :y] => [model, performance]
        )
        @test_throws ErrorException("defined combinations must be equal to the discrete parents combinations") DiscreteFunctionalNode(name, parents, models)

        models = OrderedDict(
            [:yes, :maybe] => [model, performance],
            [:yes, :y] => [model, performance],
            [:no, :y] => [model, performance],
            [:no, :n] => [model, performance]
        )
        @test_throws ErrorException("StandardNode state's keys must contain state from parent and the order of the parents states must be coherent with the order of the parents defined in node.parents") DiscreteFunctionalNode(name, parents, models)

        models = OrderedDict(
            [:yes, :n] => [model, performance],
            [:yes, :y] => [model, performance],
            [:no, :y] => [model, performance],
            [:no, :n] => [model, performance]
        )

        @test_throws ErrorException("all discrete parents of a functional node must have a non-empty parameters dictionary") DiscreteFunctionalNode(name, [root1, root3, root4], models)

        @test DiscreteFunctionalNode(name, [root1, root2, root3], models).models == models

        @test DiscreteFunctionalNode(name, [root1, root2, root3], models).name == :child

        @test Set(DiscreteFunctionalNode(name, [root1, root2, root3], models).parents) == Set([root1, root2, root3])
    end
end
## TODO add Test when a StdNode has a StdNode as parents (the same for FunctionalNodes)