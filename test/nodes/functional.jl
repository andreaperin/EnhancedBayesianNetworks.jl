@testset "Functional Nodes" begin
    @testset "ContinuousFunctionalNode" begin end

    @testset "DiscreteFunctionalNode" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.2, :no => 0.8))
        root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6))
        root3 = ContinuousRootNode(RandomVariable(Normal(), :z))
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
            [:yes, :no] => [model, performance],
            [:yes, :yes] => [model, performance],
            [:no, :yes] => [model, performance]
        )
        @test_throws ErrorException("defined combinations must be equal to the discrete parents combinations") DiscreteFunctionalNode(name, parents, models)

        models = OrderedDict(
            [:yes, :maybe] => [model, performance],
            [:yes, :yes] => [model, performance],
            [:no, :yes] => [model, performance],
            [:no, :no] => [model, performance]
        )
        @test_throws ErrorException("missmatch in defined parents combinations states and states of the parents") DiscreteFunctionalNode(name, parents, models)

    end
end
## TODO add Test when a StdNode has a StdNode as parents (the same for FunctionalNodes)