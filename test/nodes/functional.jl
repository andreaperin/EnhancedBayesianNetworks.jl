@testset "Functional Nodes" begin
    @testset "ContinuousFunctionalNode" begin

        root1 = DiscreteRootNode(:x, Dict(:yes => 0.2, :no => 0.8), Dict(:yes => [Parameter(2.2, :x)], :no => [Parameter(5.5, :x)]))
        root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :n => 0.6), Dict(:yes => [Parameter(2.2, :y)], :n => [Parameter(5.5, :y)]))
        root3 = ContinuousRootNode(:z, Normal())
        root4 = DiscreteRootNode(:k, Dict(:a => 0.2, :b => 0.8))

        name = :child
        parents = [root1, root2, root3]
        model = Model(df -> sqrt.(df.z .^ 2 + df.z .^ 2), :value1)
        models = OrderedDict(
            [:yes, :n] => [model],
            [:yes, :y] => [model],
            [:no, :n] => [model],
        )

        @test_throws ErrorException("all discrete parents of a functional node must have different named states") ContinuousFunctionalNode(name, parents, models)

        root2 = DiscreteRootNode(:y, Dict(:y => 0.4, :n => 0.6))

        @test_throws ErrorException("all discrete parents of a functional node must have a non-empty parameters dictionary") ContinuousFunctionalNode(name, [root1, root2, root3], models)

        root2 = DiscreteRootNode(:y, Dict(:y => 0.4, :n => 0.6), Dict(:y => [Parameter(2.2, :y)], :n => [Parameter(5.5, :y)]))

        @test_throws ErrorException("defined combinations must be equal to the discrete parents combinations") ContinuousFunctionalNode(name, [root1, root2, root3], models)

        models = OrderedDict(
            [:yes, :n] => [model],
            [:yes, :y] => [model],
            [:no, :n] => [model],
            [:no, :yep] => [model]
        )

        @test_throws ErrorException("StandardNode state's keys must contain state from parent and the order of the parents states must be coherent with the order of the parents defined in node.parents") ContinuousFunctionalNode(name, [root1, root2, root3], models)

        models = OrderedDict(
            [:yes, :n] => [model],
            [:yes, :y] => [model],
            [:no, :n] => [model],
        )
        @test_throws ErrorException("defined combinations must be equal to the discrete parents combinations") ContinuousFunctionalNode(name, [root1, root2, root3], models)

        models = OrderedDict(
            [:yes, :n] => [model],
            [:yes, :y] => [model],
            [:no, :n] => [model],
            [:no, :y] => [model]
        )
        node = ContinuousFunctionalNode(name, [root1, root2, root3], models)

        @test node.name == name
        @test node.parents == [root1, root2, root3]
        @test node.models == models

        evidence = [(:a, root4)]

        @test_throws ErrorException("evidence does not contain any parents of the ContinuousFunctionalNode") get_randomvariable(node, evidence)

        @test_throws ErrorException("evidence does not contain any parents of the ContinuousFunctionalNode") get_models(node, evidence)

        evidence = [(:yes, root1), (:n, root2)]
        @test get_randomvariable(node, evidence) == RandomVariable(Normal{Float64}(0.0, 1.0), :z)
        @test get_models(node, evidence) == [model]
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

        node = DiscreteFunctionalNode(name, [root1, root2, root3], models, performances, simulations)

        @test node.models == models
        @test node.performances == performances
        @test node.simulations == simulations
        @test node.name == :child
        @test Set(node.parents) == Set([root1, root2, root3])

        evidence = [(:yes, root4)]

        @test_throws ErrorException("evidence does not contain any parents of the FunctionalNode") get_models(node, evidence)

        @test_throws ErrorException("evidence does not contain any parents of the FunctionalNode") get_performance(node, evidence)

        @test_throws ErrorException("evidence does not contain any parents of the FunctionalNode") get_simulation(node, evidence)

        evidence = [(:yes, root1), (:n, root2)]

        @test get_models(node, evidence) == [model]

        @test get_performance(node, evidence) == node.performances[[:yes, :n]]

        @test get_simulation(node, evidence) == MonteCarlo(100)

    end
end
