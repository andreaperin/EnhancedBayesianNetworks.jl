@testset "Functional Nodes" begin
    @testset "ContinuousFunctionalNode" begin

        root1 = DiscreteRootNode(:x, Dict(:yes => 0.2, :no => 0.8), Dict(:yes => [Parameter(2.2, :x)], :no => [Parameter(5.5, :x)]))
        root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :n => 0.6), Dict(:yes => [Parameter(2.2, :y)], :n => [Parameter(5.5, :y)]))
        root3 = ContinuousRootNode(:z, Normal())
        root4 = DiscreteRootNode(:k, Dict(:a => 0.2, :b => 0.8))

        name = :child
        parents = [root1, root2, root3]
        model = Model(df -> sqrt.(df.z .^ 2 + df.z .^ 2), :value1)
        models = Dict(
            [:yes, :n] => [model],
            [:yes, :y] => [model],
            [:no, :n] => [model],
        )

        @test_throws ErrorException("all discrete parents of a functional node must have different named states") ContinuousFunctionalNode(name, parents, models)

        root2 = DiscreteRootNode(:y, Dict(:y => 0.4, :n => 0.6))

        @test_throws ErrorException("all discrete parents of a functional node must have a non-empty parameters dictionary") ContinuousFunctionalNode(name, [root1, root2, root3], models)

        root2 = DiscreteRootNode(:y, Dict(:y => 0.4, :n => 0.6), Dict(:y => [Parameter(2.2, :y)], :n => [Parameter(5.5, :y)]))

        @test_throws ErrorException("In node child, defined combinations are not equal to the theorical discrete parents combinations: [[:yes, :n] [:yes, :y]; [:no, :n] [:no, :y]]") ContinuousFunctionalNode(name, [root1, root2, root3], models)

        models = Dict(
            [:yes, :n] => [model],
            [:yes, :y] => [model],
            [:no, :n] => [model],
            [:no, :yep] => [model]
        )

        @test_throws ErrorException("In node child, defined parents states are not coherent with its discrete parents states") ContinuousFunctionalNode(name, [root1, root2, root3], models)

        models = Dict(
            [:yes, :n] => [model],
            [:yes, :y] => [model],
            [:no, :n] => [model],
        )
        @test_throws ErrorException("In node child, defined combinations are not equal to the theorical discrete parents combinations: [[:yes, :n] [:yes, :y]; [:no, :n] [:no, :y]]") ContinuousFunctionalNode(name, [root1, root2, root3], models)

        models = Dict(
            [:yes, :n] => [model],
            [:yes, :y] => [model],
            [:no, :n] => [model],
            [:no, :y] => [model]
        )
        node = ContinuousFunctionalNode(name, [root1, root2, root3], models)

        @test node.name == name
        @test issetequal(node.parents, [root1, root2, root3])
        @test node.models == models

        evidence = [:a]

        @test get_randomvariable(node, evidence) == RandomVariable(Normal(), :z)

        @test_throws ErrorException("evidence [:a] does not contain all the parents of the ContinuousChildNode child") get_models(node, evidence)

        evidence = [:yes, :n]
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
        models = Dict(
            [:yes] => [model],
            [:yes] => [model],
            [:no] => [model],
            [:no] => [model]
        )
        simulations = Dict(
            [:yes, :n] => MonteCarlo(100),
            [:yes, :y] => MonteCarlo(200),
            [:no, :n] => MonteCarlo(300),
            [:no, :y] => MonteCarlo(400)
        )
        performances = Dict(
            [:yes, :n] => df -> 1 .- 2 .* df.value1,
            [:yes, :y] => df -> 1 .- 2 .* df.value1,
            [:no, :n] => df -> 1 .- 2 .* df.value1,
            [:no, :y] => df -> 1 .- 2 .* df.value1
        )
        @test_throws ErrorException("In node child, defined parents states differ from number of its discrete parents") DiscreteFunctionalNode(name, parents, models, performances, simulations)

        models = Dict(
            [:yes, :n] => [model],
            [:yes, :y] => [model],
            [:no, :y] => [model]
        )
        @test_throws ErrorException("In node child, defined combinations are not equal to the theorical discrete parents combinations: [[:yes, :n], [:no, :n], [:yes, :y], [:no, :y]]") DiscreteFunctionalNode(name, parents, models, performances, simulations)

        models = Dict(
            [:yes, :maybe] => [model],
            [:yes, :y] => [model],
            [:no, :y] => [model],
            [:no, :n] => [model]
        )
        @test_throws ErrorException("In node child, defined parents states are not coherent with its discrete parents states") DiscreteFunctionalNode(name, parents, models, performances, simulations)

        models = Dict(
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
        @test issetequal(node.parents, [root1, root2, root3])

        evidence = [:yes]

        @test_throws ErrorException("evidence [:yes] does not contain all the parents of child") get_models(node, evidence)

        @test_throws ErrorException("evidence [:yes] does not contain all the parents of child") get_performance(node, evidence)

        @test_throws ErrorException("evidence [:yes] does not contain all the parents of child") get_simulation(node, evidence)

        evidence = [:yes, :n]

        @test get_models(node, evidence) == [model]

        @test get_performance(node, evidence) == node.performances[[:yes, :n]]

        @test get_simulation(node, evidence) == MonteCarlo(100)

    end
end
