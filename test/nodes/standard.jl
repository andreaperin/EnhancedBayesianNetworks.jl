@testset "Standard Nodes" begin
    @testset "ContinuousStandardNode" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
        root2 = DiscreteRootNode(:y, Dict(:y => 0.4, :n => 0.6))
        root3 = ContinuousRootNode(:z, Normal())
        root4 = DiscreteRootNode(:k, Dict(:a => 0.5, :b => 0.5))
        name = :child

        distribution = OrderedDict(
            [:yes, :yes] => Normal(),
            [:no] => Normal(1, 1)
        )
        @test_throws ErrorException("Number of symbols per parent in node.states must be equal to the number of discrete parents") ContinuousStandardNode(name, [root1], distribution)

        distribution = OrderedDict(
            [:yes] => Normal(),
            [:no] => Normal(1, 1)
        )
        @test_throws ErrorException("ContinuousStandardNode cannot have continuous parents, use ContinuousFunctionalNode instead") ContinuousStandardNode(name, [root1, root3], distribution)

        parents = [root1, root2]
        distribution = OrderedDict(
            [:yes, :y] => Normal(),
            [:no, :y] => Normal(1, 1),
            [:yes, :n] => Normal(2, 1)
        )
        @test_throws ErrorException("defined combinations in node.states must be equal to the theorical discrete parents combinations") ContinuousStandardNode(name, parents, distribution)

        distribution = OrderedDict(
            [:yes, :maybe] => Normal(),
            [:no, :y] => Normal(1, 1),
            [:yes, :n] => Normal(2, 1),
            [:no, :n] => Normal(3, 1)
        )
        @test_throws ErrorException("StandardNode state's keys must contain state from parent and the order of the parents states must be coherent with the order of the parents defined in node.parents") ContinuousStandardNode(name, parents, distribution)

        distribution = OrderedDict(
            [:yes, :y] => Normal(),
            [:no, :y] => Normal(1, 1),
            [:yes, :n] => Normal(2, 1),
            [:no, :n] => Normal(3, 1)
        )
        node = ContinuousStandardNode(name, [root1, root2], distribution)

        @test node.name == name
        @test node.parents == [root1, root2]
        @test node.distribution == distribution
        @test node.intervals == Vector{Vector{Float64}}()
        @test node.sigma == 0

        node = ContinuousStandardNode(:child, [root1], OrderedDict([:yes] => Normal(), [:no] => Normal(2, 2)))

        evidence = [(:a, root4)]
        @test_throws ErrorException("evidence does not contain any parents of the ContinuousStandardNode") get_randomvariable(node, evidence)

        evidence = [(:yes, root1)]
        @test get_randomvariable(node, evidence) == RandomVariable(Normal(), node.name)
    end

    @testset "DiscreteStandardNode" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
        root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6))
        root3 = ContinuousRootNode(:z, Normal())
        name = :child

        states = OrderedDict(
            [:yes] => Dict(:yes => -0.1, :no => 0.9),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("Probabilites must be nonnegative") DiscreteStandardNode(name, [root1], states)

        states = OrderedDict(
            [:yes] => Dict(:yes => 1.1, :no => 0.9),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("Probabilites must be less or equal to 1.0") DiscreteStandardNode(name, [root1], states)

        states = OrderedDict(
            [:yes] => Dict(:yes => 0.3, :no => 0.9),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("Probabilites must sum up to 1.0") DiscreteStandardNode(name, [root1], states)

        states = OrderedDict(
            [:yes, :yes] => Dict(:yes => 0.1, :no => 0.9),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("number of symbols per parent in node.states must be equal to the number of discrete parents") DiscreteStandardNode(name, [root1], states)

        parents = [root1, root2]
        states = OrderedDict(
            [:yes, :yes] => Dict(:yep => 0.2, :no => 0.8),
            [:no, :yes] => Dict(:yes => 0.2, :no => 0.8),
            [:yes, :no] => Dict(:yes => 0.2, :no => 0.8),
            [:no, :no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("NON coherent definition of nodes states in the ordered dict") DiscreteStandardNode(name, parents, states)

        states = OrderedDict(
            [:yes, :yes] => Dict(:yes => 0.2, :no => 0.8),
            [:no, :yes] => Dict(:yes => 0.2, :no => 0.8),
            [:yes, :no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("defined combinations in node.states must be equal to the theorical discrete parents combinations") DiscreteStandardNode(name, parents, states)

        states = OrderedDict(
            [:yes, :maybe] => Dict(:yes => 0.2, :no => 0.8),
            [:no, :yes] => Dict(:yes => 0.2, :no => 0.8),
            [:yes, :no] => Dict(:yes => 0.2, :no => 0.8),
            [:no, :no] => Dict(:yes => 0.2, :no => 0.8),
        )
        @test_throws ErrorException("StandardNode state's keys must contain state from parent and the order of the parents states must be coherent with the order of the parents defined in node.parents") DiscreteStandardNode(name, parents, states)

        states = OrderedDict(
            [:yes, :yes] => Dict(:a => 0.2, :b => 0.8),
            [:no, :yes] => Dict(:a => 0.2, :b => 0.8),
            [:yes, :no] => Dict(:a => 0.2, :b => 0.8),
            [:no, :no] => Dict(:a => 0.2, :b => 0.8)
        )

        node = DiscreteStandardNode(name, parents, states)
        @test node.name == name
        @test node.parents == parents
        @test node.states == states
        @test node.parameters == Dict{Symbol,Vector{Parameter}}()

        @test EnhancedBayesianNetworks._get_states(node) == [:a, :b]

        node = DiscreteStandardNode(name, parents, states, Dict(:a => [Parameter(1.1, :g)], :b => [Parameter(1.2, :g)]))
        evidence = [(:yes, root1)]
        @test_throws ErrorException("evidence does not contain DiscreteStandardNode in the evidence") get_parameters(node, evidence)

        evidence = [(:a, node)]
        @test get_parameters(node, evidence) == [Parameter(1.1, :g)]
    end
end
