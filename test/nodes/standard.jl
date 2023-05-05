@testset "Standard Nodes" begin
    @testset "ContinuousStandardNode" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
        root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6))
        root3 = ContinuousRootNode(RandomVariable(Normal(), :z))
        name = :child

        distribution = OrderedDict(
            [:yes, :yes] => Normal(),
            [:no] => Normal(1, 1)
        )
        @test_throws ErrorException("number of symbols per parent in node.states must be equal to the number of discrete parents") ContinuousStandardNode(name, [root1], distribution)

        parents = [root1, root2, root3]
        distribution = OrderedDict(
            [:yes, :yes] => Normal(),
            [:no, :yes] => Normal(1, 1),
            [:yes, :no] => Normal(2, 1)
        )
        @test_throws ErrorException("defined combinations in node.states must be equal to the theorical discrete parents combinations") ContinuousStandardNode(name, parents, distribution)

        distribution = OrderedDict(
            [:yes, :maybe] => Normal(),
            [:no, :yes] => Normal(1, 1),
            [:yes, :no] => Normal(2, 1),
            [:no, :no] => Normal(3, 1)
        )
        @test_throws ErrorException("StandardNode state's keys must contain state from parent and the order of the parents states must be coherent with the order of the parents defined in node.parents") ContinuousStandardNode(name, parents, distribution)
    end

    @testset "DiscreteStandardNode" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
        root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6))
        root3 = ContinuousRootNode(RandomVariable(Normal(), :z))
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

        parents = [root1, root2, root3]
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
        @test EnhancedBayesianNetworks._get_states(node) == [:a, :b]
    end
end
## TODO add Test when a StdNode has a StdNode as parents (the same for FunctionalNodes)