@testset "Child Nodes" begin
    @testset "ContinuousChildNode" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
        root2 = DiscreteRootNode(:y, Dict(:y => 0.4, :n => 0.6))
        root3 = ContinuousRootNode(:z, Normal())
        root4 = DiscreteRootNode(:k, Dict(:a => 0.5, :b => 0.5))
        name = :child

        distributions = Dict(
            [:yes, :yes] => Normal(),
            [:no] => Normal(1, 1)
        )
        @test_throws ErrorException("In node child, defined parents states differ from number of its discrete parents") ContinuousChildNode(name, [root1], distributions)

        distributions = Dict(
            [:yes] => Normal(),
            [:no] => Normal(1, 1)
        )
        @test_throws ErrorException("ContinuousChildNode child cannot have continuous parents! Use ContinuousFunctionalNode instead") ContinuousChildNode(name, [root1, root3], distributions)

        parents = [root1, root2]
        distributions = Dict(
            [:yes, :y] => Normal(),
            [:no, :y] => Normal(1, 1),
            [:yes, :n] => Normal(2, 1)
        )
        @test_throws ErrorException("In node child, defined combinations are not equal to the theorical discrete parents combinations: [[:yes, :n] [:yes, :y]; [:no, :n] [:no, :y]]") ContinuousChildNode(name, parents, distributions)

        distributions = Dict(
            [:yes, :maybe] => Normal(),
            [:no, :y] => Normal(1, 1),
            [:yes, :n] => Normal(2, 1),
            [:no, :n] => Normal(3, 1)
        )
        @test_throws ErrorException("In node child, defined parents states are not coherent with its discrete parents states") ContinuousChildNode(name, parents, distributions)

        distributions = Dict(
            [:yes, :y] => Normal(),
            [:no, :y] => Normal(1, 1),
            [:yes, :n] => Normal(2, 1),
            [:no, :n] => Normal(3, 1)
        )
        node = ContinuousChildNode(name, [root1, root2], distributions)

        @test node.name == name
        @test issetequal(node.parents, [root1, root2])
        @test node.distributions == distributions
        @test isequal(node.discretization, ApproximatedDiscretization())
        @test node.samples == Dict{Vector{Symbol},DataFrame}()

        node = ContinuousChildNode(:child, [root1], Dict([:yes] => Normal(), [:no] => Normal(2, 2)))

        evidence = [:a]
        @test_throws ErrorException("evidence [:a] does not contain all the parents of the ContinuousChildNode child") get_randomvariable(node, evidence)

        evidence = [:yes]
        @test get_randomvariable(node, evidence) == RandomVariable(Normal(), node.name)
    end

    @testset "DiscreteChildNode" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
        root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6))
        root3 = ContinuousRootNode(:z, Normal())
        name = :child

        states = Dict(
            [:yes] => Dict(:yes => -0.1, :no => 0.9),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("Probabilites must be nonnegative") DiscreteChildNode(name, [root1], states)

        states = Dict(
            [:yes] => Dict(:yes => 1.1, :no => 0.9),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("Probabilites must be less or equal to 1.0") DiscreteChildNode(name, [root1], states)

        states = Dict(
            [:yes] => Dict(:yes => 0.3, :no => 0.9),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("defined states probabilities Dict(:yes => 0.3, :no => 0.9) are wrong") DiscreteChildNode(name, [root1], states)

        states = Dict(
            [:yes] => Dict(:yes => 0.4999, :no => 0.4999),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )

        @test_logs (:warn, "total probaility should be one, but the evaluated value is 0.9998 , and will be normalized") DiscreteChildNode(name, [root1], states)

        states = Dict(
            [:yes, :yes] => Dict(:yes => 0.1, :no => 0.9),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("In node child, defined parents states differ from number of its discrete parents") DiscreteChildNode(name, [root1], states)

        parents = [root1, root2]
        states = Dict(
            [:yes, :yes] => Dict(:yep => 0.2, :no => 0.8),
            [:no, :yes] => Dict(:yes => 0.2, :no => 0.8),
            [:yes, :no] => Dict(:yes => 0.2, :no => 0.8),
            [:no, :no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("node child: non-coherent definition of nodes states") DiscreteChildNode(name, parents, states)

        states = Dict(
            [:yes, :yes] => Dict(:yes => 0.2, :no => 0.8),
            [:no, :yes] => Dict(:yes => 0.2, :no => 0.8),
            [:yes, :no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("In node child, defined combinations are not equal to the theorical discrete parents combinations: [[:yes, :yes] [:yes, :no]; [:no, :yes] [:no, :no]]") DiscreteChildNode(name, parents, states)

        states = Dict(
            [:yes, :maybe] => Dict(:yes => 0.2, :no => 0.8),
            [:no, :yes] => Dict(:yes => 0.2, :no => 0.8),
            [:yes, :no] => Dict(:yes => 0.2, :no => 0.8),
            [:no, :no] => Dict(:yes => 0.2, :no => 0.8),
        )
        @test_throws ErrorException("In node child, defined parents states are not coherent with its discrete parents states") DiscreteChildNode(name, parents, states)

        states = Dict(
            [:yes, :yes] => Dict(:a => 0.2, :b => 0.8),
            [:no, :yes] => Dict(:a => 0.2, :b => 0.8),
            [:yes, :no] => Dict(:a => 0.2, :b => 0.8),
            [:no, :no] => Dict(:a => 0.2, :b => 0.8)
        )

        node = DiscreteChildNode(name, parents, states)
        @test node.name == name
        @test issetequal(node.parents, parents)
        @test node.states == states
        @test node.parameters == Dict{Symbol,Vector{Parameter}}()
        @test node.covs == Dict{Vector{Symbol},Number}()
        @test node.samples == Dict{Symbol,Vector{Parameter}}()

        @test EnhancedBayesianNetworks._get_states(node) == [:a, :b]

        node = DiscreteChildNode(name, parents, states, Dict(:a => [Parameter(1.1, :g)], :b => [Parameter(1.2, :g)]))
        evidence = [:yes]
        @test_throws ErrorException("evidence [:yes] does not contain child") get_parameters(node, evidence)

        node_ = DiscreteChildNode(name, parents, states)
        @test_throws ErrorException("node child has an empty parameters vector") get_parameters(node_, evidence)

        evidence = [:a, :yes]
        @test get_parameters(node, evidence) == [Parameter(1.1, :g)]
    end
end
