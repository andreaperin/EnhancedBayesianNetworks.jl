@testset "Child Nodes" begin
    @testset "ContinuousChildNode" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
        root2 = DiscreteRootNode(:y, Dict(:y => 0.4, :n => 0.6))
        root3 = ContinuousRootNode(:z, Normal())
        root4 = DiscreteRootNode(:k, Dict(:a => 0.5, :b => 0.5), Dict(:a => [Parameter(1, :k)], :b => [Parameter(2, :k)]))
        model = Model(df -> df.z .+ df.k, :F)
        performance = df -> 0 .- df.F
        sim = MonteCarlo(200)
        functional = DiscreteFunctionalNode(:F, [root3, root4], [model], performance, sim)
        name = :child

        distribution = Dict(
            [:yes, :yes] => Normal(),
            [:no] => Normal(1, 1)
        )
        @test_throws ErrorException("Defined combinations, [[:yes, :yes], [:no]] ,are not equal to the theorical discrete parents combinations: [[:yes], [:no]]") ContinuousChildNode(name, [root1], distribution)

        distribution = Dict(
            [:yes] => Normal(),
            [:no] => Normal(1, 1)
        )
        @test_throws ErrorException("Children of continuous node/s [:z], must be defined through a FunctionalNode struct") ContinuousChildNode(name, [root1, root3], distribution)

        parents = [root1, root2]
        distribution = Dict(
            [:yes, :y] => Normal(),
            [:no, :y] => Normal(1, 1),
            [:yes, :n] => Normal(2, 1)
        )
        @test_throws ErrorException("Defined combinations, [[:yes, :y], [:no, :y], [:yes, :n]] ,are not equal to the theorical discrete parents combinations: [[:yes, :n] [:yes, :y]; [:no, :n] [:no, :y]]") ContinuousChildNode(name, parents, distribution)

        distribution = Dict(
            [:yes, :maybe] => Normal(),
            [:no, :y] => Normal(1, 1),
            [:yes, :n] => Normal(2, 1),
            [:no, :n] => Normal(3, 1)
        )
        @test_throws ErrorException("Defined combinations, [[:yes, :maybe], [:no, :n], [:no, :y], [:yes, :n]] ,are not equal to the theorical discrete parents combinations: [[:yes, :n] [:yes, :y]; [:no, :n] [:no, :y]]") ContinuousChildNode(name, parents, distribution)

        distribution = Dict(
            [:yes, :y] => Normal(),
            [:no, :y] => Normal(1, 1),
            [:yes, :n] => Normal(2, 1),
            [:no, :n] => Normal(3, 1)
        )

        @test_throws ErrorException("Children of functional node/s [:F], must be defined through a FunctionalNode struct") ContinuousChildNode(name, [root1, root2, functional], distribution)

        node = ContinuousChildNode(name, [root1, root2], distribution)
        @test node.name == name
        @test issetequal(node.parents, [root1, root2])
        @test node.distribution == distribution
        @test isequal(node.discretization, ApproximatedDiscretization())
        @test node.additional_info == Dict{Vector{Symbol},Dict}()

        node = ContinuousChildNode(name, [root1, root2], distribution, Dict{Vector{Symbol},Dict}())
        @test node.name == name
        @test issetequal(node.parents, [root1, root2])
        @test node.distribution == distribution
        @test isequal(node.discretization, ApproximatedDiscretization())
        @test node.additional_info == Dict{Vector{Symbol},Dict}()

        node = ContinuousChildNode(:child, [root1], Dict([:yes] => Normal(), [:no] => Normal(2, 2)))
        evidence = [:a]

        @test_throws ErrorException("evidence [:a] does not contain all the parents of the ContinuousChildNode child") get_continuous_input(node, evidence)

        evidence = [:yes]
        @test get_continuous_input(node, evidence) == RandomVariable(Normal(), node.name)
        @test EnhancedBayesianNetworks._get_node_distribution_bounds(node) == (-Inf, Inf)
        @test EnhancedBayesianNetworks._is_imprecise(node) == false

        @testset "Imprecise Child - Interval" begin
            states = Dict(
                [:yes] => (0.1, 0.3),
                [:no] => (0.6, 0.7)
            )
            child = ContinuousChildNode(:child, [root1], states)
            @test child.name == name
            @test issetequal(child.parents, [root1])
            @test child.distribution == states
            @test isequal(child.discretization, ApproximatedDiscretization())
            @test child.additional_info == Dict{Vector{Symbol},Dict}()

            @test get_continuous_input(child, [:yes]) == Interval(0.1, 0.3, :child)
            @test EnhancedBayesianNetworks._get_node_distribution_bounds(child) == (0.1, 0.7)
            @test EnhancedBayesianNetworks._is_imprecise(child)
        end
        @testset "Imprecise Child - Interval" begin
            normal_pbox = UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)])
            uniform_pbox = UnamedProbabilityBox{Uniform}([Interval(1, 3, :a), Interval(4, 5, :b)])
            states = Dict(
                [:yes] => normal_pbox,
                [:no] => uniform_pbox
            )
            child = ContinuousChildNode(:child, [root1], states)

            yes_input = get_continuous_input(child, [:yes])
            @test typeof(yes_input) == ProbabilityBox{Normal}
            @test yes_input.lb == normal_pbox.lb
            @test yes_input.ub == normal_pbox.ub
            @test yes_input.name == :child
            @test yes_input.parameters == normal_pbox.parameters

            # ! TODO uncomment this test when issue 204 in UncertaintyQuantification.jl is closed
            # no_input = get_continuous_input(child, [:no])
            # @test typeof(no_input) == ProbabilityBox{Uniform}
            # @test no_input.lb == normal_pbox.lb
            # @test no_input.ub == normal_pbox.ub
            # @test no_input.name == :child
            # @test no_input.parameters == normal_pbox.parameters
        end

    end

    @testset "DiscreteChildNode" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
        root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6), Dict(:yes => [Parameter(1, :y)], :no => [Parameter(2, :y)]))
        root3 = ContinuousRootNode(:z, Normal())
        model = Model(df -> df.z .+ df.y, :F)
        performance = df -> 0 .- df.F
        sim = MonteCarlo(200)
        functional = DiscreteFunctionalNode(:F, [root3, root2], [model], performance, sim)
        name = :child

        states = Dict(
            [:yes] => Dict(:yes => [0.1, 0.2], :no => 0.9),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )

        @test_throws ErrorException("node child has mixed interval and single value states probabilities!") DiscreteChildNode(name, [root1], states)

        states = Dict(
            [:yes] => Dict(:yes => 0.1, :no => 0.9),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )
        covs = Dict(
            [:yes] => "a",
            [:no] => 0.1
        )
        additional_info = Dict{Vector{Symbol},Dict}()
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
        @test_throws ErrorException("defined states probabilities Dict{Symbol, Real}(:yes => 0.3, :no => 0.9) are wrong") DiscreteChildNode(name, [root1], states)

        states = Dict(
            [:yes] => Dict(:yes => 0.4999, :no => 0.4999),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )

        @test_logs (:warn, "total probaility should be one, but the evaluated value is 0.9998 , and will be normalized") DiscreteChildNode(name, [root1], states)

        states = Dict(
            [:yes, :yes] => Dict(:yes => 0.1, :no => 0.9),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("Defined combinations, [[:yes, :yes], [:no]] ,are not equal to the theorical discrete parents combinations: [[:yes], [:no]]") DiscreteChildNode(name, [root1], states)

        parents = [root1, root2]
        states = Dict(
            [:yes, :yes] => Dict(:yep => 0.2, :no => 0.8),
            [:no, :yes] => Dict(:yes => 0.2, :no => 0.8),
            [:yes, :no] => Dict(:yes => 0.2, :no => 0.8),
            [:no, :no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("non-coherent definition of nodes states: [:yes, :no, :yep]") DiscreteChildNode(name, parents, states)

        states = Dict(
            [:yes, :yes] => Dict(:yes => 0.2, :no => 0.8),
            [:no, :yes] => Dict(:yes => 0.2, :no => 0.8),
            [:yes, :no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("Defined combinations, [[:yes, :no], [:yes, :yes], [:no, :yes]] ,are not equal to the theorical discrete parents combinations: [[:yes, :yes] [:yes, :no]; [:no, :yes] [:no, :no]]") DiscreteChildNode(name, parents, states)

        states = Dict(
            [:yes, :maybe] => Dict(:yes => 0.2, :no => 0.8),
            [:no, :yes] => Dict(:yes => 0.2, :no => 0.8),
            [:yes, :no] => Dict(:yes => 0.2, :no => 0.8),
            [:no, :no] => Dict(:yes => 0.2, :no => 0.8),
        )
        @test_throws ErrorException("Defined combinations, [[:yes, :no], [:no, :no], [:yes, :maybe], [:no, :yes]] ,are not equal to the theorical discrete parents combinations: [[:yes, :yes] [:yes, :no]; [:no, :yes] [:no, :no]]") DiscreteChildNode(name, parents, states)

        states = Dict(
            [:yes, :yes] => Dict(:a => 0.2, :b => 0.8),
            [:no, :yes] => Dict(:a => 0.2, :b => 0.8),
            [:yes, :no] => Dict(:a => 0.2, :b => 0.8),
            [:no, :no] => Dict(:a => 0.2, :b => 0.8)
        )

        @test_throws ErrorException("Children of functional node/s [:F], must be defined through a FunctionalNode struct") DiscreteChildNode(name, [root1, root2, functional], states)

        node = DiscreteChildNode(name, parents, states)
        @test node.name == name
        @test issetequal(node.parents, parents)
        @test node.states == states
        @test node.parameters == Dict{Symbol,Vector{Parameter}}()
        @test node.additional_info == Dict{Vector{Symbol},Dict}()

        node = DiscreteChildNode(name, parents, states, Dict{Vector{Symbol},Dict}())
        @test node.name == name
        @test issetequal(node.parents, parents)
        @test node.states == states
        @test node.parameters == Dict{Symbol,Vector{Parameter}}()
        @test node.additional_info == Dict{Vector{Symbol},Dict}()

        @test EnhancedBayesianNetworks._get_states(node) == [:a, :b]

        node = DiscreteChildNode(name, parents, states, Dict(:a => [Parameter(1.1, :g)], :b => [Parameter(1.2, :g)]))
        evidence = [:yes]
        @test_throws ErrorException("evidence [:yes] does not contain child") get_parameters(node, evidence)

        node_ = DiscreteChildNode(name, parents, states)
        @test_throws ErrorException("node child has an empty parameters vector") get_parameters(node_, evidence)

        evidence = [:a, :yes]
        @test get_parameters(node, evidence) == [Parameter(1.1, :g)]
        @test EnhancedBayesianNetworks._is_imprecise(node) == false
        @testset "Imprecise Child - Interval" begin
            parents = [root1, root2]
            states = Dict(
                [:yes, :yes] => Dict(:a => [0.4, 0.5], :b => [0.5, 0.6]),
                [:no, :yes] => Dict(:a => [0.4, 0.5], :b => [0.5, 0.6]),
                [:yes, :no] => Dict(:a => [0.4, 0.5], :b => [0.5, 0.6]),
                [:no, :no] => Dict(:a => [0.4, 0.5], :b => [0.5, 0.6])
            )
            parameters = Dict(:a => [Parameter(1, :a1)], :b => [Parameter(2, :a1)])
            child = DiscreteChildNode(name, parents, states, parameters)

            @test child.name == name
            @test issetequal(child.parents, [root1, root2])
            @test child.states == states
            @test child.parameters == parameters
            @test node.additional_info == Dict{Vector{Symbol},Dict}()

            @test EnhancedBayesianNetworks.get_parameters(child, [:a]) == [Parameter(1, :a1)]
            @test EnhancedBayesianNetworks._get_states(child) == [:a, :b]
            @test EnhancedBayesianNetworks._is_imprecise(child)

            child = DiscreteChildNode(name, [root1], Dict(
                [:yes] => Dict(:y => [0.4, 0.5], :n => [0.5, 0.6]),
                [:no] => Dict(:y => [0.5, 0.6], :n => [0.4, 0.5])
            ))
            extreme_points = EnhancedBayesianNetworks._extreme_points(child)

            n1 = DiscreteChildNode(name, [root1], Dict(
                [:yes] => Dict(:y => 0.4, :n => 0.6),
                [:no] => Dict(:y => 0.5, :n => 0.5)
            ))

            n2 = DiscreteChildNode(name, [root1], Dict(
                [:yes] => Dict(:y => 0.5, :n => 0.5),
                [:no] => Dict(:y => 0.5, :n => 0.5)
            ))

            n3 = DiscreteChildNode(name, [root1], Dict(
                [:yes] => Dict(:y => 0.5, :n => 0.5),
                [:no] => Dict(:y => 0.6, :n => 0.4)
            ))

            n4 = DiscreteChildNode(name, [root1], Dict(
                [:yes] => Dict(:y => 0.4, :n => 0.6),
                [:no] => Dict(:y => 0.6, :n => 0.4)
            ))

            @test issetequal(extreme_points, [n1, n2, n3, n4])

            child = DiscreteChildNode(name, [root1], Dict(
                [:yes] => Dict(:y => [0.1, 0.3], :n => [0.5, 0.6], :m => [0.1, 0.3]),
                [:no] => Dict(:y => [0.5, 0.6], :n => [0.4, 0.5], :m => [0.1, 0.3])
            ))
            extreme_points = @suppress EnhancedBayesianNetworks._extreme_points(child)

            @test isapprox(collect(values(extreme_points[1].states[[:yes]])), [0.1, 0.6, 0.3])
            @test isapprox(collect(values(extreme_points[2].states[[:yes]])), [0.2, 0.5, 0.3])
            @test isapprox(collect(values(extreme_points[3].states[[:yes]])), [0.3, 0.6, 0.1])
            @test isapprox(collect(values(extreme_points[4].states[[:yes]])), [0.3, 0.5, 0.2])
        end
    end
end
