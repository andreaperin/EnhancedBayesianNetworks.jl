@testset "Evaluation Net" begin

    @testset "Auxiliary Functions" begin
        root1 = DiscreteRootNode(:x, Dict(:x1 => 0.3, :x2 => 0.7), Dict(:x1 => [Parameter(0.5, :x)], :x2 => [Parameter(0.7, :x)]))
        root2 = ContinuousRootNode(:y, Normal())
        root3 = DiscreteRootNode(:z, Dict(:z1 => 0.3, :z2 => 0.7), Dict(:z1 => [Parameter(0.5, :z)], :z2 => [Parameter(0.7, :z)]))

        model1 = Model(df -> df.x .^ 2 .- 0.7 .+ df.y, :c1)
        performance1 = df -> 0.5 .- df.c1
        functional1 = DiscreteFunctionalNode(:cf1, [root1, root2], [model1], performance1, MonteCarlo(300))

        nodes = [root1, root2, root3, functional1]

        @test EnhancedBayesianNetworks._count_children(root1, nodes) == 1

        child1 = ContinuousChildNode(:ch1, [root1], Dict([:x1] => Normal(), [:x2] => Normal(1, 1)))
        _nodes = deepcopy(nodes)
        push!(_nodes, child1)
        EnhancedBayesianNetworks._clean_up!(_nodes)

        @test issetequal(_nodes, nodes)

        _nodes = deepcopy(nodes)
        new = EnhancedBayesianNetworks._evaluate(functional1)
        EnhancedBayesianNetworks._replace_node!(_nodes, functional1, new)
        @test issetequal(_nodes, [root1, root2, root3, new])

        root4 = ContinuousRootNode(:a, Normal())
        root5 = ContinuousRootNode(:b, Normal())
        root6 = ContinuousRootNode(:c, Normal())

        model2 = Model(df -> df.a .^ 2 .- 0.7 .+ df.b .- df.c, :c2)
        functional2 = ContinuousFunctionalNode(:cf1, [root4, root5, root6], [model2], MonteCarlo(300))
        _nodes = [root4, root5, root6, functional2]
        ebn = EnhancedBayesianNetwork(_nodes)
        disc_ebn = discretize(ebn)
        ebn2eval = transfer_continuous(disc_ebn)
        nodes = ebn2eval.nodes
        nodes2reduce = filter(x -> isa(x, ContinuousNode) && !isa(x, FunctionalNode), nodes)
        indices2reduce = map(x -> ebn2eval.name_to_index[x.name], nodes2reduce)
        dag = deepcopy(ebn2eval.dag)
        i = first(filter(x -> isa(x, FunctionalNode), nodes))
        evaluated_i = EnhancedBayesianNetworks._evaluate(i)
        nodes = EnhancedBayesianNetworks._replace_node!(nodes, i, evaluated_i)

        EnhancedBayesianNetworks._clean_up!(nodes)
        @test evaluated_i == nodes[1]
    end

    @testset "Main Functions" begin

        root1 = DiscreteRootNode(:x, Dict(:x1 => 0.3, :x2 => 0.7), Dict(:x1 => [Parameter(0.5, :x)], :x2 => [Parameter(0.7, :x)]))
        root2 = ContinuousRootNode(:y, Normal())
        root3 = DiscreteRootNode(:z, Dict(:z1 => 0.3, :z2 => 0.7), Dict(:z1 => [Parameter(0.5, :z)], :z2 => [Parameter(0.7, :z)]))

        model1 = Model(df -> df.x .^ 2 .- 0.7 .+ df.y, :c1)
        cont_functional1 = ContinuousFunctionalNode(:cf1, [root1, root2], [model1], MonteCarlo(300))

        model2 = Model(df -> df.z .^ 2 .- 0.7 .+ df.y, :c2)
        cont_functional2 = ContinuousFunctionalNode(:cf2, [root2, root3], [model2], MonteCarlo(300))

        model3 = Model(df -> df.c1 .* 0.5 .+ df.c2, :final1)
        performance1 = df -> df.final1 .- 0.5
        discrete_functional1 = DiscreteFunctionalNode(:fd1, [cont_functional1, cont_functional2], [model3], performance1, MonteCarlo(300), Dict(:fail_fd1 => [Parameter(1, :fd1)], :safe_fd1 => [Parameter(0, :fd1)]))

        model4 = Model(df -> df.c2 .* 0.5, :c3)
        continuous_functional3 = ContinuousFunctionalNode(:c3, [cont_functional2], [model4], MonteCarlo(300))

        model5 = Model(df -> 0.5 .+ df.c3, :tot)
        performance2 = df -> 0.5 .- df.tot
        discrete_functional = DiscreteFunctionalNode(:fd, [discrete_functional1, continuous_functional3], [model5], performance2, MonteCarlo(300))

        nodes = [root1, root2, root3, cont_functional1, cont_functional2, discrete_functional1, continuous_functional3, discrete_functional]
        ebn = EnhancedBayesianNetwork(nodes)

        res1 = EnhancedBayesianNetworks._evaluate_routine(ebn)

        fadjlist = Vector{Vector{Int}}([[4], [5], [4, 5], [5], []])
        badjlist = Vector{Vector{Int}}([[], [], [], [1, 3], [2, 3, 4]])

        name_to_index = Dict(:y => 2, :fd => 5, :fd1 => 4, :z => 3, :x => 1)

        @test res1.dag == DiGraph(5, fadjlist, badjlist)
        @test res1.name_to_index == name_to_index
        @test typeof(res1.nodes[4]) == DiscreteChildNode

        evaluated_ebn = evaluate(ebn)

        fadjlist = Vector{Vector{Int}}([[3], [3, 4], [4], []])
        badjlist = Vector{Vector{Int}}([[], [], [1, 2], [2, 3]])

        name_to_index = Dict(:fd => 4, :fd1 => 3, :z => 2, :x => 1)

        @test evaluated_ebn.dag == DiGraph(4, fadjlist, badjlist)
        @test evaluated_ebn.name_to_index == name_to_index
        @test typeof(evaluated_ebn.nodes[4]) == DiscreteChildNode

        interval = (1.10, 1.30)
        root1 = DiscreteRootNode(:A, Dict(:a1 => 0.5, :a2 => 0.5), Dict(:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]))
        root2 = ContinuousRootNode(:B, interval)
        root3 = ContinuousRootNode(:P, Uniform(-10, 10))
        model = Model(df -> df.A .+ df.B .+ df.P, :C)
        sim = DoubleLoop(MonteCarlo(100_000))
        performance = df -> 2 .- df.C
        disc_functional = DiscreteFunctionalNode(:C, [root1, root2, root3], [model], performance, sim)

        nodes = [root1, root2, root3, disc_functional]
        ebn = EnhancedBayesianNetwork(nodes)
        credal = evaluate(ebn)
        @test typeof(credal) == CredalNetwork
    end

    @testset "No Ancestors case" begin
        root2 = ContinuousRootNode(:B, Normal())
        model = Model(df -> df.B .+ 1, :C)
        sim = MonteCarlo(100_000)
        performance = df -> 2 .- df.C
        cont_functional = ContinuousFunctionalNode(:C, [root2], [model], sim)

        ebn = EnhancedBayesianNetwork([root2, cont_functional])

        eebn = evaluate(ebn)

        @test length(eebn.nodes) == 1
        @test typeof.(eebn.nodes) == [ContinuousRootNode]

        disc_functional = DiscreteFunctionalNode(:C, [root2], [model], performance, sim)
        ebn = EnhancedBayesianNetwork([root2, disc_functional])

        eebn = evaluate(ebn)

        @test length(eebn.nodes) == 1
        @test typeof.(eebn.nodes) == [DiscreteRootNode]
    end

    @testset "Imprecise Node with discretization" begin
        root1 = DiscreteRootNode(:A, Dict(:y => 0.5, :n => 0.5))
        root2 = ContinuousChildNode(:B, [root1], Dict(
                [:y] => UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)]),
                [:n] => UnamedProbabilityBox{Normal}([Interval(1, 2, :μ), Interval(1, 2, :σ)])
            ), ApproximatedDiscretization([-1, 0, 1], 2)
        )
        model = Model(df -> df.B .+ 1, :C)
        sim = DoubleLoop(MonteCarlo(100_000))
        performance = df -> 2 .- df.C
        disc_functional = DiscreteFunctionalNode(:C, [root2], [model], performance, sim)
        ebn = EnhancedBayesianNetwork([root1, root2, disc_functional])

        @test_throws ErrorException("node C has as imprecise parents only one or more child nodes with a discretization srtucture defined. They are approximated with Uniform and Exponential assumption and they are no more imprecise. A prices simulation technique must be selected") eebn = @suppress evaluate(ebn)

        root2 = ContinuousRootNode(:B, (-1, 1))
        model = Model(df -> df.B .+ 1, :C)
        sim = MonteCarlo(100_000)
        performance = df -> 2 .- df.C

        disc_functional = DiscreteFunctionalNode(:C, [root2], [model], performance, sim)
        ebn = EnhancedBayesianNetwork([root2, disc_functional])

        @test_throws ErrorException("node C has MonteCarlo(100000) as simulation technique, but have [:B] as imprecise parent/s. DoubleLoop or RandomSlicing technique must be employeed instead.") eebn = @suppress evaluate(ebn)
    end
end