@testset "Evaluation Net" begin

    @testset "Auxiliary Functions" begin
        root0 = DiscreteRootNode(:x0, Dict(:x01 => [0.3, 0.4], :x02 => [0.6, 0.7]), Dict(:x01 => [Parameter(0.5, :x0)], :x02 => [Parameter(0.7, :x0)]))
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
        disc_ebn = EnhancedBayesianNetworks._discretize(ebn)
        ebn2eval = _transfer_continuous(disc_ebn)
        nodes = ebn2eval.nodes
        nodes2reduce = filter(x -> isa(x, ContinuousNode) && !isa(x, FunctionalNode), nodes)
        indices2reduce = map(x -> ebn2eval.name_to_index[x.name], nodes2reduce)
        dag = deepcopy(ebn2eval.dag)
        i = first(filter(x -> isa(x, FunctionalNode), nodes))
        evaluated_i = EnhancedBayesianNetworks._evaluate(i)
        nodes = EnhancedBayesianNetworks._replace_node!(nodes, i, evaluated_i)

        EnhancedBayesianNetworks._clean_up!(nodes)
        @test evaluated_i == nodes[1]

        nodes_bn = [root3, root1]
        ebn = EnhancedBayesianNetwork(nodes_bn)
        bn = EnhancedBayesianNetworks.get_specific_network(nodes_bn)
        @test isa(bn, BayesianNetwork)
        @test issetequal(bn.nodes, nodes_bn)
        bn = EnhancedBayesianNetworks.get_specific_network(ebn)
        @test isa(bn, BayesianNetwork)
        @test issetequal(bn.nodes, nodes_bn)

        nodes_ebn = [root1, root2, root3, functional1]
        ebn = EnhancedBayesianNetworks.get_specific_network(nodes_ebn)
        @test isa(ebn, EnhancedBayesianNetwork)
        @test issetequal(ebn.nodes, nodes_ebn)
        ebn = EnhancedBayesianNetworks.get_specific_network(ebn)
        @test isa(ebn, EnhancedBayesianNetwork)
        @test issetequal(ebn.nodes, nodes_ebn)

        nodes_cn = [root0, root1]
        ebn = EnhancedBayesianNetwork(nodes_cn)
        cn = EnhancedBayesianNetworks.get_specific_network(nodes_cn)
        @test isa(cn, CredalNetwork)
        @test issetequal(cn.nodes, nodes_cn)
        cn = EnhancedBayesianNetworks.get_specific_network(ebn)
        @test isa(bn, BayesianNetwork)
        @test issetequal(bn.nodes, nodes_bn)
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
        evaluated_ebn2 = evaluate_with_envelope(ebn)

        fadjlist = Vector{Vector{Int}}([[3], [3, 4], [4], []])
        badjlist = Vector{Vector{Int}}([[], [], [1, 2], [2, 3]])

        name_to_index = Dict(:fd => 4, :fd1 => 3, :z => 2, :x => 1)

        @test evaluated_ebn.dag == DiGraph(4, fadjlist, badjlist)
        @test evaluated_ebn.name_to_index == name_to_index
        @test typeof(evaluated_ebn.nodes[4]) == DiscreteChildNode
        @test evaluated_ebn.dag == evaluated_ebn2.dag
        @test evaluated_ebn.name_to_index == evaluated_ebn2.name_to_index
        @test typeof(evaluated_ebn2.nodes[4]) == DiscreteChildNode

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

    @testset "Add missing nodes to envelopes" begin
        using .MathConstants: γ
        Uᵣ = ContinuousRootNode(:Uᵣ, Normal())
        μ_gamma = 60
        cov_gamma = 0.2
        M = DiscreteRootNode(:M, Dict(:new => 0.5, :old => 0.5))
        α, θ = distribution_parameters(μ_gamma, μ_gamma * cov_gamma, Gamma)
        V = ContinuousChildNode(:V, [M], Dict(
            [:new] => Gamma(α, θ),
            [:old] => Gamma(α - 1, 2.4)
        ))
        μ_gumbel = 50
        cov_gumbel = 0.4
        μ_loc, β = distribution_parameters(μ_gumbel, cov_gumbel * μ_gumbel, Gumbel)
        H = ContinuousRootNode(:H, Gumbel(μ_loc, β))
        function plastic_moment_capacities(uᵣ)
            ρ = 0.5477
            μ = 150
            cov = 0.2
            λ, ζ = distribution_parameters(μ, μ * cov, LogNormal)
            normal_μ = λ + ρ * ζ * uᵣ
            normal_std = sqrt((1 - ρ^2) * ζ^2)
            exp(rand(Normal(normal_μ, normal_std)))
        end
        parents = [Uᵣ]
        model1 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r1)
        model2 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r2)
        model3 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r3)
        model4 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r4)
        model5 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r5)
        R1 = ContinuousFunctionalNode(:R1, parents, [model1], MonteCarlo(10^6))
        R2 = ContinuousFunctionalNode(:R2, parents, [model2], MonteCarlo(10^6))
        R3 = ContinuousFunctionalNode(:R3, parents, [model3], MonteCarlo(10^6))
        R4 = ContinuousFunctionalNode(:R4, parents, [model4], MonteCarlo(10^6))
        R5 = ContinuousFunctionalNode(:R5, parents, [model5], MonteCarlo(10^6))
        function frame_model(r1, r2, r3, r4, r5, v, h)
            g1 = r1 + r2 + r4 + r5 - 5 * h
            g2 = r2 + 2 * r3 + r4 - 5 * v
            g3 = r1 + 2 * r3 + 2 * r4 + r5 - 5 * h - 5 * v
            return minimum([g1, g2, g3])
        end
        model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.r4, df.r5, df.V, df.H), :G)
        performance = df -> df.G
        simulation = MonteCarlo(10^6)
        frame = DiscreteFunctionalNode(:E, [R1, R2, R3, R4, R5, V, H], [model], performance, simulation)
        L = DiscreteChildNode(:L, [M], Dict(
                [:old] => Dict(:yesL => 0.5, :noL => 0.5),
                [:new] => Dict(:yesL => 0.2, :noL => 0.8),
            ), Dict(:noL => [Parameter(1, :L)], :yesL => [Parameter(2, :L)]))
        r9 = ContinuousRootNode(:R9, Normal())
        model2 = Model(df -> df.L .^ 2 .* df.R9, :P)
        frame2 = DiscreteFunctionalNode(:E2, [L, r9], [model2], df -> df.P, simulation)
        nodes = [Uᵣ, M, V, H, R1, R2, R3, R4, R5, r9, frame, L, frame2]
        ebn = EnhancedBayesianNetwork(nodes)
        ebns = markov_envelope(ebn)
        ebns = EnhancedBayesianNetworks._add_missing_nodes_to_envelope.(ebns)
        @test issetequal([Uᵣ, M, V, H, R1, R2, R3, R4, R5, frame], ebns[1])
        @test issetequal([r9, M, frame2, L], ebns[2])

        res = evaluate_with_envelope(ebn)
        @test isa(res, BayesianNetwork)
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