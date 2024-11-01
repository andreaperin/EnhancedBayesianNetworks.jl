@testset "Evaluation Net" begin

    @testset "Main Functions" begin

        root1 = DiscreteRootNode(:x, Dict(:x1 => 0.3, :x2 => 0.7), Dict(:x1 => [Parameter(0.5, :x)], :x2 => [Parameter(0.7, :x)]))
        root2 = ContinuousRootNode(:y, Normal())
        root3 = DiscreteRootNode(:z, Dict(:z1 => 0.3, :z2 => 0.7), Dict(:z1 => [Parameter(0.5, :z)], :z2 => [Parameter(0.7, :z)]))

        model1 = Model(df -> df.x .^ 2 .- 0.7 .+ df.y, :c1)
        cont_functional1 = ContinuousFunctionalNode(:cf1, [model1], MonteCarlo(300))

        model2 = Model(df -> df.z .^ 2 .- 0.7 .+ df.y, :c2)
        cont_functional2 = ContinuousFunctionalNode(:cf2, [model2], MonteCarlo(300))

        model3 = Model(df -> df.c1 .* 0.5 .+ df.c2, :final1)
        performance1 = df -> df.final1 .- 0.5
        discrete_functional1 = DiscreteFunctionalNode(:fd1, [model3], performance1, MonteCarlo(300), Dict(:fail_fd1 => [Parameter(1, :fd1)], :safe_fd1 => [Parameter(0, :fd1)]))

        model4 = Model(df -> df.c2 .* 0.5, :c3)
        cont_functional3 = ContinuousFunctionalNode(:c3, [model4], MonteCarlo(300))

        model5 = Model(df -> 0.5 .+ df.c3, :tot)
        performance2 = df -> 0.5 .- df.tot
        discrete_functional = DiscreteFunctionalNode(:fd, [model5], performance2, MonteCarlo(300))

        nodes = [root1, root2, root3, cont_functional1, cont_functional2, discrete_functional1, cont_functional3, discrete_functional]

        ebn = EnhancedBayesianNetwork(nodes)
        add_child!(ebn, root1, cont_functional1)
        add_child!(ebn, root2, cont_functional1)
        add_child!(ebn, root2, cont_functional2)
        add_child!(ebn, root3, cont_functional2)
        add_child!(ebn, cont_functional1, discrete_functional1)
        add_child!(ebn, cont_functional2, discrete_functional1)
        add_child!(ebn, cont_functional2, cont_functional3)
        add_child!(ebn, discrete_functional1, discrete_functional)
        add_child!(ebn, cont_functional3, discrete_functional)
        order_net!(ebn)
        net1 = deepcopy(ebn)

        evaluate!(ebn)

        @test ebn.adj_matrix == sparse([0.0 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0 1.0; 0.0 0.0 0.0 1.0 1.0; 0.0 0.0 0.0 0.0 1.0; 0.0 0.0 0.0 0.0 0.0])
        @test ebn.topology_dict == Dict(:y => 2, :fd => 5, :fd1 => 4, :z => 3, :x => 1)
        @test isa(ebn.nodes[4], DiscreteChildNode)
        @test isa(ebn.nodes[5], DiscreteChildNode)

        interval = (1.10, 1.30)
        root1 = DiscreteRootNode(:A, Dict(:a1 => 0.5, :a2 => 0.5), Dict(:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]))
        root2 = ContinuousRootNode(:B, interval)
        root3 = ContinuousRootNode(:P, Uniform(-10, 10))
        model = Model(df -> df.A .+ df.B .+ df.P, :C)
        sim = DoubleLoop(MonteCarlo(100_000))
        performance = df -> 2 .- df.C
        disc_functional = DiscreteFunctionalNode(:C, [model], performance, sim)

        nodes = [root1, root2, root3, disc_functional]
        ebn = EnhancedBayesianNetwork(nodes)
        add_child!(ebn, root1, disc_functional)
        add_child!(ebn, root2, disc_functional)
        add_child!(ebn, root3, disc_functional)
        order_net!(ebn)
        net2 = deepcopy(ebn)

        evaluate!(ebn)

        @test EnhancedBayesianNetworks._is_imprecise(ebn.nodes[end])

        reduce!(net1)
        @test all(isa.(net1.nodes, DiscreteNode))
        reduce!(net1)
        @test all(isa.(net1.nodes, DiscreteNode))
        reduce!(net2)
        @test all(isa.(net1.nodes, DiscreteNode))
    end

    @testset "Evaluate with envelopes" begin
        using .MathConstants: γ
        Uᵣ = ContinuousRootNode(:Uᵣ, Normal())
        μ_gamma = 60
        cov_gamma = 0.2
        M = DiscreteRootNode(:M, Dict(:new => 0.5, :old => 0.5))
        α, θ = distribution_parameters(μ_gamma, μ_gamma * cov_gamma, Gamma)
        V = ContinuousChildNode(:V, Dict(
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
        R1 = ContinuousFunctionalNode(:R1, [model1], MonteCarlo(10^6))
        R2 = ContinuousFunctionalNode(:R2, [model2], MonteCarlo(10^6))
        R3 = ContinuousFunctionalNode(:R3, [model3], MonteCarlo(10^6))
        R4 = ContinuousFunctionalNode(:R4, [model4], MonteCarlo(10^6))
        R5 = ContinuousFunctionalNode(:R5, [model5], MonteCarlo(10^6))
        function frame_model(r1, r2, r3, r4, r5, v, h)
            g1 = r1 + r2 + r4 + r5 - 5 * h
            g2 = r2 + 2 * r3 + r4 - 5 * v
            g3 = r1 + 2 * r3 + 2 * r4 + r5 - 5 * h - 5 * v
            return minimum([g1, g2, g3])
        end
        model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.r4, df.r5, df.V, df.H), :G)
        performance = df -> df.G
        simulation = MonteCarlo(10^6)

        frame = DiscreteFunctionalNode(:E, [model], performance, simulation)
        L = DiscreteChildNode(:L, Dict(
                [:old] => Dict(:yesL => 0.5, :noL => 0.5),
                [:new] => Dict(:yesL => 0.2, :noL => 0.8),
            ), Dict(:noL => [Parameter(1, :L)], :yesL => [Parameter(2, :L)]))
        r9 = ContinuousRootNode(:R9, Normal())
        model2 = Model(df -> df.L .^ 2 .* df.R9, :P)
        frame2 = DiscreteFunctionalNode(:E2, [model2], df -> df.P, simulation)
        nodes = [Uᵣ, M, V, H, R1, R2, R3, R4, R5, r9, frame, L, frame2]
        ebn = EnhancedBayesianNetwork(nodes)
        add_child!(ebn, M, V)
        add_child!(ebn, Uᵣ, R1)
        add_child!(ebn, Uᵣ, R2)
        add_child!(ebn, Uᵣ, R3)
        add_child!(ebn, Uᵣ, R4)
        add_child!(ebn, Uᵣ, R5)
        add_child!(ebn, R1, frame)
        add_child!(ebn, R2, frame)
        add_child!(ebn, R3, frame)
        add_child!(ebn, R4, frame)
        add_child!(ebn, R5, frame)
        add_child!(ebn, V, frame)
        add_child!(ebn, H, frame)
        add_child!(ebn, M, L)
        add_child!(ebn, L, frame2)
        add_child!(ebn, r9, frame2)
        order_net!(ebn)
        envelopes = markov_envelope(ebn)

        @test issetequal(EnhancedBayesianNetworks._add_root2envelope(ebn, envelopes[2]), [L, frame2, r9, M])
        @test issetequal(EnhancedBayesianNetworks._add_root2envelope(ebn, envelopes[1]), envelopes[1])

        ebn1 = EnhancedBayesianNetworks._build_envelope_edges(ebn, envelopes[1])
        ebn2 = EnhancedBayesianNetworks._build_envelope_edges(ebn, envelopes[2])

        @test ebn1.adj_matrix == sparse([
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
            0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0 1.0 0.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])

        @test ebn1.topology_dict == Dict(:M => 2, :H => 1, :R2 => 6, :R5 => 9, :V => 4, :R4 => 8, :R3 => 7, :R1 => 5, :E => 10, :Uᵣ => 3)
        @test issetequal(ebn1.nodes, [Uᵣ, M, V, H, R1, R2, R3, R4, R5, frame])

        @test ebn2.adj_matrix == sparse([
            0.0 0.0 0.0 1.0;
            0.0 0.0 1.0 0.0;
            0.0 0.0 0.0 1.0;
            0.0 0.0 0.0 0.0])
        @test ebn2.topology_dict == Dict(:R9 => 1, :M => 2, :L => 3, :E2 => 4)
        @test issetequal(ebn2.nodes, [r9, L, frame2, M])

        eebns = evaluate_with_envelopes(ebn)
        @test any(isa.(eebns[1].nodes, FunctionalNode)) == false
        @test any(isa.(eebns[2].nodes, FunctionalNode)) == false
    end

    @testset "No Ancestors case" begin
        root2 = ContinuousRootNode(:B, Normal())
        model = Model(df -> df.B .+ 1, :C)
        sim = MonteCarlo(100_000)
        performance = df -> 2 .- df.C
        cont_functional = ContinuousFunctionalNode(:C, [model], sim)

        ebn = EnhancedBayesianNetwork([root2, cont_functional])
        add_child!(ebn, root2, cont_functional)
        order_net!(ebn)
        evaluate!(ebn)

        @test isa(ebn.nodes[end], ContinuousRootNode)

        disc_functional = DiscreteFunctionalNode(:C, [model], performance, sim)
        ebn = EnhancedBayesianNetwork([root2, disc_functional])
        add_child!(ebn, root2, disc_functional)
        order_net!(ebn)
        evaluate!(ebn)

        @test isa(ebn.nodes[end], DiscreteRootNode)
    end

    @testset "Imprecise Node with discretization" begin
        root1 = DiscreteRootNode(:A, Dict(:y => 0.5, :n => 0.5))
        root2 = ContinuousChildNode(:B, Dict(
                [:y] => UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)]),
                [:n] => UnamedProbabilityBox{Normal}([Interval(1, 2, :μ), Interval(1, 2, :σ)])
            ), ApproximatedDiscretization([-1, 0, 1], 2)
        )
        model = Model(df -> df.B .+ 1, :C)
        sim = DoubleLoop(MonteCarlo(100_000))
        performance = df -> 2 .- df.C
        disc_functional = DiscreteFunctionalNode(:C, [model], performance, sim)
        ebn = EnhancedBayesianNetwork([root1, root2, disc_functional])
        add_child!(ebn, root1, root2)
        add_child!(ebn, root2, disc_functional)
        order_net!(ebn)

        @test_throws ErrorException("node C has as imprecise parents only one or more child nodes with a discretization srtucture defined. They are approximated with Uniform and Exponential assumption and they are no more imprecise. A prices simulation technique must be selected") @suppress evaluate!(ebn)

        root2 = ContinuousRootNode(:B, (-1, 1))
        model = Model(df -> df.B .+ 1, :C)
        sim = MonteCarlo(100_000)
        performance = df -> 2 .- df.C

        disc_functional = DiscreteFunctionalNode(:C, [model], performance, sim)
        ebn = EnhancedBayesianNetwork([root2, disc_functional])
        add_child!(ebn, root2, disc_functional)
        order_net!(ebn)

        @test_throws ErrorException("node C has MonteCarlo(100000) as simulation technique, but have [:B] as imprecise parent/s. DoubleLoop or RandomSlicing technique must be employeed instead") @suppress evaluate!(ebn)
    end
end