@testset "Variable Elimination" begin
    v = DiscreteRootNode(:V, Dict(:yesV => 0.01, :noV => 0.99))
    s = DiscreteRootNode(:S, Dict(:yesS => 0.5, :noS => 0.5))
    t = DiscreteChildNode(:T, Dict(
        [:yesV] => Dict(:yesT => 0.05, :noT => 0.95),
        [:noV] => Dict(:yesT => 0.01, :noT => 0.99))
    )

    l = DiscreteChildNode(:L, Dict(
        [:yesS] => Dict(:yesL => 0.1, :noL => 0.9),
        [:noS] => Dict(:yesL => 0.01, :noL => 0.99))
    )

    b = DiscreteChildNode(:B, Dict(
        [:yesS] => Dict(:yesB => 0.6, :noB => 0.4),
        [:noS] => Dict(:yesB => 0.3, :noB => 0.7))
    )

    e = DiscreteChildNode(:E, Dict(
        [:yesT, :yesL] => Dict(:yesE => 1, :noE => 0),
        [:yesT, :noL] => Dict(:yesE => 1, :noE => 0),
        [:noT, :yesL] => Dict(:yesE => 1, :noE => 0),
        [:noT, :noL] => Dict(:yesE => 0, :noE => 01))
    )

    d = DiscreteChildNode(:D, Dict(
        [:yesB, :yesE] => Dict(:yesD => 0.9, :noD => 0.1),
        [:yesB, :noE] => Dict(:yesD => 0.8, :noD => 0.2),
        [:noB, :yesE] => Dict(:yesD => 0.7, :noD => 0.3),
        [:noB, :noE] => Dict(:yesD => 0.1, :noD => 0.9))
    )

    x = DiscreteChildNode(:X, Dict(
        [:yesE] => Dict(:yesX => 0.98, :noX => 0.02),
        [:noE] => Dict(:yesX => 0.05, :noX => 0.95))
    )

    bn = BayesianNetwork([v, s, t, l, b, e, d, x])
    add_child!(bn, v, t)
    add_child!(bn, s, l)
    add_child!(bn, s, b)
    add_child!(bn, t, e)
    add_child!(bn, l, e)
    add_child!(bn, b, d)
    add_child!(bn, e, d)
    add_child!(bn, e, x)
    order!(bn)
    evidence = Evidence()
    factors = map(n -> Factor(bn, n.name, evidence), bn.nodes)
    dimensions = map(f -> f.dimensions, factors)

    adj_moral = EnhancedBayesianNetworks._structure_adj_matrix(dimensions, bn.topology_dict)

    @test adj_moral == sparse([
        0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 1.0 1.0 0.0 0.0 0.0;
        1.0 0.0 0.0 1.0 0.0 1.0 0.0 0.0;
        0.0 1.0 1.0 0.0 0.0 1.0 0.0 0.0;
        0.0 1.0 0.0 0.0 0.0 1.0 1.0 0.0;
        0.0 0.0 1.0 1.0 1.0 0.0 1.0 1.0;
        0.0 0.0 0.0 0.0 1.0 1.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0
    ])
    @test map(x -> EnhancedBayesianNetworks._n_eliminated_edges(dimensions, bn.topology_dict, x), collect(values(bn.topology_dict))) == [3, 2, 3, 1, 2, 3, 1, 5]
    @test map(x -> EnhancedBayesianNetworks._n_added_edges(dimensions, bn.topology_dict, x), collect(values(bn.topology_dict))) == [2, 0, 2, 0, 1, 2, 0, 8]

    listv = EnhancedBayesianNetworks._order_with_minimal_increase_in_complexity(factors, bn.topology_dict)
    @test listv == [(:D, 0.0), (:V, 0.0), (:X, 0.0), (:S, 0.5), (:T, 0.6666666666666666), (:L, 0.6666666666666666), (:B, 0.6666666666666666), (:E, 1.6)]

    @testset "Inference Precise" begin
        inf = PreciseInferenceState(bn, :B, Dict(:X => :yesX))
        res = infer(inf)

        @test res.dimensions == [:B]
        @test isapprox(res.potential, [0.5063261560155387, 0.49367384398446135])
        @test res.states_mapping == Dict(:B => Dict(:yesB => 1, :noB => 2))

        a = DiscreteRootNode(:a, Dict(:yesa => 1.0, :noa => 0.0))
        b = DiscreteRootNode(:b, Dict(:yesb => 0.0, :nob => 1.0))
        c = DiscreteChildNode(:c, Dict(
            [:yesa, :yesb] => Dict(:yesc => 0.1, :noc => 0.9),
            [:yesa, :nob] => Dict(:yesc => 1.0, :noc => 0.0),
            [:noa, :yesb] => Dict(:yesc => 0.2, :noc => 0.8),
            [:noa, :nob] => Dict(:yesc => 0.4, :noc => 0.6))
        )

        bn = BayesianNetwork([a, b, c])
        add_child!(bn, a, c)
        add_child!(bn, b, c)
        order!(bn)

        ϕ = infer(bn, :a)
        @test length(ϕ) == 2
        f = ϕ[:a=>:yesa]::Factor
        @test isapprox(f.potential[1], 1.0, atol=0.02)
        f = ϕ[:a=>:noa]::Factor
        @test isapprox(f.potential[1], 0.0, atol=0.02)

        ϕ = infer(bn, :c)::Factor
        @test isapprox(ϕ[:c=>:yesc].potential[1], 1.0, atol=0.02)
        @test isapprox(ϕ[:c=>:noc].potential[1], 0.0, atol=0.02)

        ϕ = infer(bn, [:b, :c])

        @test size(ϕ) == (2, 2)
        @test isapprox(ϕ[:b=>:yesb, :c=>:yesc].potential[1], 0.0, atol=0.02)
        @test isapprox(ϕ[:b=>:nob, :c=>:yesc].potential[1], 1.0, atol=0.02)
        @test isapprox(ϕ[:b=>:yesb, :c=>:noc].potential[1], 0.0, atol=0.02)
        @test isapprox(ϕ[:b=>:nob, :c=>:noc].potential[1], 0.0, atol=0.02)

        d = DiscreteRootNode(:D, Dict(:yesD => 0.6, :noD => 0.4))
        i = DiscreteRootNode(:I, Dict(:yesI => 0.7, :noI => 0.3))
        g = DiscreteChildNode(:G, Dict(
            [:yesD, :yesI] => Dict(:firstG => 0.3, :secondG => 0.4, :thirdG => 0.3),
            [:noD, :yesI] => Dict(:firstG => 0.9, :secondG => 0.08, :thirdG => 0.02),
            [:yesD, :noI] => Dict(:firstG => 0.05, :secondG => 0.25, :thirdG => 0.7),
            [:noD, :noI] => Dict(:firstG => 0.5, :secondG => 0.3, :thirdG => 0.2))
        )
        l = DiscreteChildNode(:L, Dict(
            [:firstG] => Dict(:yesL => 0.1, :noL => 0.9),
            [:secondG] => Dict(:yesL => 0.4, :noL => 0.6),
            [:thirdG] => Dict(:yesL => 0.99, :noL => 0.01))
        )
        s = DiscreteChildNode(:S, Dict(
            [:yesI] => Dict(:yesS => 0.95, :noS => 0.05),
            [:noI] => Dict(:yesS => 0.2, :noS => 0.8))
        )

        bn = BayesianNetwork([d, i, g, l, s])
        add_child!(bn, d, g)
        add_child!(bn, i, g)
        add_child!(bn, g, l)
        add_child!(bn, i, s)
        order!(bn)

        inf = PreciseInferenceState(bn, [:G], Evidence(:D => :yesD, :I => :yesI))
        ϕ = infer(bn, :G, Evidence(:D => :yesD, :I => :yesI))
        @test isapprox(ϕ.potential[1], 0.3, atol=0.05)
        @test isapprox(ϕ.potential[2], 0.4, atol=0.05)
        @test isapprox(ϕ.potential[3], 0.3, atol=0.05)
    end

    @testset "Inference Imprecise" begin
        F = DiscreteRootNode(:F, Dict(:Ft => [0.4, 0.5], :Ff => [0.5, 0.6]))
        B = DiscreteRootNode(:B, Dict(:Bt => 0.5, :Bf => 0.5))
        L = DiscreteChildNode(:L, Dict(
            [:Ft] => Dict(:Lt => 0.3, :Lf => 0.4, :L2 => 0.3),
            [:Ff] => Dict(:Lt => 0.05, :Lf => 0.85, :L2 => 0.1)
        ))
        D = DiscreteChildNode(:D, Dict(
            [:Ft, :Bt] => Dict(:Dt => 0.8, :Df => 0.2),
            [:Ft, :Bf] => Dict(:Dt => 0.1, :Df => 0.9),
            [:Ff, :Bt] => Dict(:Dt => 0.1, :Df => 0.9),
            [:Ff, :Bf] => Dict(:Dt => 0.7, :Df => 0.3)
        ))
        H = DiscreteChildNode(:H, Dict(
            [:Dt] => Dict(:Ht => 0.6, :Hf => 0.4),
            [:Df] => Dict(:Ht => 0.3, :Hf => 0.7)
        ))
        cn = CredalNetwork([F, B, L, D, H])
        add_child!(cn, F, L)
        add_child!(cn, F, D)
        add_child!(cn, B, D)
        add_child!(cn, D, H)
        order!(cn)

        evidence = Dict(
            :D => :Dt,
        )
        query = [:L, :F]

        inference_state = ImpreciseInferenceState(cn, query, evidence)
        ϕ = infer(inference_state)
        mat = hcat(
            [[0.128571, 0.158824], [0.171429, 0.211765], [0.128571, 0.158824]],
            [[0.0235294, 0.0285714], [0.4, 0.485714], [0.0470588, 0.0571429]])

        @test isequal(ϕ.dimensions, [:L, :F])
        @test isequal(ϕ.states_mapping, Dict(
            :F => Dict(:Ft => 1, :Ff => 2),
            :L => Dict(:Lt => 1, :Lf => 2, :L2 => 3)))
        @test isapprox(ϕ.potential, mat, atol=0.01)

        ϕ1 = infer(cn, [:L, :F], evidence)
        @test isequal(ϕ.dimensions, ϕ1.dimensions)
        @test isequal(ϕ.states_mapping, ϕ1.states_mapping)
        @test isapprox(ϕ.potential, ϕ1.potential)
    end
    # @testset "Straub Example" begin
    #     using .MathConstants: γ

    #     n = 10^6
    #     Uᵣ = ContinuousRootNode(:Uᵣ, Normal())
    #     μ_gamma = 60
    #     cov_gamma = 0.2
    #     α, θ = distribution_parameters(μ_gamma, μ_gamma * cov_gamma, Gamma)
    #     V = ContinuousRootNode(:V, Gamma(α, θ))

    #     μ_gumbel = 50
    #     cov_gumbel = 0.4
    #     μ_loc, β = distribution_parameters(μ_gumbel, cov_gumbel * μ_gumbel, Gumbel)
    #     H = ContinuousRootNode(:H, Gumbel(μ_loc, β))

    #     function plastic_moment_capacities(uᵣ)
    #         ρ = 0.5477
    #         μ = 150
    #         cov = 0.2

    #         λ, ζ = distribution_parameters(μ, μ * cov, LogNormal)

    #         normal_μ = λ + ρ * ζ * uᵣ
    #         normal_std = sqrt((1 - ρ^2) * ζ^2)
    #         exp(rand(Normal(normal_μ, normal_std)))
    #     end

    #     model1 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r1)
    #     model2 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r2)
    #     model3 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r3)
    #     model4 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r4)
    #     model5 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r5)

    #     function frame_model(r1, r2, r3, r4, r5, v, h)
    #         g1 = r1 + r2 + r4 + r5 - 5 * h
    #         g2 = r2 + 2 * r3 + r4 - 5 * v
    #         g3 = r1 + 2 * r3 + 2 * r4 + r5 - 5 * h - 5 * v
    #         return minimum([g1, g2, g3])
    #     end

    #     R1 = ContinuousFunctionalNode(:R1, [model1], MonteCarlo(n))
    #     R2 = ContinuousFunctionalNode(:R2, [model2], MonteCarlo(n))
    #     R3 = ContinuousFunctionalNode(:R3, [model3], MonteCarlo(n))

    #     @testset "No Evidence" begin
    #         R4 = ContinuousFunctionalNode(:R4, [model4], MonteCarlo(n))
    #         R5 = ContinuousFunctionalNode(:R5, [model5], MonteCarlo(n))

    #         model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.r4, df.r5, df.V, df.H), :G)
    #         performance = df -> df.G
    #         simulation = MonteCarlo(n)
    #         frame = DiscreteFunctionalNode(:E, [model], performance, simulation)

    #         nodes = [Uᵣ, V, H, R1, R2, R3, R4, R5, frame]

    #         net = EnhancedBayesianNetwork(nodes)

    #         add_child!(net, Uᵣ, R1)
    #         add_child!(net, Uᵣ, R2)
    #         add_child!(net, Uᵣ, R3)
    #         add_child!(net, Uᵣ, R4)
    #         add_child!(net, Uᵣ, R5)
    #         add_child!(net, R1, frame)
    #         add_child!(net, R2, frame)
    #         add_child!(net, R3, frame)
    #         add_child!(net, R4, frame)
    #         add_child!(net, R5, frame)
    #         add_child!(net, V, frame)
    #         add_child!(net, H, frame)
    #         order!(net)
    #         evaluate!(net)

    #         @test isapprox(net.nodes[end].states[:safe_E], 0.973871; atol=0.01)
    #         @test isapprox(net.nodes[end].states[:fail_E], 0.026129; atol=0.01)
    #     end

    #     @testset "Evidence" begin
    #         n2 = 2000
    #         discretization1 = ApproximatedDiscretization(collect(range(50, 250, 21)), 1)
    #         discretization2 = ApproximatedDiscretization(collect(range(50.01, 250.01, 21)), 1)
    #         R4 = ContinuousFunctionalNode(:R4, [model4], MonteCarlo(n2), discretization1)
    #         R5 = ContinuousFunctionalNode(:R5, [model5], MonteCarlo(n2), discretization2)

    #         model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.R4, df.R5, df.V, df.H), :G)
    #         performance = df -> df.G
    #         simulation = MonteCarlo(n2)
    #         frame = DiscreteFunctionalNode(:E, [model], performance, simulation)

    #         nodes = [Uᵣ, V, H, R1, R2, R3, R4, R5, frame]

    #         net = EnhancedBayesianNetwork(nodes)

    #         add_child!(net, Uᵣ, R1)
    #         add_child!(net, Uᵣ, R2)
    #         add_child!(net, Uᵣ, R3)
    #         add_child!(net, Uᵣ, R4)
    #         add_child!(net, Uᵣ, R5)
    #         add_child!(net, R1, frame)
    #         add_child!(net, R2, frame)
    #         add_child!(net, R3, frame)
    #         add_child!(net, R4, frame)
    #         add_child!(net, R5, frame)
    #         add_child!(net, V, frame)
    #         add_child!(net, H, frame)
    #         order!(net)

    #         @suppress evaluate!(net)

    #         @test net.adj_matrix == sparse([
    #             0.0 0.0 0.0 1.0 1.0 0.0 0.0 1.0;
    #             0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
    #             0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
    #             0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0;
    #             0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
    #             0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
    #             0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
    #             0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
    #         ])
    #         @test net.topology_dict == Dict(:R5_d => 5,
    #             :R4_d => 4,
    #             :H => 3,
    #             :R5 => 7,
    #             :V => 2,
    #             :R4 => 6,
    #             :E => 8,
    #             :Uᵣ => 1)

    #         EnhancedBayesianNetworks._eliminate_continuous_node!(net, net.nodes[7])
    #         @test net.adj_matrix == sparse([
    #             0.0 0.0 0.0 1.0 1.0 0.0 1.0;
    #             0.0 0.0 0.0 0.0 0.0 0.0 1.0;
    #             0.0 0.0 0.0 0.0 0.0 0.0 1.0;
    #             0.0 0.0 0.0 0.0 0.0 1.0 0.0;
    #             0.0 0.0 0.0 0.0 0.0 0.0 1.0;
    #             0.0 0.0 0.0 0.0 0.0 0.0 1.0;
    #             0.0 0.0 0.0 0.0 0.0 0.0 0.0])

    #         reduce!(net)
    #         @test net.adj_matrix == sparse([
    #             0.0 0.0 1.0;
    #             0.0 0.0 1.0;
    #             0.0 0.0 0.0])

    #         bn = BayesianNetwork(net)

    #         evidence2 = Dict(
    #             :R4_d => Symbol([140.0, 150.0]),
    #             :R5_d => Symbol([90.01, 100.01])
    #         )
    #         # ϕ2 = infer(bn, :E, evidence2)

    #         # @test all(isapprox.(ϕ2.potential, [0.965, 0.035], atol=0.05))
    #     end
    # end
end