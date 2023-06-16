@testset "Factors Methods" begin
    v = DiscreteRootNode(:V, Dict(:yesV => 0.01, :noV => 0.99))
    s = DiscreteRootNode(:S, Dict(:yesS => 0.5, :noS => 0.5))
    t = DiscreteStandardNode(:T, [v], OrderedDict(
        [:yesV] => Dict(:yesT => 0.05, :noT => 0.95),
        [:noV] => Dict(:yesT => 0.01, :noT => 0.99))
    )

    l = DiscreteStandardNode(:L, [s], OrderedDict(
        [:yesS] => Dict(:yesL => 0.1, :noL => 0.9),
        [:noS] => Dict(:yesL => 0.01, :noL => 0.99))
    )

    b = DiscreteStandardNode(:B, [s], OrderedDict(
        [:yesS] => Dict(:yesB => 0.6, :noB => 0.4),
        [:noS] => Dict(:yesB => 0.3, :noB => 0.7))
    )

    e = DiscreteStandardNode(:E, [t, l], OrderedDict(
        [:yesT, :yesL] => Dict(:yesE => 1, :noE => 0),
        [:yesT, :noL] => Dict(:yesE => 1, :noE => 0),
        [:noT, :yesL] => Dict(:yesE => 1, :noE => 0),
        [:noT, :noL] => Dict(:yesE => 0, :noE => 01))
    )

    d = DiscreteStandardNode(:D, [b, e], OrderedDict(
        [:yesB, :yesE] => Dict(:yesD => 0.9, :noD => 0.1),
        [:yesB, :noE] => Dict(:yesD => 0.8, :noD => 0.2),
        [:noB, :yesE] => Dict(:yesD => 0.7, :noD => 0.3),
        [:noB, :noE] => Dict(:yesD => 0.1, :noD => 0.9))
    )

    x = DiscreteStandardNode(:X, [e], OrderedDict(
        [:yesE] => Dict(:yesX => 0.98, :noX => 0.02),
        [:noE] => Dict(:yesX => 0.05, :noX => 0.95))
    )

    bn = BayesianNetwork([v, s, t, l, b, e, d, x])
    evidence = Evidence()
    factors = map(n -> Factor(bn, n.name, evidence), bn.nodes)
    n = 8
    fadjlist = [[2, 1], [3, 1], [5, 4], [6, 5], [6, 3], [5, 3], [7, 6], [8, 2], [8, 6], [2, 6]]
    moral_graph = EnhancedBayesianNetworks._moral_graph_from_dimensions([x.dimensions for x in factors], bn.name_to_index)

    @test moral_graph == SimpleGraph(n, fadjlist)

    listv = minimal_increase_in_complexity(factors, bn.name_to_index)
    @test Set([x[1] for x in listv[1:3]]) == Set([:X, :D, :V])
    @test listv[4][1] == :S
    @test Set([x[1] for x in listv[5:7]]) == Set([:B, :L, :T])
    @test listv[end][1] == :E

    inf = InferenceState(bn, :B, Dict(:X => :yesX))
    @test all(isapprox.(infer(inf).potential, [0.226 0.284; 0.279 0.2088]; atol=0.01))

    a = DiscreteRootNode(:a, Dict(:yesa => 1.0, :noa => 0.0))
    b = DiscreteRootNode(:b, Dict(:yesb => 0.0, :nob => 1.0))
    c = DiscreteStandardNode(:c, [a, b], OrderedDict(
        [:yesa, :yesb] => Dict(:yesc => 0.1, :noc => 0.9),
        [:yesa, :nob] => Dict(:yesc => 1.0, :noc => 0.0),
        [:noa, :yesb] => Dict(:yesc => 0.2, :noc => 0.8),
        [:noa, :nob] => Dict(:yesc => 0.4, :noc => 0.6))
    )

    bn = BayesianNetwork([a, b, c])
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
    g = DiscreteStandardNode(:G, [d, i], OrderedDict(
        [:yesD, :yesI] => Dict(:firstG => 0.3, :secondG => 0.4, :thirdG => 0.3),
        [:noD, :yesI] => Dict(:firstG => 0.9, :secondG => 0.08, :thirdG => 0.02),
        [:yesD, :noI] => Dict(:firstG => 0.05, :secondG => 0.25, :thirdG => 0.7),
        [:noD, :noI] => Dict(:firstG => 0.5, :secondG => 0.3, :thirdG => 0.2))
    )
    l = DiscreteStandardNode(:L, [g], OrderedDict(
        [:firstG] => Dict(:yesL => 0.1, :noL => 0.9),
        [:secondG] => Dict(:yesL => 0.4, :noL => 0.6),
        [:thirdG] => Dict(:yesL => 0.99, :noL => 0.01))
    )
    s = DiscreteStandardNode(:S, [i], OrderedDict(
        [:yesI] => Dict(:yesS => 0.95, :noS => 0.05),
        [:noI] => Dict(:yesS => 0.2, :noS => 0.8))
    )

    bn = BayesianNetwork([d, i, g, l, s])

    ϕ = infer(bn, :G, Evidence(:D => :yesD, :I => :yesI))
    @test isapprox(ϕ.potential[1], 0.3, atol=0.05)
    @test isapprox(ϕ.potential[2], 0.4, atol=0.05)
    @test isapprox(ϕ.potential[3], 0.3, atol=0.05)
end