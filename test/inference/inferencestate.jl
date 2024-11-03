@testset "Inference State" begin
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
    bn = BayesianNetwork([v, s, t, l])
    add_child!(bn, v, t)
    add_child!(bn, s, l)
    order_net!(bn)

    @test_throws ErrorException("Query O is not in reduced bayesian network") PreciseInferenceState(bn, :O, Evidence())

    @test_throws ErrorException("all nodes in the Dict(:H => :yesH) have to be in the network") PreciseInferenceState(bn, :V, Dict(:H => :yesH))

    @test_throws ErrorException("node states in Dict(:S => :yesH) must be coherent with the one defined in the network") PreciseInferenceState(bn, :V, Dict(:S => :yesH))

    @test_throws ErrorException("Query V is part of the evidence") PreciseInferenceState(bn, :V, Dict(:V => :yesV))

    @test PreciseInferenceState(bn, :V, Evidence()).bn == PreciseInferenceState(bn, [:V], Evidence()).bn
    @test PreciseInferenceState(bn, :V, Evidence()).query == PreciseInferenceState(bn, [:V], Evidence()).query
    @test PreciseInferenceState(bn, :V, Evidence()).evidence == PreciseInferenceState(bn, [:V], Evidence()).evidence

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
    order_net!(cn)

    a = ImpreciseInferenceState(cn, :D, Evidence())
    @test a.cn == cn
    @test a.evidence == Evidence()
    @test a.query == [:D]
end