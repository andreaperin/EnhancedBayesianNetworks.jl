@testset "Inference State" begin

    cpt_v = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:V)
    cpt_v[:V=>:yesV] = 0.01
    cpt_v[:V=>:noV] = 0.99
    v = DiscreteNode(:V, cpt_v)

    cpt_s = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:S)
    cpt_s[:S=>:yesS] = 0.5
    cpt_s[:S=>:noS] = 0.5
    s = DiscreteNode(:S, cpt_s)

    cpt_t = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:V, :T])
    cpt_t[:V=>:yesV, :T=>:yesT] = 0.05
    cpt_t[:V=>:yesV, :T=>:noT] = 0.95
    cpt_t[:V=>:noV, :T=>:yesT] = 0.01
    cpt_t[:V=>:noV, :T=>:noT] = 0.99
    t = DiscreteNode(:T, cpt_t)

    cpt_l = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:S, :L])
    cpt_l[:S=>:yesS, :L=>:yesL] = 0.1
    cpt_l[:S=>:yesS, :L=>:noL] = 0.9
    cpt_l[:S=>:noS, :L=>:yesL] = 0.01
    cpt_l[:S=>:noS, :L=>:noL] = 0.99
    l = DiscreteNode(:L, cpt_l)

    bn = BayesianNetwork([v, s, t, l])
    add_child!(bn, v, t)
    add_child!(bn, s, l)
    order!(bn)

    @test_throws ErrorException("Query O is not in reduced bayesian network") PreciseInferenceState(bn, :O, Evidence())

    @test_throws ErrorException("all nodes in the Dict(:H => :yesH) have to be in the network") PreciseInferenceState(bn, :V, Dict(:H => :yesH))

    @test_throws ErrorException("node states in Dict(:S => :yesH) must be coherent with the one defined in the network") PreciseInferenceState(bn, :V, Dict(:S => :yesH))

    @test_throws ErrorException("Query V is part of the evidence") PreciseInferenceState(bn, :V, Dict(:V => :yesV))

    @test PreciseInferenceState(bn, :V, Evidence()).bn == PreciseInferenceState(bn, [:V], Evidence()).bn
    @test PreciseInferenceState(bn, :V, Evidence()).query == PreciseInferenceState(bn, [:V], Evidence()).query
    @test PreciseInferenceState(bn, :V, Evidence()).evidence == PreciseInferenceState(bn, [:V], Evidence()).evidence

    cpt_f = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}(:F)
    cpt_f[:F=>:Ft] = (0.4, 0.5)
    cpt_f[:F=>:Ff] = (0.5, 0.6)
    F = DiscreteNode(:F, cpt_f)

    cpt_b = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:B)
    cpt_b[:B=>:Bt] = 0.5
    cpt_b[:B=>:Bf] = 0.5
    B = DiscreteNode(:B, cpt_b)

    cpt_l = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:F, :L])
    cpt_l[:F=>:Ft, :L=>:Lt] = 0.3
    cpt_l[:F=>:Ft, :L=>:Lf] = 0.4
    cpt_l[:F=>:Ft, :L=>:L2] = 0.3
    cpt_l[:F=>:Ff, :L=>:Lt] = 0.05
    cpt_l[:F=>:Ff, :L=>:Lf] = 0.85
    cpt_l[:F=>:Ff, :L=>:L2] = 0.1
    L = DiscreteNode(:L, cpt_l)

    cpt_d = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:F, :B, :D])
    cpt_d[:F=>:Ft, :B=>:Bt, :D=>:Dt] = 0.8
    cpt_d[:F=>:Ft, :B=>:Bt, :D=>:Df] = 0.2
    cpt_d[:F=>:Ft, :B=>:Bf, :D=>:Dt] = 0.1
    cpt_d[:F=>:Ft, :B=>:Bf, :D=>:Df] = 0.9
    cpt_d[:F=>:Ff, :B=>:Bt, :D=>:Dt] = 0.1
    cpt_d[:F=>:Ff, :B=>:Bt, :D=>:Df] = 0.9
    cpt_d[:F=>:Ff, :B=>:Bf, :D=>:Dt] = 0.7
    cpt_d[:F=>:Ff, :B=>:Bf, :D=>:Df] = 0.3
    D = DiscreteNode(:D, cpt_d)

    cpt_h = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:D, :H])
    cpt_h[:D=>:Dt, :H=>:Ht] = 0.6
    cpt_h[:D=>:Dt, :H=>:Hf] = 0.4
    cpt_h[:D=>:Df, :H=>:Ht] = 0.3
    cpt_h[:D=>:Df, :H=>:Hf] = 0.7
    H = DiscreteNode(:H, cpt_h)

    cn = CredalNetwork([F, B, L, D, H])

    add_child!(cn, F, L)
    add_child!(cn, F, D)
    add_child!(cn, B, D)
    add_child!(cn, D, H)
    order!(cn)

    a = ImpreciseInferenceState(cn, :D, Evidence())
    @test a.cn == cn
    @test a.evidence == Evidence()
    @test a.query == [:D]
end