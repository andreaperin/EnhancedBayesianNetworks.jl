@testset "Inference State" begin
    v = DiscreteNode(:V, DataFrame(:V => [:yesV, :noV], :Prob => [0.01, 0.99]))
    s = DiscreteNode(:S, DataFrame(:S => [:yesS, :noS], :Prob => [0.5, 0.5]))
    t = DiscreteNode(:T, DataFrame(:V => [:yesV, :yesV, :noV, :noV], :T => [:yesT, :noT, :yesT, :noT], :Prob => [0.05, 0.95, 0.01, 0.99]))
    l = DiscreteNode(:L, DataFrame(:S => [:yesS, :yesS, :noS, :noS], :L => [:yesL, :noL, :yesL, :noL], :Prob => [0.1, 0.9, 0.01, 0.99]))
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

    F = DiscreteNode(:F, DataFrame(:F => [:Ft, :Ff], :Prob => [[0.4, 0.5], [0.5, 0.6]]))
    B = DiscreteNode(:B, DataFrame(:B => [:Bt, :Bf], :Prob => [0.5, 0.5]))
    L = DiscreteNode(:L, DataFrame(:F => [:Ft, :Ft, :Ft, :Ff, :Ff, :Ff], :L => [:Lt, :Lf, :L2, :Lt, :Lf, :L2], :Prob => [0.3, 0.4, 0.3, 0.05, 0.85, 0.1]))
    D = DiscreteNode(:D, DataFrame(:F => [:Ft, :Ft, :Ft, :Ft, :Ff, :Ff, :Ff, :Ff], :B => [:Bt, :Bt, :Bf, :Bf, :Bt, :Bt, :Bf, :Bf], :D => [:Dt, :Df, :Dt, :Df, :Dt, :Df, :Dt, :Df], :Prob => [0.8, 0.2, 0.1, 0.9, 0.1, 0.9, 0.7, 0.3]))
    H = DiscreteNode(:H, DataFrame(:D => [:Dt, :Dt, :Df, :Df], :H => [:Ht, :Hf, :Ht, :Hf], :Prob => [0.6, 0.4, 0.3, 0.7]))

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