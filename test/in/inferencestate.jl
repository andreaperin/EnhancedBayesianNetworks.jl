@testset "Inference State" begin
    v = DiscreteRootNode(:V, Dict(:yesV => 0.01, :noV => 0.99))
    s = DiscreteRootNode(:S, Dict(:yesS => 0.5, :noS => 0.5))
    t = DiscreteChildNode(:T, [v], Dict(
        [:yesV] => Dict(:yesT => 0.05, :noT => 0.95),
        [:noV] => Dict(:yesT => 0.01, :noT => 0.99))
    )
    l = DiscreteChildNode(:L, [s], Dict(
        [:yesS] => Dict(:yesL => 0.1, :noL => 0.9),
        [:noS] => Dict(:yesL => 0.01, :noL => 0.99))
    )
    bn = BayesianNetwork([v, s, t, l])

    @test_throws ErrorException("Query O is not in reduced bayesian network") InferenceState(bn, :O, Evidence())

    @test_throws ErrorException("all nodes in the Dict(:H => :yesH) have to be in the network") InferenceState(bn, :V, Dict(:H => :yesH))

    @test_throws ErrorException("node states in Dict(:S => :yesH) must be coherent with the one defined in the network") InferenceState(bn, :V, Dict(:S => :yesH))

    @test_throws ErrorException("Query V is part of the evidence") InferenceState(bn, :V, Dict(:V => :yesV))

    @test InferenceState(bn, :V, Evidence()).bn == InferenceState(bn, [:V], Evidence()).bn
    @test InferenceState(bn, :V, Evidence()).query == InferenceState(bn, [:V], Evidence()).query
    @test InferenceState(bn, :V, Evidence()).evidence == InferenceState(bn, [:V], Evidence()).evidence


end