@testset "Evidence Verification" begin

    v = DiscreteNode(:V, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:V => [:yesV, :noV], :Π => [0.01, 0.99])))
    s = DiscreteNode(:S, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:S => [:yesS, :noS], :Π => [0.5, 0.5])))
    t = DiscreteNode(:T, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:V => [:yesV, :yesV, :noV, :noV], :T => [:yesT, :noT, :yesT, :noT], :Π => [0.05, 0.95, 0.01, 0.99])))
    l = DiscreteNode(:L, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:S => [:yesS, :yesS, :noS, :noS], :L => [:yesL, :noL, :yesL, :noL], :Π => [0.1, 0.9, 0.01, 0.99])))
    bn = BayesianNetwork([v, s, t, l])
    add_child!(bn, v, t)
    add_child!(bn, s, l)
    order!(bn)

    a1 = Evidence(:O => :yesL)
    a2 = Evidence(:L => :yesO)

    @test_throws ErrorException("all nodes in the Dict(:O => :yesL) have to be in the network") EnhancedBayesianNetworks._verify_evidence(a1, bn)
    @test_throws ErrorException("node states in Dict(:L => :yesO) must be coherent with the one defined in the network") EnhancedBayesianNetworks._verify_evidence(a2, bn)

    a = Evidence(:L => :noL)
    @test isnothing(EnhancedBayesianNetworks._verify_evidence(a, bn))
end