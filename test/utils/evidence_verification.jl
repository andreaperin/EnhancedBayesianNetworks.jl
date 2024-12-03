@testset "Evidence Verification" begin
    v = DiscreteNode(:V, DataFrame(:V => [:yesV, :noV], :Prob => [0.01, 0.99]))
    s = DiscreteNode(:S, DataFrame(:S => [:yesS, :noS], :Prob => [0.5, 0.5]))
    t = DiscreteNode(:T, DataFrame(:V => [:yesV, :yesV, :noV, :noV], :T => [:yesT, :noT, :yesT, :noT], :Prob => [0.05, 0.95, 0.01, 0.99]))
    l = DiscreteNode(:L, DataFrame(:S => [:yesS, :yesS, :noS, :noS], :L => [:yesL, :noL, :yesL, :noL], :Prob => [0.1, 0.9, 0.01, 0.99]))
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

    b1 = Evidence(:V => :yesV)
    b2 = Evidence(:L => :yesL)

    @test EnhancedBayesianNetworks._are_consistent(a, b1) == true
    @test_throws ErrorException("not consistent evidences: Dict(:L => :noL); Dict(:L => :yesL)") EnhancedBayesianNetworks._are_consistent(a, b2)
end