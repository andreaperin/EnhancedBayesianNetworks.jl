@testset "Evidence Verification" begin
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

    a1 = Evidence(:O => :yesL)
    a2 = Evidence(:L => :yesO)

    @test_throws ErrorException("all nodes in the Dict(:O => :yesL) have to be in the network") EnhancedBayesianNetworks.verify_evidence(a1, bn)
    @test_throws ErrorException("node states in Dict(:L => :yesO) must be coherent with the one defined in the network") EnhancedBayesianNetworks.verify_evidence(a2, bn)

    a = Evidence(:L => :noL)
    @test isnothing(EnhancedBayesianNetworks.verify_evidence(a, bn))

    b1 = Evidence(:V => :yesV)
    b2 = Evidence(:L => :yesL)

    @test EnhancedBayesianNetworks.consistent(a, b1) == true
    @test_throws ErrorException("not consistent evidences: Dict(:L => :noL); Dict(:L => :yesL)") EnhancedBayesianNetworks.consistent(a, b2)
end