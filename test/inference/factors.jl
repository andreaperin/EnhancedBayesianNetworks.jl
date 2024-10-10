@testset "Factors" begin
    potential = [0.95 0.99; 0.05 0.01]
    states_mapping = Dict(:T => Dict(:yesT => 2, :noT => 1), :V => Dict(:yesV => 1, :noV => 2))

    @test_throws ErrorException("Dimensions must be unique") Factor([:V, :V], potential, states_mapping)
    @test_throws ErrorException("potential must have as many dimensions as length of dimensions") Factor([:V], potential, states_mapping)
    @test_throws ErrorException("Having a dimension called potential will cause problems") Factor([:V, :potential], potential, states_mapping)
    @test_throws ErrorException("states mapping keys have to be coherent with defined dimensions") Factor([:V, :L], potential, states_mapping)

    v = DiscreteRootNode(:V, Dict(:yesV => 0.01, :noV => 0.90, :maybe => 0.09))
    s = DiscreteRootNode(:S, Dict(:yesS => 0.5, :noS => 0.5))
    t = DiscreteChildNode(:T, [v], Dict(
        [:yesV] => Dict(:yesT => 0.05, :noT => 0.95),
        [:noV] => Dict(:yesT => 0.01, :noT => 0.99),
        [:maybe] => Dict(:yesT => 0.01, :noT => 0.99)
    ))
    l = DiscreteChildNode(:L, [s, v], Dict(
        [:yesS, :yesV] => Dict(:yesL => 0.1, :noL => 0.9),
        [:noS, :yesV] => Dict(:yesL => 0.5, :noL => 0.5),
        [:yesS, :noV] => Dict(:noL => 0.2, :yesL => 0.8),
        [:noS, :noV] => Dict(:yesL => 0.01, :noL => 0.99),
        [:yesS, :maybe] => Dict(:yesL => 0.4, :noL => 0.6),
        [:noS, :maybe] => Dict(:yesL => 0.3, :noL => 0.7)
    ))

    bn = BayesianNetwork([v, s, t, l])

    cpd_l = get_cpd(bn, :L)
    ϕ_l = factorize_cpd(cpd_l)
    pot = stack([
        [0.7 0.6; 0.3 0.4],
        [0.5 0.9; 0.5 0.1],
        [0.99 0.2; 0.01 0.8]
    ])
    states_mapping = Dict(
        :L => Dict(:noL => 1, :yesL => 2),
        :S => Dict(:noS => 1, :yesS => 2),
        :V => Dict(:maybe => 1, :noV => 3, :yesV => 2)
    )

    @test ϕ_l.dimensions == [:L, :S, :V]
    @test ϕ_l.potential == pot
    @test ϕ_l.states_mapping == states_mapping

    ϕ_l = convert(Factor, cpd_l)
    @test ϕ_l.dimensions == [:L, :S, :V]
    @test ϕ_l.potential == pot
    @test ϕ_l.states_mapping == states_mapping

    evidence = Dict(:T => :yesT)
    @test Factor(bn, :S, evidence).dimensions == [:S]
    @test Factor(bn, :S, evidence).potential == [0.5, 0.5]
    @test Factor(bn, :S, evidence).states_mapping == Dict(:S => Dict(:noS => 1, :yesS => 2))

    cpd_t = get_cpd(bn, :T)
    ϕ_t = factorize_cpd(cpd_t)
    inds = Array{Any}(undef, length(ϕ_t.dimensions))
    inds[:] .= Colon()
    @test EnhancedBayesianNetworks._translate_index(ϕ_t, Evidence()) == inds
    inds = Array{Any}(undef, length(ϕ_t.dimensions))
    inds[:] .= Colon()
    inds[1] = 2
    EnhancedBayesianNetworks._translate_index(ϕ_t, Dict(:T => :yesT)) == inds
end
