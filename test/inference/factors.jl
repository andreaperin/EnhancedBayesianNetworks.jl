@testset "Factors" begin
    potential = [0.95 0.99; 0.05 0.01]
    states_mapping = Dict(:T => Dict(:yesT => 2, :noT => 1), :V => Dict(:yesV => 1, :noV => 2))

    @test_throws ErrorException("Dimensions must be unique") Factor([:V, :V], potential, states_mapping)
    @test_throws ErrorException("potential must have as many dimensions as length of dimensions") Factor([:V], potential, states_mapping)
    @test_throws ErrorException("Having a dimension called potential will cause problems") Factor([:V, :potential], potential, states_mapping)
    @test_throws ErrorException("states mapping keys have to be coherent with defined dimensions") Factor([:V, :L], potential, states_mapping)

    v = DiscreteNode(:V, DataFrame(:V => [:yesV, :noV, :maybe], :Prob => [0.01, 0.90, 0.09]))
    s = DiscreteNode(:S, DataFrame(:S => [:yesS, :noS], :Prob => [0.5, 0.5]))
    t = DiscreteNode(:T, DataFrame(:V => [:yesV, :yesV, :noV, :noV, :maybe, :maybe], :T => [:yesT, :noT, :yesT, :noT, :yesT, :noT], :Prob => [0.05, 0.95, 0.01, 0.99, 0.01, 0.99]))
    l = DiscreteNode(:L, DataFrame(:S => [:yesS, :yesS, :yesS, :yesS, :yesS, :yesS, :noS, :noS, :noS, :noS, :noS, :noS], :V => [:yesV, :yesV, :noV, :noV, :maybe, :maybe, :yesV, :yesV, :noV, :noV, :maybe, :maybe], :L => [:yesL, :noL, :yesL, :noL, :yesL, :noL, :yesL, :noL, :yesL, :noL, :yesL, :noL], :Prob => [0.1, 0.9, 0.5, 0.5, 0.2, 0.8, 0.01, 0.99, 0.4, 0.6, 0.3, 0.7]))

    bn = BayesianNetwork([v, s, t, l])
    add_child!(bn, v, t)
    add_child!(bn, s, l)
    add_child!(bn, v, l)
    order!(bn)

    cpd_l = cpd(bn, :L)
    ϕ_l = factorize_cpd(cpd_l)
    # pot = stack([
    #     [0.7 0.6; 0.3 0.4],
    #     [0.5 0.9; 0.5 0.1],
    #     [0.99 0.2; 0.01 0.8]
    # ])

    pot = zeros(2, 3, 2)
    pot[:, :, 1] = [0.7 0.5 0.99; 0.3 0.5 0.01]
    pot[:, :, 2] = [0.6 0.9 0.2; 0.4 0.1 0.8]
    states_mapping = Dict(
        :L => Dict(:noL => 1, :yesL => 2),
        :S => Dict(:noS => 1, :yesS => 2),
        :V => Dict(:maybe => 1, :noV => 3, :yesV => 2)
    )

    @test ϕ_l.dimensions == [:L, :V, :S]
    @test ϕ_l.potential == pot
    @test ϕ_l.states_mapping == states_mapping

    ϕ_l = convert(Factor, cpd_l)
    @test ϕ_l.dimensions == [:L, :V, :S]
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
    @test EnhancedBayesianNetworks._translate_index(ϕ_t, Dict(:T => :yesT)) == inds
end
