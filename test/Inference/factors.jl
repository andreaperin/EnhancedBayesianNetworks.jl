@testset "Factors" begin
    dimensions = [:T, :V]
    potential = [0.95 0.99; 0.05 0.01]
    states_mapping = Dict(:T => Dict(:yesT => 2, :noT => 1), :V => Dict(:yesV => 1, :noV => 2))

    @test_throws ErrorException("Dimensions must be unique") Factor([:V, :V], potential, states_mapping)
    @test_throws ErrorException("potential must have as many dimensions as length of dimensions") Factor([:V], potential, states_mapping)
    @test_throws ErrorException("Having a dimension called potential will cause problems") Factor([:V, :potential], potential, states_mapping)
    @test_throws ErrorException("states mapping keys have to be coherent with defined dimensions") Factor([:V, :L], potential, states_mapping)


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
    bn = BayesianNetwork([v, s, t, l])
    cpd_t = get_cpd(bn, :T)
    ϕ_t = factorize_cpd(cpd_t)

    @test ϕ_t.dimensions == dimensions
    @test ϕ_t.potential == potential
    @test ϕ_t.states_mapping == states_mapping

    evidence = Dict(:T => :yesT)
    @test Factor(bn, :S, evidence).dimensions == [:S]
    @test Factor(bn, :S, evidence).potential == [0.5, 0.5]
    @test Factor(bn, :S, evidence).states_mapping == Dict(:S => Dict(:noS => 1, :yesS => 2))
end
