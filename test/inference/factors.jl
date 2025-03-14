@testset "Factors" begin
    @testset "Factor Stucture" begin
        potential = [0.95 0.99; 0.05 0.01]
        states_mapping = Dict(:T => Dict(:yesT => 2, :noT => 1), :V => Dict(:yesV => 1, :noV => 2))

        @test_throws ErrorException("Dimensions must be unique") Factor([:V, :V], potential, states_mapping)
        @test_throws ErrorException("potential must have as many dimensions as length of dimensions") Factor([:V], potential, states_mapping)
        @test_throws ErrorException("Having a dimension called potential will cause problems") Factor([:V, :potential], potential, states_mapping)
        @test_throws ErrorException("states mapping keys have to be coherent with defined dimensions") Factor([:V, :L], potential, states_mapping)

        v = DiscreteNode(:V, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:V => [:yesV, :noV, :maybe], :Π => [0.01, 0.90, 0.09])))
        s = DiscreteNode(:S, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:S => [:yesS, :noS], :Π => [0.5, 0.5])))
        t = DiscreteNode(:T, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:V => [:yesV, :yesV, :noV, :noV, :maybe, :maybe], :T => [:yesT, :noT, :yesT, :noT, :yesT, :noT], :Π => [0.05, 0.95, 0.01, 0.99, 0.01, 0.99])))
        l = DiscreteNode(:L, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:S => [:noS, :noS, :noS, :noS, :noS, :noS, :yesS, :yesS, :yesS, :yesS, :yesS, :yesS], :V => [:maybe, :maybe, :noV, :noV, :yesV, :yesV, :maybe, :maybe, :noV, :noV, :yesV, :yesV], :L => [:noL, :yesL, :noL, :yesL, :noL, :yesL, :noL, :yesL, :noL, :yesL, :noL, :yesL], :Π => [0.7, 0.3, 0.99, 0.01, 0.5, 0.5, 0.6, 0.4, 0.2, 0.8, 0.9, 0.1])))

        bn = BayesianNetwork([v, s, t, l])
        add_child!(bn, v, t)
        add_child!(bn, s, l)
        add_child!(bn, v, l)
        order!(bn)

        ϕ_l = factorize(l.cpt)
        pot = zeros(2, 2, 3)
        pot[:, :, 1] = [0.7 0.6; 0.3 0.4]
        pot[:, :, 2] = [0.99 0.2; 0.01 0.8]
        pot[:, :, 3] = [0.5 0.9; 0.5 0.1]

        states_mapping = Dict(
            :L => Dict(:noL => 1, :yesL => 2),
            :S => Dict(:noS => 1, :yesS => 2),
            :V => Dict(:maybe => 1, :noV => 2, :yesV => 3)
        )

        @test ϕ_l.dimensions == [:L, :S, :V]
        @test ϕ_l.potential == pot
        @test ϕ_l.states_mapping == states_mapping

        ϕ_l = convert(Factor, l.cpt)
        @test ϕ_l.dimensions == [:L, :S, :V]
        @test ϕ_l.potential == pot
        @test ϕ_l.states_mapping == states_mapping

        evidence = Dict(:T => :yesT)
        @test Factor(bn, :S, evidence).dimensions == [:S]
        @test Factor(bn, :S, evidence).potential == [0.5, 0.5]
        @test Factor(bn, :S, evidence).states_mapping == Dict(:S => Dict(:noS => 1, :yesS => 2))
        @test Factor(s, evidence) == Factor(bn, :S, evidence)

        cpd_t = t.cpt
        ϕ_t = factorize(cpd_t)
        inds = Array{Any}(undef, length(ϕ_t.dimensions))
        inds[:] .= Colon()
        @test EnhancedBayesianNetworks._translate_index(ϕ_t, Evidence()) == inds
        inds = Array{Any}(undef, length(ϕ_t.dimensions))
        inds[:] .= Colon()
        inds[1] = 2
        @test EnhancedBayesianNetworks._translate_index(ϕ_t, Dict(:T => :yesT)) == inds
    end
    @testset "Factors Methods" begin
        dimensions = [:T, :V]
        potential = [0.95 0.99; 0.05 0.01]
        states_mapping = Dict(:T => Dict(:yesT => 2, :noT => 1), :V => Dict(:yesV => 1, :noV => 2))
        ϕ = Factor(dimensions, potential, states_mapping)

        @test size(ϕ) == (2, 2)
        @test size(ϕ, :T) == 2
        @test names(ϕ) == dimensions
        @test ∈(:V, ϕ) == true
        @test ∈(:H, ϕ) == false
        @test indexin(:V, ϕ) == 2
        @test indexin([:V, :T], ϕ) == [2, 1]
        @test length(ϕ) == 4

        ϕ_e = ϕ[Dict(:T => :noT)]
        @test ϕ_e.dimensions == [:V]
        @test ϕ_e.potential == [0.95, 0.99]
        @test ϕ_e.states_mapping == Dict(:V => Dict(:yesV => 1, :noV => 2))

        ϕ = Factor(dimensions, potential, states_mapping)

        red = EnhancedBayesianNetworks._reducedim(+, ϕ, :T)
        @test red.dimensions == [:V]
        @test red.potential == [1.0, 1.0]
        @test red.states_mapping == Dict(:V => Dict(:yesV => 1, :noV => 2))

        red = sum(ϕ, :T)
        @test red.dimensions == [:V]
        @test red.potential == [1.0, 1.0]
        @test red.states_mapping == Dict(:V => Dict(:yesV => 1, :noV => 2))

        res = permutedims(ϕ, [2, 1])
        @test res.dimensions == [:V, :T]
        @test res.potential == [0.95 0.05; 0.99 0.01]
        @test res.states_mapping == Dict(:V => Dict(:yesV => 1, :noV => 2), :T => Dict(:yesT => 2, :noT => 1))

        ϕ = Factor([:X, :Y], [1.0 2.0; 3.0 4.0; 5.0 6.0], Dict(:X => Dict(:yesx => 2, :nox => 1), :Y => Dict(:yesy => 1, :noy => 2)))
        @test all(isapprox.(
            broadcast(*, ϕ, [:Y, :X], [[10, 0.1], 100.0]).potential,
            Float64[1000 20; 3000 40; 5000 60]))

        @test_throws ErrorException("Dimension is not in the factor") broadcast(*, ϕ, [:X, :Z], [[10, 1, 0.1], [1, 2, 3]])
        @test_throws DimensionMismatch broadcast(*, ϕ, :X, [2016, 58.0])
        @test_throws TypeError(:broadcast!, "Invalid broadcast value", Union{Float64,Vector{Float64}}, [2016]) broadcast(*, ϕ, :X, [2016])

        ϕ = Factor(dimensions, potential, states_mapping)
        EnhancedBayesianNetworks._reducedim!(+, ϕ, :T)
        @test ϕ.dimensions == [:V]
        @test ϕ.potential == [1.0, 1.0]
        @test ϕ.states_mapping == Dict(:V => Dict(:yesV => 1, :noV => 2))
    end
    @testset "Factors Algebra" begin
        ϕ1 = Factor([:S], [0.5, 0.5, 0.7], Dict(:S => Dict(:noS => 1, :yesS => 2)))
        ϕ2 = Factor([:B, :S], [0.3 0.6; 0.7 0.4], Dict(:B => Dict(:yesB => 1, :noB => 2), :S => Dict(:noS => 1, :yesS => 2)))

        @test_throws ErrorException("Common dimensions must have same size") ϕ1 * ϕ2

        ϕ1 = Factor([:S], [0.5, 0.5], Dict(:S => Dict(:noS => 1, :yesS => 2)))
        ϕ12 = ϕ1 * ϕ2
        @test ϕ12.potential[:] == [0.15, 0.35, 0.3, 0.2]
    end
end
