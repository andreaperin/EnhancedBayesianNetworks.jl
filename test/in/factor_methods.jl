@testset "Factors Methods" begin
    dimensions = [:T, :V]
    potential = [0.95 0.99; 0.05 0.01]
    states_mapping = Dict(:T => Dict(:yesT => 2, :noT => 1), :V => Dict(:yesV => 1, :noV => 2))
    ϕ = Factor(dimensions, potential, states_mapping)

    @test size(ϕ) == (2, 2)
    @test size(ϕ, :T) == 2
    @test names(ϕ) == dimensions
    @test in(:V, ϕ) == true
    @test in(:H, ϕ) == false
    @test indexin(:V, ϕ) == 2
    @test indexin([:V, :T], ϕ) == [2, 1]
    @test length(ϕ) == 4

    ϕ_e = ϕ[Dict(:T => :noT)]
    @test ϕ_e.dimensions == [:V]
    @test ϕ_e.potential == [0.95, 0.99]
    @test ϕ_e.states_mapping == Dict(:V => Dict(:yesV => 1, :noV => 2))

    ϕ = Factor(dimensions, potential, states_mapping)

    red = reducedim(+, ϕ, :T)
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

    ϕ = Factor(dimensions, potential, states_mapping)
    reducedim!(+, ϕ, :T)
    @test ϕ.dimensions == [:V]
    @test ϕ.potential == [1.0, 1.0]
    @test ϕ.states_mapping == Dict(:V => Dict(:yesV => 1, :noV => 2))
end