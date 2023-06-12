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

end