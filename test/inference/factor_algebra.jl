@testset "Factors Algebra" begin

    ϕ1 = Factor([:S], [0.5, 0.5], Dict(:S => Dict(:noS => 1, :yesS => 2)))
    ϕ2 = Factor([:B, :S], [0.3 0.6; 0.7 0.4], Dict(:B => Dict(:yesB => 1, :noB => 2), :S => Dict(:noS => 1, :yesS => 2)))

    ϕ12 = ϕ1 * ϕ2
    @test ϕ12.potential[:] == [0.15, 0.35, 0.3, 0.2]
end