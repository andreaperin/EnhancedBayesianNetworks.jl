@testset "CPD" begin
    target = :A
    parents = [:B, :C]
    parents_states_mapping_dict = Dict(
        :B => Dict(:b1 => 1, :b2 => 2),
        :C => Dict(:c1 => 1, :c2 => 2)
    )
    parental_ncategories = [2, 2]
    states = [:a1, :a2]
    distributions = Dict(
        [:b1, :c1] => Dict(:a1 => 0.5, :a2 => 0.5),
        [:b1, :c2] => Dict(:a1 => 0.5, :a2 => 0.5),
        [:b2, :c1] => Dict(:a1 => 0.5, :a2 => 0.5),
        [:b2, :c2] => Dict(:a1 => 0.5, :a2 => 0.5)
    )

    cpd = ConditionalProbabilityDistribution(target, parents, parents_states_mapping_dict, parental_ncategories, states, distributions)

    @test cpd.target == target
    @test cpd.parents == parents
    @test cpd.parents_states_mapping_dict == parents_states_mapping_dict
    @test cpd.parental_ncategories == parental_ncategories
    @test cpd.states == states
    @test cpd.distributions == distributions

end
