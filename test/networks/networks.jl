# @testset "CPD" begin
#     target = :A
#     parents = [:B, :C]
#     parental_ncategories = [2, 2]
#     states = [:a1, :a2]
#     distribution = DataFrame(:B => [:b1, :b1, :b1, :b1, :b2, :b2, :b2, :b2], :C => [:c1, :c1, :c2, :c2, :c1, :c1, :c2, :c2], :Prob => [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

#     cpd = ConditionalProbabilityDistribution(target, parents, parental_ncategories, states, distribution)

#     @test cpd.target == target
#     @test cpd.parents == parents
#     @test cpd.parental_ncategories == parental_ncategories
#     @test cpd.states == states
#     @test cpd.probabilities == distribution
# end