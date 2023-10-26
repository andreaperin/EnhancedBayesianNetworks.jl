@testset "General Nodes Operation" begin
    root1 = DiscreteRootNode(:X1, Dict(:y => 0.2, :n => 0.8))
    root2 = DiscreteRootNode(:X2, Dict(:yes => 0.4, :no => 0.6), Dict(:yes => [Parameter(2.2, :X2)], :no => [Parameter(5.5, :X2)]))
    root3 = ContinuousRootNode(:Y1, Normal(), ExactDiscretization([0, 0.2, 1]))

    child1_states = Dict(
        [:y] => Dict(:c1y => 0.3, :c1n => 0.7),
        [:n] => Dict(:c1y => 0.4, :c1n => 0.6),)
    child1 = DiscreteChildNode(:C1, [root1, root3], child1_states, Dict(:c1y => [Parameter(1, :X1)], :c1n => [Parameter(0, :X1)]))

    child2 = ContinuousChildNode(:C2, [root2], Dict([:yes] => Normal(), [:no] => Normal(1, 1)))

    functional1_parents = [child1, child2]
    disc_D = ApproximatedDiscretization([-1.1, 0, 0.11], 2)
    model1 = [Model(df -> (df.X1 .^ 2) ./ 2 .- df.C2, :fun1)]
    simulation1 = MonteCarlo(300)
    functional1_node = ContinuousFunctionalNode(:F1, functional1_parents, model1, simulation1, disc_D)
    @testset "Discrete Ancestors" begin
        @test issetequal([root2, child1], discrete_ancestors(functional1_node))
    end
    @testset "States Combinations" begin
        @test issetequal([[:c1y, :yes], [:c1n, :yes], [:c1y, :no], [:c1n, :no]], state_combinations(functional1_node))
    end

end