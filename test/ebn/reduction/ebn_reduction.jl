@testset "eBN reduction" begin

    root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
    root2 = DiscreteRootNode(:z, Dict(:y => 0.2, :n => 0.8), Dict(:y => [Parameter(1, :z)], :n => [Parameter(0, :z)]))

    states_child1 = Dict([:yes] => Dict(:a => 0.5, :b => 0.5), [:no] => Dict(:a => 0.5, :b => 0.5))
    child1 = DiscreteChildNode(:child1, [root1], states_child1, Dict(:a => [Parameter(1, :child1)], :b => [Parameter(0, :child1)]))

    distributions_child2 = Dict([:a] => Normal(), [:b] => Normal(2, 2))
    child2 = ContinuousChildNode(:child2, [child1], distributions_child2)

    model = Model(df -> sqrt.(df.child1 .^ 2 + df.child2 .- df.z .^ 2), :value1)
    simulation = MonteCarlo(400)
    performance = df -> 1 .- 2 .* df.value1
    functional = DiscreteFunctionalNode(:functional, [child1, child2, root2], [model], performance, simulation)

    ebn = EnhancedBayesianNetwork([root1, root2, child1, child2, functional])

    functional_n = DiscreteFunctionalNode(:functional, [root2, child1], [model], performance, simulation)

    badjlist = Vector{Vector{Int}}([[], [1], [], [2, 3]])
    fadjlist = Vector{Vector{Int}}([[2], [4], [4], []])
    resulting_dag = DiGraph(3, fadjlist, badjlist)

    @test issetequal(EnhancedBayesianNetworks._reduce_node(ebn, child2), [root1, root2, child1, functional_n])

    rbn = reduce(ebn)
    @test rbn.dag == resulting_dag
end