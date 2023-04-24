@testset "Enhanced Bayesian Networks" begin
    @testset "DiGraphFunctions" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
        root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6))
        root3 = ContinuousRootNode(RandomVariable(Normal(), :z))
        name = :child
        parents = [root1, root2, root3]
        distribution = OrderedDict(
            [:yes, :yes] => Normal(),
            [:no, :yes] => Normal(1, 1),
            [:yes, :no] => Normal(2, 1),
            [:no, :no] => Normal(3, 1)
        )
        child_node = ContinuousStandardNode(name, parents, distribution)
        nodes = [root1, root2, root3, child_node]

        @test EnhancedBayesianNetworks._build_digraph(nodes) == SimpleDiGraph{Int64}(3, [[4], [4], [4], Int64[]], [Int64[], Int64[], Int64[], [1, 2, 3]])

        @test EnhancedBayesianNetworks._topological_ordered_dag(nodes)[1] == SimpleDiGraph{Int64}(3, [[4], [4], [4], Int64[]], [Int64[], Int64[], Int64[], [1, 2, 3]])

        @test EnhancedBayesianNetworks._topological_ordered_dag(nodes)[2] == [root3, root2, root1, child_node]

        @test EnhancedBayesianNetworks._topological_ordered_dag(nodes)[3] == Dict(:z => 1, :y => 2, :x => 3, :child => 4)
    end

    @testset "EnhancedBayesianNetwork" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))

        states_child1 = OrderedDict(
            [:yes] => Dict(:a => 0.5, :b => 0.5),
            [:no] => Dict(:a => 0.5, :b => 0.5)
        )
        child1 = DiscreteStandardNode(:child1, [root1], states_child1)

        distributions_child2 = OrderedDict(
            [:a] => Normal(),
            [:b] => Normal(2, 2)
        )
        child2 = ContinuousStandardNode(:child2, [child1], distributions_child2)

        model = Model(df -> sqrt.(df.child1 .^ 2 + df.child2 .^ 2), :value1)
        performance = Model(df -> 1 .- 2 .* df.value1, :value2)
        models = OrderedDict(
            [:a] => [model, performance],
            [:b] => [model, performance]
        )

        functional = DiscreteFunctionalNode(:functional, [child1, child2], models)

        badjlist = Vector{Vector{Int}}([[], [1], [2], [2, 3]])
        fadjlist = Vector{Vector{Int}}([[2], [3, 4], [4], []])

        @test EnhancedBayesianNetwork([root1, child1, child2, functional]).dag == EnhancedBayesianNetwork(DiGraph(4, fadjlist, badjlist), [root1, child1, child2, functional], Dict(:x => 1, :child1 => 2, :child2 => 3, :functional => 4)).dag

        @test EnhancedBayesianNetwork([root1, child1, child2, functional]).nodes == EnhancedBayesianNetwork(DiGraph(4, fadjlist, badjlist), [root1, child1, child2, functional], Dict(:x => 1, :child1 => 2, :child2 => 3, :functional => 4)).nodes

        @test EnhancedBayesianNetwork([root1, child1, child2, functional]).name_to_index == EnhancedBayesianNetwork(DiGraph(4, fadjlist, badjlist), [root1, child1, child2, functional], Dict(:x => 1, :child1 => 2, :child2 => 3, :functional => 4)).name_to_index
    end
end
