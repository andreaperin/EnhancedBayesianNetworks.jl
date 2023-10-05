@testset "Discretization Enhanced Bayesian Network" begin

    root1 = DiscreteRootNode(:x, Dict(:y => 0.2, :n => 0.8), Dict(:y => [Parameter(1, :x)], :n => [Parameter(0, :x), Parameter(5.6, :x1)]))
    root2 = DiscreteRootNode(:y, Dict(:yes => 0.4, :no => 0.6), Dict(:yes => [Parameter(2.2, :y)], :no => [Parameter(5.5, :y)]))
    discretization_root3 = ExactDiscretization([-Inf, 0, Inf])
    root3 = ContinuousRootNode(:z1, Normal(), discretization_root3)

    standard1_name = :α
    standard1_parents = [root1, root2]
    standard1_states = Dict(
        [:y, :yes] => Dict(:a => 0.2, :b => 0.8),
        [:n, :yes] => Dict(:a => 0.3, :b => 0.7),
        [:y, :no] => Dict(:a => 0.4, :b => 0.6),
        [:n, :no] => Dict(:a => 0.5, :b => 0.5)
    )
    standard1_parameters = Dict(:a => [Parameter(3, :α)], :b => [Parameter(10, :α)])
    standard1_node = DiscreteChildNode(standard1_name, standard1_parents, standard1_states, standard1_parameters)

    standard2_name = :β
    standard2_parents = [root1]
    standard2_states = Dict(
        [:y] => Normal(),
        [:n] => Normal(2, 2)
    )
    standard2_states = Dict(
        [:y] => Normal(),
        [:n] => Normal(2, 2)
    )
    discretization_standard2 = ApproximatedDiscretization([-Inf, 0.1, Inf], 1.5)
    standard2_node = ContinuousChildNode(standard2_name, standard2_parents, standard2_states, discretization_standard2)

    functional2_name = :f2
    functional2_parents = [standard1_node, root3]
    functional2_model = Model(df -> (df.α .^ 2 + df.z .^ 2) ./ 2, :value1)
    functional2_simulation = MonteCarlo(800)
    functional2_performance = df -> 1 .- 2 .* df.value1
    functional2_node = DiscreteFunctionalNode(functional2_name, functional2_parents, [functional2_model], functional2_performance, functional2_simulation)

    nodes = [standard1_node, root1, root3, root2, standard2_node, functional2_node]
    ebn = EnhancedBayesianNetwork(nodes)
    disc_ebn = discretize!(ebn)

    name_to_index = Dict(
        :α => 3,
        :β_d => 4,
        :z1_d => 6,
        :y => 1,
        :z1 => 7,
        :f2 => 8,
        :β => 5,
        :x => 2
    )
    fadjlist = [[3], [3, 4], [8], [5], Int64[], [7], [8], Int64[]]
    badjlist = [Int64[], Int64[], [1, 2], [2], [4], Int64[], [6], [3, 7]]
    ne1 = 7

    z_d_node = DiscreteRootNode(:z1_d,
        Dict(
            Symbol("[-Inf, 0.0]") => 0.5,
            Symbol("[0.0, Inf]") => 0.5
        )
    )
    z_c_node = ContinuousChildNode(:z1, [z_d_node],
        Dict(
            [Symbol("[-Inf, 0.0]")] => truncated(Normal(0.0, 1.0), -Inf, 0.0),
            [Symbol("[0.0, Inf]")] => truncated(Normal(0.0, 1.0), 0.0, Inf)
        )
    )

    β_d_node = DiscreteChildNode(:β_d, [root1], Dict(
        [:y] => Dict(
            Symbol("[-Inf, 0.1]") => 0.539827837277029,
            Symbol("[0.1, Inf]") => 0.460172162722971
        ),
        [:n] => Dict(
            Symbol("[-Inf, 0.1]") => 0.17105612630848183,
            Symbol("[0.1, Inf]") => 0.8289438736915182
        )
    ))

    β_c_node = ContinuousChildNode(:β, [β_d_node], Dict(
        [Symbol("[-Inf, 0.1]")] => truncated(Normal(0.1, 1.5), -Inf, 0.1),
        [Symbol("[0.1, Inf]")] => truncated(Normal(0.1, 1.5), 0.1, Inf)
    ))

    functional2_r_node = DiscreteFunctionalNode(functional2_name, [standard1_node, z_c_node], [functional2_model], functional2_performance, functional2_simulation)

    @test all(isequal.(disc_ebn.nodes, [root2, root1, standard1_node, β_d_node, β_c_node, z_d_node, z_c_node, functional2_r_node]))
    @test disc_ebn.dag == SimpleDiGraph{Int64}(ne1, fadjlist, badjlist)
    @test disc_ebn.name_to_index == name_to_index
end

