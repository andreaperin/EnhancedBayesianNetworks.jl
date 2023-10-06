@testset "Evaluate Structural Reliability Problem" begin
    root1 = DiscreteRootNode(:X1, Dict(:y => 0.2, :n => 0.8), Dict(:y => [Parameter(1, :X1)], :n => [Parameter(0, :X1)]))
    root2 = DiscreteRootNode(:X2, Dict(:yes => 0.4, :no => 0.6), Dict(:yes => [Parameter(2.2, :X2)], :no => [Parameter(5.5, :X2)]))
    root3 = ContinuousRootNode(:Y1, Uniform(-1, 1))
    root4 = ContinuousRootNode(:Y2, Normal())

    functional1_parents = [root1, root3]
    model1 = [Model(df -> df.Y1 .+ df.X1, :fun1)]
    simulation1 = MonteCarlo(1000000)
    functional1_node = ContinuousFunctionalNode(:F1, functional1_parents, model1, simulation1)

    standard3_parents = [root1]
    standard3_states = Dict(
        [:y] => Normal(),
        [:n] => Normal(2, 2)
    )
    standard3_node = ContinuousChildNode(:Std3, standard3_parents, standard3_states)

    functional2_name = :F2
    functional2_parents = [root2, root3, root4]
    model2 = [Model(df -> (df.X2 .^ 2 .+ df.Y1 .^ 2) ./ 2 .- df.Y1 .- df.Y2, :fun2)]
    performance = df -> 1 .- 2 .* df.fun2
    simulation2 = MonteCarlo(100000)
    functional2_node = DiscreteFunctionalNode(functional2_name, functional2_parents, model2, performance, simulation2)

    continuous_srpnode1 = EnhancedBayesianNetworks._build_structuralreliabilityproblem_node(functional1_node)
    evaluated_node1 = evaluate!(continuous_srpnode1)

    discrete_srpnode2 = EnhancedBayesianNetworks._build_structuralreliabilityproblem_node(functional2_node)
    evaluated_node2 = evaluate!(discrete_srpnode2)

    nodes = [root1, root2, root3, root4, functional1_node, functional2_node, standard3_node]
    ebn = EnhancedBayesianNetwork(nodes)

    @testset "Evaluate StructuralReliabilityProblemNode" begin
        @testset "ContinuousStructuralReliabilityProblemNode" begin
            target1 = Uniform(-1, 1)
            target2 = Uniform(0, 2)
            @test evaluated_node1.name == :F1
            @test issetequal(evaluated_node1.parents, continuous_srpnode1.parents)
            @test isequal(evaluated_node1.discretization, ApproximatedDiscretization())
            @test isapprox(mean(get_randomvariable(evaluated_node1, [:y]).dist), mean(target2); atol=0.1)
            @test isapprox(var(get_randomvariable(evaluated_node1, [:y]).dist), var(target2); atol=0.1)
            @test isapprox(mean(get_randomvariable(evaluated_node1, [:n]).dist), mean(target1); atol=0.1)
            @test isapprox(var(get_randomvariable(evaluated_node1, [:n]).dist), var(target1); atol=0.1)
        end
        @testset "DiscreteStructuralReliabilityProblemNode" begin
            @test evaluated_node2.name == :F2
            @test issetequal(evaluated_node2.parents, discrete_srpnode2.parents)
            @test isequal(evaluated_node2.parameters, Dict{Symbol,Vector{Parameter}}())
            @test isapprox(evaluated_node2.states[[:yes]][:safe_F2], 0.03328; atol=0.1)
            @test isapprox(evaluated_node2.states[[:yes]][:fail_F2], 0.96672; atol=0.1)
            @test isapprox(evaluated_node2.states[[:no]][:safe_F2], 0.0; atol=0.1)
            @test isapprox(evaluated_node2.states[[:no]][:fail_F2], 1.0; atol=0.1)
        end
    end

    @testset "Update network" begin

        updated_ebn = EnhancedBayesianNetworks.update_network!(ebn, functional1_node, continuous_srpnode1)

        fadjlist = [[2, 3], Int64[], Int64[], [7], [7], [7], Int64[]]
        badjlist = [Int64[], [1], [1], Int64[], Int64[], Int64[], [4, 5, 6]]
        ne1 = 5
        name_to_index = Dict(:Std3 => 2, :Y2 => 6, :F1 => 3, :Y1 => 5, :X2 => 4, :X1 => 1, :F2 => 7)

        @test updated_ebn.dag == SimpleDiGraph(ne1, fadjlist, badjlist)
        @test updated_ebn.name_to_index == name_to_index
        @test issetequal(filter(x -> x.name != :F1 && x.name != :F2, updated_ebn.nodes), filter(x -> x.name != :F1 && x.name != :F2, ebn.nodes))
        @test isequal(filter(x -> x.name == :F1, updated_ebn.nodes)[1], continuous_srpnode1)

        update_ebn2 = EnhancedBayesianNetworks.update_network!(updated_ebn, continuous_srpnode1, evaluated_node1)
        fadjlist = [[4], [4], [4], Int64[], [6, 7], Int64[], Int64[]]
        badjlist = [Int64[], Int64[], Int64[], [1, 2, 3], Int64[], [5], [5]]
        ne1 = 5
        name_to_index = Dict(:Std3 => 6, :Y2 => 1, :F1 => 7, :Y1 => 2, :X2 => 3, :X1 => 5, :F2 => 4)

        @test update_ebn2.dag == SimpleDiGraph(ne1, fadjlist, badjlist)
        @test update_ebn2.name_to_index == name_to_index
        @test issetequal(filter(x -> x.name != :F1 && x.name != :F2, update_ebn2.nodes), filter(x -> x.name != :F1 && x.name != :F2, ebn.nodes))
        @test isequal(filter(x -> x.name == :F1, update_ebn2.nodes)[1], evaluated_node1)

    end

    @testset "Evaluate EnhancedBayesianNetwork Single Layer" begin

        root1 = DiscreteRootNode(:X1, Dict(:y => 0.2, :n => 0.8), Dict(:y => [Parameter(1, :X1)], :n => [Parameter(0, :X1)]))
        root2 = DiscreteRootNode(:X2, Dict(:yes => 0.4, :no => 0.6), Dict(:yes => [Parameter(2.2, :X2)], :no => [Parameter(5.5, :X2)]))
        root3 = ContinuousRootNode(:Y1, Uniform(-1, 1))
        root4 = ContinuousRootNode(:Y2, Normal())

        functional1_parents = [root1, root3]
        model1 = [Model(df -> (df.X1 .^ 2) ./ 2 .- df.Y1, :fun1)]
        simulation1 = MonteCarlo(300)
        functional1_node = ContinuousFunctionalNode(:F1, functional1_parents, model1, simulation1)

        standard3_parents = [root1]
        standard3_states = Dict(
            [:y] => Normal(),
            [:n] => Normal(2, 2)
        )
        standard3_node = ContinuousChildNode(:Std3, standard3_parents, standard3_states)

        functional2_name = :F2
        functional2_parents = [root2, root3, root4, functional1_node]
        model2 = [Model(df -> (df.X2 .^ 2 .+ df.Y1 .^ 2) ./ 2 .- df.Y1 .- df.Y2 .+ df.F1, :fun2)]
        performance = df -> 1 .- 2 .* df.fun2
        simulation2 = MonteCarlo(300)
        functional2_node = DiscreteFunctionalNode(functional2_name, functional2_parents, model2, performance, simulation2)

        nodes = [root1, root2, root3, root4, functional1_node, functional2_node, standard3_node]
        ebn = EnhancedBayesianNetwork(nodes)

        single_layer, node_names = EnhancedBayesianNetworks._evaluate_single_layer(ebn)

        fadjlist = [[7], [7], [7], [5, 6], Int64[], [7], Int64[]]
        badjlist = [Int64[], Int64[], Int64[], Int64[], [4], [4], [1, 2, 3, 6]]
        ne1 = 6
        name_to_index = Dict(:Y2 => 1, :Std3 => 5, :F1 => 6, :Y1 => 2, :X2 => 3, :X1 => 4, :F2 => 7)

        @test node_names == [:F1]
        @test single_layer.dag == SimpleDiGraph(ne1, fadjlist, badjlist)
        @test single_layer.name_to_index == name_to_index
        @test issetequal(filter(x -> x.name != :F1 && x.name != :F2, single_layer.nodes), filter(x -> x.name != :F1 && x.name != :F2, ebn.nodes))
        @test isa(filter(x -> x.name == :F1, single_layer.nodes)[1], ContinuousChildNode)
        @test isa(filter(x -> x.name == :F2, single_layer.nodes)[1], DiscreteFunctionalNode)

    end

    @testset "Evaluate EnhancedBayesianNetwork Global" begin
        e_ebn = evaluate!(ebn)
        node_names = [i.name for i in filter(x -> isa(x, FunctionalNode), ebn.nodes)]
        node_to_test = filter(x -> x.name in node_names, e_ebn.nodes)
        @test all([!isa(x, FunctionalNode) for x in node_to_test])
    end
end