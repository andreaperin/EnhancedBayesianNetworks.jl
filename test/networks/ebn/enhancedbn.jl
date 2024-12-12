@testset "EnhancedBayesianNetwork" begin

    @testset "Structure" begin
        weather = DiscreteNode(:w, DataFrame(:w => [:sunny, :cloudy], :Prob => [0.5, 0.5]))
        sprinkler_states = DataFrame(:w => [:sunny, :sunny, :cloudy, :cloudy], :s => [:on, :off, :on, :off], :Prob => [0.9, 0.1, 0.2, 0.8])
        sprinkler = DiscreteNode(:s, sprinkler_states)
        rain_state = DataFrame(:w => [:sunny, :sunny, :cloudy, :cloudy], :r => [:no_rain, :rain, :no_rain, :rain], :Prob => [0.9, 0.1, 0.2, 0.8])
        rain = DiscreteNode(:r, rain_state)

        grass_states = DataFrame(:s => [:on, :on, :on, :on, :off, :off, :off, :off], :r => [:no_rain, :no_rain, :rain, :rain, :no_rain, :no_rain, :rain, :rain], :g => [:dry, :wet, :dry, :wet, :dry, :wet, :dry, :wet], :Prob => [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
        grass = DiscreteNode(:g, grass_states)

        nodes = [weather, grass, rain, sprinkler]
        net = EnhancedBayesianNetwork(nodes)

        @test net.adj_matrix == sparse(zeros(length(nodes), length(nodes)))
        @test net.topology_dict == Dict(:w => 1, :g => 2, :r => 3, :s => 4)
    end

    @testset "add_child!" begin
        weather = DiscreteNode(:w, DataFrame(:w => [:sunny, :cloudy], :Prob => [0.5, 0.5]))
        sprinkler_states = DataFrame(
            :w => [:sunny, :sunny, :cloudy, :cloudy], :s => [:on, :off, :on, :off], :Prob => [0.9, 0.1, 0.2, 0.8]
        )
        sprinkler = DiscreteNode(:s, sprinkler_states)
        rain_state = DataFrame(:w => [:sunny, :sunny, :cloudy, :cloudy], :r => [:no_rain, :rain, :no_rain, :rain], :Prob => [0.9, 0.1, 0.2, 0.8])
        rain = DiscreteNode(:r, rain_state)

        grass_states = DataFrame(:s => [:on, :on, :on, :on, :off, :off, :off, :off], :r => [:no_rain, :no_rain, :rain, :rain, :no_rain, :no_rain, :rain, :rain], :g => [:dry, :wet, :dry, :wet, :dry, :wet, :dry, :wet], :Prob => [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
        grass = DiscreteNode(:g, grass_states)

        nodes = [weather, grass, rain, sprinkler]
        net = EnhancedBayesianNetwork(nodes)
        topology_dict = Dict(:w => 1, :s => 4, :g => 2, :r => 3)
        adj_matrix = spzeros(4, 4)
        @test net.topology_dict == topology_dict
        @test issetequal(net.nodes, nodes)
        @test net.adj_matrix == adj_matrix

        @test_throws ErrorException("Recursion on the same node 'w' is not allowed in EnhancedBayesianNetworks") add_child!(net, :w, :w)
        @test_throws ErrorException("node 'w' is a root node and cannot have parents") add_child!(net, :s, :w)
        rain_state1 = DataFrame(:a => [:sunny, :sunny, :cloudy, :cloudy], :r => [:no_rain, :rain, :no_rain, :rain], :Prob => [0.9, 0.1, 0.2, 0.8])
        rain1 = DiscreteNode(:r, rain_state1)
        nodes1 = [weather, grass, rain1, sprinkler]
        net1 = EnhancedBayesianNetwork(nodes1)
        @test_throws ErrorException("trying to set node 'r' as child of node 'w', but 'r' has a cpt that does not contains 'w' in the scenarios: $(rain1.cpt)") add_child!(net1, weather, rain1)
        rain_state2 = DataFrame(:w => [:sunny, :sunny, :cloudies, :cloudies], :r => [:no_rain, :rain, :no_rain, :rain], :Prob => [0.9, 0.1, 0.2, 0.8])
        rain2 = DiscreteNode(:r, rain_state2)
        nodes2 = [weather, grass, rain2, sprinkler]
        net2 = EnhancedBayesianNetwork(nodes2)
        @test_throws ErrorException("child node 'r' has scenarios [:cloudies, :sunny], that is not coherent with its parent node 'w' with states [:cloudy, :sunny]") add_child!(net2, weather, rain2)

        grass_model = Model(df -> df.r .+ df.s, :g)
        grass_simulation = MonteCarlo(200)
        grass_performance = df -> df.g
        grass1 = DiscreteFunctionalNode(:g, [grass_model], grass_performance, grass_simulation)
        nodes4 = [weather, rain, sprinkler, grass1]
        net4 = EnhancedBayesianNetwork(nodes4)
        @test_throws ErrorException("functional node 'g' can have only functional children. 'r' is not a functional node") add_child!(net4, :g, :r)
        net_new1 = deepcopy(net)
        net_new2 = deepcopy(net)
        add_child!(net, weather, rain)
        adj_matrix_net = sparse([
            0.0 0.0 1.0 0.0;
            0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0
        ])
        @test net.topology_dict == topology_dict
        @test net.nodes == nodes
        @test net.adj_matrix == adj_matrix_net
        add_child!(net_new1, :w, :r)
        @test net_new1 == net
        add_child!(net_new2, 1, 3)
        add_child!(net_new2, :w, :r)
    end

    @testset "parents & children" begin
        weather = DiscreteNode(:w, DataFrame(:w => [:sunny, :cloudy], :Prob => [0.5, 0.5]))
        sprinkler_states = DataFrame(
            :w => [:sunny, :sunny, :cloudy, :cloudy], :s => [:on, :off, :on, :off], :Prob => [0.9, 0.1, 0.2, 0.8]
        )
        sprinkler = DiscreteNode(:s, sprinkler_states)
        rain_state = DataFrame(:w => [:sunny, :sunny, :cloudy, :cloudy], :r => [:no_rain, :rain, :no_rain, :rain], :Prob => [0.9, 0.1, 0.2, 0.8])
        rain = DiscreteNode(:r, rain_state)

        grass_model = Model(df -> df.r .+ df.s, :g)
        grass_simulation = MonteCarlo(200)
        grass_performance = df -> df.g
        grass = DiscreteFunctionalNode(:g, [grass_model], grass_performance, grass_simulation)
        nodes = [weather, rain, sprinkler, grass]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, :w, :r)
        add_child!(net, :w, :s)
        add_child!(net, :s, :g)
        add_child!(net, :r, :g)

        @test parents(net, :w) == (Int64[], Symbol[], AbstractNode[])
        @test parents(net, 1) == (Int64[], Symbol[], AbstractNode[])
        @test parents(net, weather) == (Int64[], Symbol[], AbstractNode[])

        @test parents(net, :g) == ([2, 3], [:r, :s], [rain, sprinkler])
        @test parents(net, 4) == ([2, 3], [:r, :s], [rain, sprinkler])
        @test parents(net, grass) == ([2, 3], [:r, :s], [rain, sprinkler])

        @test children(net, :w) == ([2, 3], [:r, :s], [rain, sprinkler])
        @test children(net, 1) == ([2, 3], [:r, :s], [rain, sprinkler])
        @test children(net, weather) == ([2, 3], [:r, :s], [rain, sprinkler])

        @test children(net, :g) == (Int64[], Symbol[], AbstractNode[])
        @test children(net, 4) == (Int64[], Symbol[], AbstractNode[])
        @test children(net, grass) == (Int64[], Symbol[], AbstractNode[])
    end
    @testset "discrete_ancestors" begin
        root1 = DiscreteNode(:X1, DataFrame(:X1 => [:y, :n], :Prob => [0.2, 0.8]))
        root2 = DiscreteNode(:X2, DataFrame(:X2 => [:yes, :no], :Prob => [0.4, 0.6]), Dict(:yes => [Parameter(2.2, :X2)], :no => [Parameter(5.5, :X2)]))
        root3 = ContinuousNode{UnivariateDistribution}(:Y1, DataFrame(:Prob => Normal()), ExactDiscretization([0, 0.2, 1]))

        child1_states = DataFrame(:X1 => [:y, :y, :n, :n], :C1 => [:c1y, :c1n, :c1y, :c1n], :Prob => [0.3, 0.7, 0.4, 0.6])
        child1 = DiscreteNode(:C1, child1_states, Dict(:c1y => [Parameter(1, :X1)], :c1n => [Parameter(0, :X1)]))

        child2 = ContinuousNode{UnivariateDistribution}(:C2, DataFrame(:X2 => [:yes, :no], :Prob => [Normal(), Normal(1, 1)]))

        functional1_parents = [child1, child2]
        disc_D = ApproximatedDiscretization([-1.1, 0, 0.11], 2)
        model1 = [Model(df -> (df.X1 .^ 2) ./ 2 .- df.C2, :fun1)]
        simulation1 = MonteCarlo(300)
        functional1_node = ContinuousFunctionalNode(:F1, model1, simulation1, disc_D)

        net = EnhancedBayesianNetwork([root1, root2, root3, child1, child2, functional1_node])
        add_child!(net, root1, child1)
        add_child!(net, root2, child2)
        add_child!(net, child1, functional1_node)
        add_child!(net, child2, functional1_node)

        @test issetequal(discrete_ancestors(net, functional1_node), [root2, child1])
        @test isempty(discrete_ancestors(net, root1))
        th_scenarios = [
            Dict(:C1 => :c1n, :X2 => :no)
            Dict(:C1 => :c1y, :X2 => :no)
            Dict(:C1 => :c1n, :X2 => :yes)
            Dict(:C1 => :c1y, :X2 => :yes)]
        @test EnhancedBayesianNetworks._theoretical_scenarios(net, functional1_node) == th_scenarios
        @test EnhancedBayesianNetworks._theoretical_scenarios(net, root1) == [Dict()]
    end

    @testset "verify net" begin
        weather = DiscreteNode(:w, DataFrame(:w => [:sunny, :cloudy], :Prob => [0.5, 0.5]))
        sprinkler_states = DataFrame(
            :w => [:sunny, :sunny, :cloudy, :cloudy], :s => [:on, :off, :on, :off], :Prob => [0.9, 0.1, 0.2, 0.8]
        )
        sprinkler = DiscreteNode(:s, sprinkler_states)
        rain_state = DataFrame(:w => [:sunny, :sunny, :cloudy, :cloudy], :r => [:no_rain, :rain, :no_rain, :rain], :Prob => [0.9, 0.1, 0.2, 0.8])
        rain = DiscreteNode(:r, rain_state)

        grass_model = Model(df -> df.r .+ df.s, :g)
        grass_simulation = MonteCarlo(200)
        grass_performance = df -> df.g
        grass = DiscreteFunctionalNode(:g, [grass_model], grass_performance, grass_simulation)
        nodes = [weather, rain, sprinkler, grass]
        net = EnhancedBayesianNetwork(nodes)
        @test isnothing(EnhancedBayesianNetworks._verify_child_node(net, weather))
        @test_throws ErrorException("node 'r''s cpt requires exctly the nodes '[:w]' to be its parents, but provided parents are 'Symbol[]'") EnhancedBayesianNetworks._verify_child_node(net, rain)
        @test_throws ErrorException("node 'r''s cpt requires exctly the nodes '[:w]' to be its parents, but provided parents are 'Symbol[]'") EnhancedBayesianNetworks._verify_net(net)

        add_child!(net, :w, :r)
        add_child!(net, :w, :s)

        @test_throws ErrorException("functional node 'g' must have at least one parent") EnhancedBayesianNetworks._verify_functional_node(net, grass)
        @test_throws ErrorException("functional node 'g' must have at least one parent") EnhancedBayesianNetworks._verify_net(net)

        add_child!(net, :s, :g)
        add_child!(net, :r, :g)

        @test_throws ErrorException("node 's' is a discrete parent of a functional node and cannot have an empty parameters vector") EnhancedBayesianNetworks._non_empty_parameters_vector(net, sprinkler)
        @test_throws ErrorException("node 'r' is a discrete parent of a functional node and cannot have an empty parameters vector") EnhancedBayesianNetworks._non_empty_parameters_vector(net, rain)
        @suppress @test_throws ErrorException("node 'r' is a discrete parent of a functional node and cannot have an empty parameters vector") EnhancedBayesianNetworks._verify_net(net)

        sprinkler2 = DiscreteNode(:s, sprinkler_states, Dict(:on => [Parameter(1, :S)], :off => [Parameter(2, :S)]))
        nodes2 = [weather, rain, sprinkler2, grass]
        net2 = EnhancedBayesianNetwork(nodes2)
        add_child!(net2, weather, sprinkler2)
        add_child!(net2, weather, rain)
        add_child!(net2, sprinkler2, grass)
        @test isnothing(EnhancedBayesianNetworks._non_empty_parameters_vector(net, sprinkler2))
        @test_logs (:warn, "functional node 'g' have no continuous parents. All the simulations will return the same output") EnhancedBayesianNetworks._verify_functional_node(net2, grass)
        @test_logs (:warn, "functional node 'g' have no continuous parents. All the simulations will return the same output") EnhancedBayesianNetworks._verify_net(net2)
        @suppress isnothing(EnhancedBayesianNetworks._verify_functional_node(net2, grass))
        @suppress isnothing(EnhancedBayesianNetworks._verify_net(net2))

        nodes3 = [weather, rain, sprinkler, grass]
        net3 = EnhancedBayesianNetwork(nodes3)
        @test_throws ErrorException("functional node 'g' must have at least one parent") EnhancedBayesianNetworks._verify_functional_node(net3, grass)
        add_child!(net2, sprinkler2, grass)
        @test_logs (:warn, "functional node 'g' have no continuous parents. All the simulations will return the same output") EnhancedBayesianNetworks._verify_functional_node(net2, grass)
        @suppress isnothing(EnhancedBayesianNetworks._verify_functional_node(net2, grass)
        )

        sprinkler_states3 = DataFrame(
            :w => [:sunny, :sunny, :cloudy, :cloudy, :t, :t, :not_t, :not_t], :s => [:on, :off, :on, :off, :on, :off, :on, :off], :Prob => [0.9, 0.1, 0.2, 0.8, 0.1, 0.9, 0.4, 0.6]
        )
        sprinkler3 = DiscreteNode(:s, sprinkler_states3)
        net3 = EnhancedBayesianNetwork([weather, rain, sprinkler3, grass])
        add_child!(net3, :w, :r)
        add_child!(net3, :w, :s)
        add_child!(net3, :s, :g)
        add_child!(net3, :r, :g)
        @test_throws ErrorException("node 's' has defined cpt scenarios $(sprinkler3.cpt) not coherent with the theoretical one [Dict(:w => :cloudy), Dict(:w => :sunny)]") EnhancedBayesianNetworks._verify_child_node(net3, sprinkler3)
        @test_throws ErrorException("node 's' has defined cpt scenarios $(sprinkler3.cpt) not coherent with the theoretical one [Dict(:w => :cloudy), Dict(:w => :sunny)]") EnhancedBayesianNetworks._verify_net(net3)
    end

    @testset "order network" begin
        root = DiscreteNode(:A, DataFrame(:A => [:a1, :a2], :Prob => [0.5, 0.5]))
        child1 = DiscreteNode(:B, DataFrame(:D => [:d1, :d2, :d1, :d2, :d1, :d2, :d1, :d2], :A => [:a1, :a1, :a2, :a2, :a1, :a1, :a2, :a2], :B => [:b1, :b2, :b1, :b2, :b1, :b2, :b1, :b2], :Prob => [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
        child2 = DiscreteNode(:C, DataFrame(:B => [:b1, :b1, :b2, :b2], :C => [:c1, :c2, :c1, :c2], :Prob => [0.5, 0.5, 0.5, 0.5]))
        child3 = DiscreteNode(:D, DataFrame(:C => [:c1, :c1, :c2, :c2], :D => [:d1, :d2, :d1, :d2], :Prob => [0.5, 0.5, 0.5, 0.5]))
        net = EnhancedBayesianNetwork([root, child1, child2, child3])
        add_child!(net, :A, :B)
        add_child!(net, :B, :C)
        add_child!(net, :C, :D)
        add_child!(net, :D, :B)
        @test EnhancedBayesianNetworks._is_cyclic_dfs(net.adj_matrix)
        @test_throws ErrorException("network is cyclic!") order!(net)

        weather = DiscreteNode(:w, DataFrame(:w => [:sunny, :cloudy], :Prob => [0.5, 0.5]))
        sprinkler_states = DataFrame(:w => [:sunny, :sunny, :cloudy, :cloudy], :s => [:on, :off, :on, :off], :Prob => [0.9, 0.1, 0.2, 0.8])
        sprinkler = DiscreteNode(:s, sprinkler_states)
        rain_state = DataFrame(:w => [:sunny, :sunny, :cloudy, :cloudy], :r => [:no_rain, :rain, :no_rain, :rain], :Prob => [0.9, 0.1, 0.2, 0.8])
        rain = DiscreteNode(:r, rain_state)
        grass_states = DataFrame(:s => [:on, :on, :on, :on, :off, :off, :off, :off], :r => [:no_rain, :no_rain, :rain, :rain, :no_rain, :no_rain, :rain, :rain], :g => [:dry, :wet, :dry, :wet, :dry, :wet, :dry, :wet], :Prob => [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
        grass = DiscreteNode(:g, grass_states)
        nodes = [weather, sprinkler, rain, grass]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, :w, :s)
        add_child!(net, :w, :r)
        add_child!(net, :s, :g)
        add_child!(net, :r, :g)

        order!(net)
        @test net.adj_matrix == sparse(Matrix([0 1.0 1.0 0; 0 0 0 1.0; 0 0 0 1.0; 0 0 0 0]))
        @test net.topology_dict == Dict(:w => 1, :s => 2, :g => 4, :r => 3)
        @test net.nodes == [weather, sprinkler, rain, grass]

        @test isnothing(EnhancedBayesianNetworks._verify_net(net))
        rain_state = DataFrame(:w => [:sunny, :sunny, :cloudy, :cloudy], :g => [:no_rain, :rain, :no_rain, :rain], :Prob => [0.9, 0.1, 0.2, 0.8])
        rain = DiscreteNode(:g, rain_state)
        nodes = [weather, sprinkler, rain, grass]
        @test_throws ErrorException("network nodes names must be unique") EnhancedBayesianNetwork(nodes)
        rain_state = DataFrame(:w => [:sunny, :sunny, :cloudy, :cloudy], :r => [:on, :off, :on, :off], :Prob => [0.9, 0.1, 0.2, 0.8])
        rain = DiscreteNode(:r, rain_state)
        nodes = [weather, sprinkler, rain, grass]
        @test_throws ErrorException("network nodes states must be unique") EnhancedBayesianNetwork(nodes)
    end

    @testset "add & remove nodes" begin

        weather = DiscreteNode(:w, DataFrame(:w => [:sunny, :cloudy], :Prob => [0.5, 0.5]))
        sprinkler_states = DataFrame(:w => [:sunny, :sunny, :cloudy, :cloudy], :s => [:on, :off, :on, :off], :Prob => [0.9, 0.1, 0.2, 0.8])
        sprinkler = DiscreteNode(:s, sprinkler_states)
        rain_state = DataFrame(:w => [:sunny, :sunny, :cloudy, :cloudy], :r => [:no_rain, :rain, :no_rain, :rain], :Prob => [0.9, 0.1, 0.2, 0.8])
        rain = DiscreteNode(:r, rain_state)
        grass_states = DataFrame(:s => [:on, :on, :on, :on, :off, :off, :off, :off], :r => [:no_rain, :no_rain, :rain, :rain, :no_rain, :no_rain, :rain, :rain], :g => [:dry, :wet, :dry, :wet, :dry, :wet, :dry, :wet], :Prob => [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
        grass = DiscreteNode(:g, grass_states)
        nodes = [weather, sprinkler, rain, grass]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, :w, :s)
        add_child!(net, :w, :r)
        add_child!(net, :s, :g)
        add_child!(net, :r, :g)
        order!(net)
        net1 = deepcopy(net)
        net2 = deepcopy(net)
        net3 = deepcopy(net)

        EnhancedBayesianNetworks._remove_node!(net1, grass)
        EnhancedBayesianNetworks._remove_node!(net2, :g)
        EnhancedBayesianNetworks._remove_node!(net3, 4)


        @test issetequal(net1.nodes, [weather, sprinkler, rain])
        adj = [
            0.0 1.0 1.0;
            0.0 0.0 0.0;
            0.0 0.0 0.0
        ]
        @test net1.adj_matrix == adj
        @test net1.topology_dict == Dict(:w => 1, :r => 3, :s => 2)
        @test net2 == net1
        @test net3 == net1

        net4 = deepcopy(net1)
        net5 = deepcopy(net1)
        net6 = deepcopy(net1)

        EnhancedBayesianNetworks._add_node!(net4, grass)
        EnhancedBayesianNetworks._add_node!(net5, grass)
        EnhancedBayesianNetworks._add_node!(net6, grass)

        add_child!(net4, :s, :g)
        add_child!(net4, :r, :g)
        order!(net4)
        add_child!(net5, :s, :g)
        add_child!(net5, :r, :g)
        order!(net5)
        add_child!(net6, :s, :g)
        add_child!(net6, :r, :g)
        order!(net6)

        @test net4 == net
        @test net5 == net
        @test net6 == net
    end

    @testset "Markov Blankets" begin
        x1 = DiscreteNode(:x1, DataFrame(:x1 => [:x1y, :x1n], :Prob => [0.5, 0.5]))
        x2 = DiscreteNode(:x2, DataFrame(:x2 => [:x2y, :x2n], :Prob => [0.5, 0.5]))
        x4 = DiscreteNode(:x4, DataFrame(:x4 => [:x4y, :x4n], :Prob => [0.5, 0.5]))
        x8 = DiscreteNode(:x8, DataFrame(:x8 => [:x8y, :x8n], :Prob => [0.5, 0.5]))
        x3_states = DataFrame(
            :x1 => [:x1y, :x1y, :x1n, :x1n],
            :x3 => [:x3y, :x3n, :x3y, :x3n],
            :Prob => [0.5, 0.5, 0.5, 0.5]
        )

        x3 = DiscreteNode(:x3, x3_states)
        x5_states = DataFrame(
            :x2 => [:x2y, :x2y, :x2n, :x2n],
            :x5 => [:x5y, :x5n, :x5y, :x5n],
            :Prob => [0.5, 0.5, 0.5, 0.5]
        )
        x5 = DiscreteNode(:x5, x5_states)
        x7_states = DataFrame(
            :x4 => [:x4y, :x4y, :x4n, :x4n],
            :x7 => [:x7y, :x7n, :x7y, :x7n],
            :Prob => [0.5, 0.5, 0.5, 0.5]
        )
        x7 = DiscreteNode(:x7, x7_states)
        x11_states = DataFrame(
            :x8 => [:x8y, :x8y, :x8n, :x8n],
            :x11 => [:x11y, :x11n, :x11y, :x11n],
            :Prob => [0.5, 0.5, 0.5, 0.5]
        )
        x11 = DiscreteNode(:x11, x11_states)
        x6_states = DataFrame(
            :x4 => [:x4y, :x4y, :x4y, :x4y, :x4n, :x4n, :x4n, :x4n],
            :x3 => [:x3y, :x3y, :x3n, :x3n, :x3y, :x3y, :x3n, :x3n],
            :x6 => [:x6y, :x6n, :x6y, :x6n, :x6y, :x6n, :x6y, :x6n],
            :Prob => [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        )
        x6 = DiscreteNode(:x6, x6_states)
        x9_states = DataFrame(
            :x6 => [:x6y, :x6y, :x6y, :x6y, :x6n, :x6n, :x6n, :x6n],
            :x5 => [:x5y, :x5y, :x5n, :x5n, :x5y, :x5y, :x5n, :x5n],
            :x9 => [:x9y, :x9n, :x9y, :x9n, :x9y, :x9n, :x9y, :x9n],
            :Prob => [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        )
        x9 = DiscreteNode(:x9, x9_states)
        x10_states = DataFrame(
            :x6 => [:x6y, :x6y, :x6y, :x6y, :x6n, :x6n, :x6n, :x6n],
            :x8 => [:x8y, :x8y, :x8n, :x8n, :x8y, :x8y, :x8n, :x8n],
            :x10 => [:x10y, :x10n, :x10y, :x10n, :x10y, :x10n, :x10y, :x10n],
            :Prob => [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        )
        x10 = DiscreteNode(:x10, x10_states)
        x12_states = DataFrame(
            :x9 => [:x9y, :x9y, :x9n, :x9n],
            :x12 => [:x12y, :x12n, :x12y, :x12n],
            :Prob => [0.5, 0.5, 0.5, 0.5]
        )
        x12 = DiscreteNode(:x12, x12_states)
        x13_states = DataFrame(
            :x10 => [:x10y, :x10y, :x10n, :x10n],
            :x13 => [:x13y, :x13n, :x13y, :x13n],
            :Prob => [0.5, 0.5, 0.5, 0.5]
        )
        x13 = DiscreteNode(:x13, x13_states)
        nodes = [x1, x2, x4, x8, x5, x7, x11, x3, x6, x9, x10, x12, x13]
        ebn = EnhancedBayesianNetwork(nodes)
        add_child!(ebn, :x1, :x3)
        add_child!(ebn, :x2, :x5)
        add_child!(ebn, :x4, :x7)
        add_child!(ebn, :x8, :x11)
        add_child!(ebn, :x3, :x6)
        add_child!(ebn, :x4, :x6)
        add_child!(ebn, :x5, :x9)
        add_child!(ebn, :x6, :x9)
        add_child!(ebn, :x6, :x10)
        add_child!(ebn, :x8, :x10)
        add_child!(ebn, :x9, :x12)
        add_child!(ebn, :x10, :x13)
        order!(ebn)

        markov_bl = markov_blanket(ebn, 9)
        @test issetequal(markov_bl[1], [5 10 4 11 3 8])
        @test issetequal(markov_bl[2], [:x3, :x4, :x5, :x8, :x9, :x10])
        @test issetequal(markov_bl[3], [x3, x4, x5, x8, x9, x10])
        @test markov_blanket(ebn, :x6) == markov_bl
        @test markov_blanket(ebn, x6) == markov_bl
    end

    @testset "Markov Envelopes" begin
        Y1 = DiscreteNode(:y1, DataFrame(:y1 => [:yy1, :yn1], :Prob => [0.5, 0.5]), Dict(:yy1 => [Parameter(0.5, :y1)], :yn1 => [Parameter(0.8, :y1)]))
        X1 = ContinuousNode{UnivariateDistribution}(:x1, DataFrame(:Prob => Normal()))
        X2 = ContinuousNode{UnivariateDistribution}(:x2, DataFrame(:Prob => Normal()))
        X3 = ContinuousNode{UnivariateDistribution}(:x3, DataFrame(:Prob => Normal()))

        model = Model(df -> df.y1 .+ df.x1, :y2)
        models = [model]
        simulation = MonteCarlo(200)
        performance = df -> df.y2
        Y2 = DiscreteFunctionalNode(:y2, models, performance, simulation)

        model = Model(df -> df.x1 .+ df.x2, :y3)
        models = [model]
        simulation = MonteCarlo(200)
        performance = df -> df.y3
        Y3 = DiscreteFunctionalNode(:y3, models, performance, simulation)

        model = Model(df -> df.x3 .+ df.x2, :y4)
        models = [model]
        simulation = MonteCarlo(200)
        performance = df -> df.y4
        Y4 = DiscreteFunctionalNode(:y4, models, performance, simulation)

        model = Model(df -> df.x3, :y5)
        models = [model]
        simulation = MonteCarlo(200)
        performance = df -> df.y5
        parameter = Dict(:fail_y5 => [Parameter(1, :y5)], :fail_y5 => [Parameter(0, :y5)])
        Y5 = DiscreteFunctionalNode(:y5, models, performance, simulation, parameter)

        model = Model(df -> df.y3, :x4)
        models = [model]
        simulation = MonteCarlo(200)
        X4 = ContinuousFunctionalNode(:x4, models, simulation)

        model = Model(df -> df.x4, :y6)
        models = [model]
        simulation = MonteCarlo(200)
        performance = df -> df.y6
        Y6 = DiscreteFunctionalNode(:y6, models, performance, simulation)
        nodes = [X1, X2, X3, Y1, Y2, Y3, Y4, Y5, X4, Y6]
        ebn = EnhancedBayesianNetwork(nodes)

        add_child!(ebn, :y1, :y2)
        add_child!(ebn, :x1, :y2)
        add_child!(ebn, :x1, :y3)
        add_child!(ebn, :x2, :y3)
        add_child!(ebn, :x2, :y4)
        add_child!(ebn, :x3, :y4)
        add_child!(ebn, :x3, :y5)
        add_child!(ebn, :y5, :x4)
        add_child!(ebn, :x4, :y6)
        @suppress order!(ebn)

        @test issetequal(EnhancedBayesianNetworks._get_markov_group(ebn, Y5), [Y5, X4, X3])
        envelopes = markov_envelope(ebn)
        @test issetequal(envelopes[1], [X1, X2, X3, Y1, Y2, Y3, Y4, Y5])
        @test issetequal(envelopes[2], [Y6, Y5, X4])
    end
end