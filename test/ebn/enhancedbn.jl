@testset "EnhancedBayesianNetwork" begin

    @testset "Structure" begin
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
        @test_throws ErrorException("trying to set node 'r' as child of node 'w', but 'r' has a cpt that does not contains 'w' in the scenarios: $rain_state1") add_child!(net1, weather, rain1)
        rain_state2 = DataFrame(:w => [:sunny, :sunny, :cloudies, :cloudies], :r => [:no_rain, :rain, :no_rain, :rain], :Prob => [0.9, 0.1, 0.2, 0.8])
        rain2 = DiscreteNode(:r, rain_state2)
        nodes2 = [weather, grass, rain2, sprinkler]
        net2 = EnhancedBayesianNetwork(nodes2)
        @test_throws ErrorException("child node 'r' has scenarios [:sunny, :cloudies], that is not coherent with its parent node 'w' with states [:sunny, :cloudy]") add_child!(net2, weather, rain2)

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
        th_scenarios = [Dict(:C1 => :c1y, :X2 => :yes)
            Dict(:C1 => :c1n, :X2 => :yes)
            Dict(:C1 => :c1y, :X2 => :no)
            Dict(:C1 => :c1n, :X2 => :no)]
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
        add_child!(net, :w, :r)
        @test isnothing(EnhancedBayesianNetworks._verify_child_node(net, weather))
        @test_throws ErrorException("node 's''s cpt requires exctly the nodes '[:w]' to be its parents, but provided parents are 'Symbol[]'") EnhancedBayesianNetworks._verify_child_node(net, sprinkler)
        @test_throws ErrorException("node 's''s cpt requires exctly the nodes '[:w]' to be its parents, but provided parents are 'Symbol[]'") EnhancedBayesianNetworks._verify_net(net)

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
        @test_throws ErrorException("node 's' has defined cpt scenarios $(sprinkler3.cpt) not coherent with the theoretical one [Dict(:w => :sunny), Dict(:w => :cloudy)]") EnhancedBayesianNetworks._verify_child_node(net3, sprinkler3)
        @test_throws ErrorException("node 's' has defined cpt scenarios $(sprinkler3.cpt) not coherent with the theoretical one [Dict(:w => :sunny), Dict(:w => :cloudy)]") EnhancedBayesianNetworks._verify_net(net3)
    end
end