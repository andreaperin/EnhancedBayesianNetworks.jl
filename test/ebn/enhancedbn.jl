@testset "EnhancedBayesianNetwork" begin

    @testset "EnhancedBayesianNetwork" begin
        weather = DiscreteRootNode(:w, Dict(:sunny => 0.5, :cloudy => 0.5))
        sprinkler_states = Dict(
            [:sunny] => Dict(:on => 0.9, :off => 0.1),
            [:cloudy] => Dict(:on => 0.2, :off => 0.8)
        )
        sprinkler = DiscreteChildNode(:s, sprinkler_states)
        rain_state = Dict(
            [:sunny] => Dict(:no_rain => 0.9, :rain => 0.1),
            [:cloudy] => Dict(:no_rain => 0.2, :rain => 0.8)
        )
        rain = DiscreteChildNode(:r, rain_state)
        grass_states = Dict(
            [:on, :no_rain] => Dict(:dry => 0.9, :wet => 0.1),
            [:on, :rain] => Dict(:dry => 0.9, :wet => 0.1),
            [:off, :no_rain] => Dict(:dry => 0.9, :wet => 0.1),
            [:off, :rain] => Dict(:dry => 0.9, :wet => 0.1)
        )
        grass = DiscreteChildNode(:g, grass_states)

        nodes = [weather, grass, rain, sprinkler]
        net = EnhancedBayesianNetwork(nodes)

        @test net.adj_matrix == sparse(zeros(length(nodes), length(nodes)))
        @test net.topology_dict == Dict(:w => 1, :g => 2, :r => 3, :s => 4)
    end

    @testset "add_child!" begin
        weather = DiscreteRootNode(:w, Dict(:sunny => 0.5, :cloudy => 0.5))
        sprinkler_states = Dict(
            [:sunny] => Dict(:on => 0.9, :off => 0.1),
            [:cloudy] => Dict(:on => 0.2, :off => 0.8)
        )
        sprinkler = DiscreteChildNode(:s, sprinkler_states)
        rain_state = Dict(
            [:random] => Dict(:no_rain => 0.9, :rain => 0.1),
            [:cloudy] => Dict(:no_rain => 0.2, :rain => 0.8)
        )
        rain = DiscreteChildNode(:r, rain_state)
        grass_states = Dict(
            [:on, :no_rain] => Dict(:dry => 0.9, :wet => 0.1),
            [:on, :rain] => Dict(:dry => 0.9, :wet => 0.1),
            [:off, :no_rain] => Dict(:dry => 0.9, :wet => 0.1),
            [:off, :rain] => Dict(:dry => 0.9, :wet => 0.1)
        )
        grass = DiscreteChildNode(:g, grass_states)

        nodes = [weather, grass, rain, sprinkler]
        net = EnhancedBayesianNetwork(nodes)

        @test_throws ErrorException("root node w cannot have parents") add_child!(net, :r, :w)
        @test_throws ErrorException("child node r has scenarios [[:random]], that do not contain any of [:cloudy, :sunny] from its parent w") add_child!(net, :w, :r)
        @test_throws ErrorException("Recursion on the same node is not allowed in EnhancedBayesianNetworks") add_child!(net, :w, :w)

        rain_state = Dict(
            [:sunny] => Dict(:no_rain => 0.9, :rain => 0.1),
            [:cloudy] => Dict(:no_rain => 0.2, :rain => 0.8)
        )
        rain = DiscreteChildNode(:r, rain_state)
        nodes = [weather, grass, rain, sprinkler]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, :w, :r)

        r = net.topology_dict[:w]
        c = net.topology_dict[:r]
        @test net.adj_matrix[r, c] == 1

        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, rain)

        r = net.topology_dict[:w]
        c = net.topology_dict[:r]
        @test net.adj_matrix[r, c] == 1

        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, 1, 3)

        r = net.topology_dict[:w]
        c = net.topology_dict[:r]
        @test net.adj_matrix[r, c] == 1

        grass_model = Model(df -> df.r .+ df.s, :g)
        grass_simulation = MonteCarlo(200)
        grass_performance = df -> df.g
        grass = DiscreteFunctionalNode(:g, [grass_model], grass_performance, grass_simulation)
        nodes = [weather, grass, rain, sprinkler]
        net = EnhancedBayesianNetwork(nodes)
        @test_throws ErrorException("Functional node g can have only functional children, and r is not") add_child!(net, :g, :r)

    end

    @testset "verify nodes" begin
        weather = DiscreteRootNode(:w, Dict(:sunny => 0.5, :cloudy => 0.5))
        sprinkler_states = Dict(
            [:sunny] => Dict(:on => 0.9, :off => 0.1),
            [:cloudy] => Dict(:on => 0.2, :off => 0.8)
        )
        sprinkler = DiscreteChildNode(:s, sprinkler_states)
        rain_state = Dict(
            [:sunny] => Dict(:no_rain => 0.9, :rain => 0.1),
            [:cloudy] => Dict(:no_rain => 0.2, :rain => 0.8)
        )
        rain = DiscreteChildNode(:r, rain_state)
        rain_cont_dist = Dict(
            [:sunny] => Normal(),
            [:cloudy] => Normal()
        )
        rain_cont = ContinuousChildNode(:rc, rain_cont_dist)
        grass_states = Dict(
            [:on, :no_rain, :random] => Dict(:dry => 0.9, :wet => 0.1),
            [:on, :rain] => Dict(:dry => 0.9, :wet => 0.1),
            [:off, :no_rain] => Dict(:dry => 0.9, :wet => 0.1),
            [:off, :rain] => Dict(:dry => 0.9, :wet => 0.1)
        )
        grass = DiscreteChildNode(:g, grass_states)

        grass_functional_model = Model(df -> df.rc .+ df.s, :gf)
        perfomance = df -> df.rc .- 1
        simulation = MonteCarlo(100)
        grass_functional = DiscreteFunctionalNode(:gf, [grass_functional_model], perfomance, simulation)

        @test_throws ErrorException("parents combinations Set{Symbol}[Set([:on, :no_rain])], are missing in node g defined scenarios Set{Symbol}[Set([:on, :random, :no_rain]), Set([:off, :rain]), Set([:off, :no_rain]), Set([:rain, :on])]") EnhancedBayesianNetworks._verify_node(grass, [sprinkler, rain])

        @test isnothing(EnhancedBayesianNetworks._verify_node(sprinkler, [weather]))

        sprinkler = DiscreteChildNode(:s, sprinkler_states, Dict(:on => [Parameter(1, :s)], :off => [Parameter(2, :s)]))
        @test_logs (:warn, "functional nodes gf have no continuous parents") EnhancedBayesianNetworks._verify_node(grass_functional, [sprinkler])

        sprinkler = DiscreteChildNode(:s, sprinkler_states)
        @test_throws ErrorException("node/s [:s] are discrete and parents of the functional node gf, therefore a parameter argument must be defined") EnhancedBayesianNetworks._verify_node(grass_functional, [rain_cont, sprinkler])

        sprinkler = DiscreteChildNode(:s, sprinkler_states, Dict(:on => [Parameter(1, :s)], :off => [Parameter(2, :s)]))

        @test isnothing(EnhancedBayesianNetworks._verify_node(grass_functional, [rain_cont, sprinkler]))
    end

    @testset "EnhancedBayesianNetwork" begin
        weather = DiscreteRootNode(:w, Dict(:sunny => 0.5, :cloudy => 0.5))
        sprinkler_states = Dict(
            [:sunny] => Dict(:on => 0.9, :off => 0.1),
            [:cloudy] => Dict(:on => 0.2, :off => 0.8)
        )
        sprinkler = DiscreteChildNode(:s, sprinkler_states)
        rain_state = Dict(
            [:sunny] => Dict(:no_rain => 0.9, :rain => 0.1),
            [:cloudy] => Dict(:no_rain => 0.2, :rain => 0.8)
        )
        rain = DiscreteChildNode(:r, rain_state)
        grass_states = Dict(
            [:on, :no_rain] => Dict(:dry => 0.9, :wet => 0.1),
            [:on, :rain] => Dict(:dry => 0.9, :wet => 0.1),
            [:off, :no_rain] => Dict(:dry => 0.9, :wet => 0.1),
            [:off, :rain] => Dict(:dry => 0.9, :wet => 0.1)
        )
        grass = DiscreteChildNode(:g, grass_states)

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

        @test EnhancedBayesianNetworks._get_edges(net.adj_matrix) == [(1, 2), (1, 3), (2, 4), (3, 4)]

        grass_parents = ([2, 3], [:s, :r], [sprinkler, rain])

        @test get_parents(net, 4) == grass_parents
        @test get_parents(net, :g) == grass_parents
        @test get_parents(net, grass) == grass_parents

        weather_childern = ([2, 3], [:s, :r], [sprinkler, rain])

        @test get_children(net, 1) == weather_childern
        @test get_children(net, :w) == weather_childern
        @test get_children(net, weather) == weather_childern

        rain_state = Dict(
            [:sunny] => Dict(:sunny => 0.9, :rain => 0.1),
            [:cloudy] => Dict(:sunny => 0.2, :rain => 0.8)
        )
        rain = DiscreteChildNode(:r, rain_state)

        nodes = [weather, sprinkler, rain, grass]
        @test_throws ErrorException("network nodes states must be unique") EnhancedBayesianNetwork(nodes)

        rain = DiscreteChildNode(:w, rain_state)

        nodes = [weather, sprinkler, rain, grass]
        @test_throws ErrorException("network nodes names must be unique") EnhancedBayesianNetwork(nodes)
    end

    @testset "Markov Blanket" begin
        x1 = DiscreteRootNode(:x1, Dict(:x1y => 0.5, :x1n => 0.5))
        x2 = DiscreteRootNode(:x2, Dict(:x2y => 0.5, :x2n => 0.5))
        x4 = DiscreteRootNode(:x4, Dict(:x4y => 0.5, :x4n => 0.5))
        x8 = DiscreteRootNode(:x8, Dict(:x8y => 0.5, :x8n => 0.5))
        x3_states = Dict(
            [:x1y] => Dict(:x3y => 0.5, :x3n => 0.5),
            [:x1n] => Dict(:x3y => 0.5, :x3n => 0.5)
        )
        x3 = DiscreteChildNode(:x3, x3_states)
        x5_states = Dict(
            [:x2y] => Dict(:x5y => 0.5, :x5n => 0.5),
            [:x2n] => Dict(:x5y => 0.5, :x5n => 0.5)
        )
        x5 = DiscreteChildNode(:x5, x5_states)
        x7_states = Dict(
            [:x4y] => Dict(:x7y => 0.5, :x7n => 0.5),
            [:x4n] => Dict(:x7y => 0.5, :x7n => 0.5)
        )
        x7 = DiscreteChildNode(:x7, x7_states)
        x11_states = Dict(
            [:x8y] => Dict(:x11y => 0.5, :x11n => 0.5),
            [:x8n] => Dict(:x11y => 0.5, :x11n => 0.5)
        )
        x11 = DiscreteChildNode(:x11, x11_states)
        x6_states = Dict(
            [:x4y, :x3y] => Dict(:x6y => 0.5, :x6n => 0.5),
            [:x4y, :x3n] => Dict(:x6y => 0.5, :x6n => 0.5),
            [:x4n, :x3y] => Dict(:x6y => 0.5, :x6n => 0.5),
            [:x4n, :x3n] => Dict(:x6y => 0.5, :x6n => 0.5)
        )
        x6 = DiscreteChildNode(:x6, x6_states)
        x9_states = Dict(
            [:x6y, :x5y] => Dict(:x9y => 0.5, :x9n => 0.5),
            [:x6y, :x5n] => Dict(:x9y => 0.5, :x9n => 0.5),
            [:x6n, :x5y] => Dict(:x9y => 0.5, :x9n => 0.5),
            [:x6n, :x5n] => Dict(:x9y => 0.5, :x9n => 0.5)
        )
        x9 = DiscreteChildNode(:x9, x9_states)
        x10_states = Dict(
            [:x6y, :x8y] => Dict(:x10y => 0.5, :x10n => 0.5),
            [:x6y, :x8n] => Dict(:x10y => 0.5, :x10n => 0.5),
            [:x6n, :x8y] => Dict(:x10y => 0.5, :x10n => 0.5),
            [:x6n, :x8n] => Dict(:x10y => 0.5, :x10n => 0.5)
        )
        x10 = DiscreteChildNode(:x10, x10_states)
        x12_states = Dict(
            [:x9y] => Dict(:x12y => 0.5, :x12n => 0.5),
            [:x9n] => Dict(:x12y => 0.5, :x12n => 0.5)
        )
        x12 = DiscreteChildNode(:x12, x12_states)
        x13_states = Dict(
            [:x10y] => Dict(:x13y => 0.5, :x13n => 0.5),
            [:x10n] => Dict(:x13y => 0.5, :x13n => 0.5)
        )
        x13 = DiscreteChildNode(:x13, x13_states)
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
    end

    @testset "Remove/Add Node" begin
        weather = DiscreteRootNode(:w, Dict(:sunny => 0.5, :cloudy => 0.5))
        sprinkler_states = Dict(
            [:sunny] => Dict(:on => 0.9, :off => 0.1),
            [:cloudy] => Dict(:on => 0.2, :off => 0.8)
        )
        sprinkler = DiscreteChildNode(:s, sprinkler_states)
        rain_state = Dict(
            [:sunny] => Dict(:no_rain => 0.9, :rain => 0.1),
            [:cloudy] => Dict(:no_rain => 0.2, :rain => 0.8)
        )
        rain = DiscreteChildNode(:r, rain_state)
        grass_states = Dict(
            [:on, :no_rain] => Dict(:dry => 0.9, :wet => 0.1),
            [:on, :rain] => Dict(:dry => 0.9, :wet => 0.1),
            [:off, :no_rain] => Dict(:dry => 0.9, :wet => 0.1),
            [:off, :rain] => Dict(:dry => 0.9, :wet => 0.1)
        )
        grass = DiscreteChildNode(:g, grass_states)

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

        EnhancedBayesianNetworks._remove_node!(net1, 2)
        EnhancedBayesianNetworks._remove_node!(net2, :s)
        EnhancedBayesianNetworks._remove_node!(net3, sprinkler)


        @test net1.adj_matrix == sparse([0 1.0 0; 0 0 1.0; 0 0 0])
        @test net1.topology_dict == Dict(:w => 1, :g => 3, :r => 2)
        @test issetequal(net1.nodes, [weather, grass, rain])

        @test net2 == net1
        @test net3 == net1

        EnhancedBayesianNetworks._add_node!(net3, sprinkler)

        @test net3.adj_matrix == sparse([0 1.0 0 0; 0 0 1.0 0; 0 0 0 0; 0 0 0 0])
        @test issetequal(net3.nodes, [weather, grass, rain, sprinkler])
        @test net3.topology_dict == Dict(:w => 1, :s => 4, :g => 3, :r => 2)
    end

    @testset "Ancestors" begin
        root1 = DiscreteRootNode(:X1, Dict(:y => 0.2, :n => 0.8))
        root2 = DiscreteRootNode(:X2, Dict(:yes => 0.4, :no => 0.6), Dict(:yes => [Parameter(2.2, :X2)], :no => [Parameter(5.5, :X2)]))
        root3 = ContinuousRootNode(:Y1, Normal(), ExactDiscretization([0, 0.2, 1]))

        child1_states = Dict(
            [:y] => Dict(:c1y => 0.3, :c1n => 0.7),
            [:n] => Dict(:c1y => 0.4, :c1n => 0.6),)
        child1 = DiscreteChildNode(:C1, child1_states, Dict(:c1y => [Parameter(1, :X1)], :c1n => [Parameter(0, :X1)]))

        child2 = ContinuousChildNode(:C2, Dict([:yes] => Normal(), [:no] => Normal(1, 1)))

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
        order!(net)

        @test issetequal(EnhancedBayesianNetworks._get_discrete_ancestors(net, functional1_node), [root2, child1])
        @test isempty(EnhancedBayesianNetworks._get_discrete_ancestors(net, root1))

        @test issetequal([[:c1y, :yes], [:c1n, :yes], [:c1y, :no], [:c1n, :no]], EnhancedBayesianNetworks._get_node_theoretical_scenarios(net, functional1_node))
        @test isempty(EnhancedBayesianNetworks._get_node_theoretical_scenarios(net, root1))

        @testset "Extreme Points" begin
            arg = DiscreteRootNode(:ARG, Dict(:y => [0.2, 0.3], :n => [0.4, 0.6], :m => [0.3, 0.6]))
            probs = EnhancedBayesianNetworks._extreme_points_states_probabilities(arg.states)
            @test isapprox(probs[1][:m], 0.3)
            @test isapprox(probs[1][:n], 0.4)
            @test isapprox(probs[1][:y], 0.3)
            @test isapprox(probs[2][:m], 0.4)
            @test isapprox(probs[2][:n], 0.4)
            @test isapprox(probs[2][:y], 0.2)
            @test isapprox(probs[3][:m], 0.3)
            @test isapprox(probs[3][:n], 0.5)
            @test isapprox(probs[3][:y], 0.2)
        end
    end
    @testset "Cyclicity" begin
        root = DiscreteRootNode(:A, Dict(:a1 => 0.5, :a2 => 0.5))
        child1 = DiscreteChildNode(:B, Dict(
            [:a1, :d1] => Dict(:b1 => 0.5, :b2 => 0.5),
            [:a2, :d1] => Dict(:b1 => 0.5, :b2 => 0.5),
            [:a1, :d2] => Dict(:b1 => 0.5, :b2 => 0.5),
            [:a2, :d2] => Dict(:b1 => 0.5, :b2 => 0.5)
        ))
        child2 = DiscreteChildNode(:C, Dict(
            [:b1] => Dict(:c1 => 0.5, :c2 => 0.5),
            [:b2] => Dict(:c1 => 0.5, :c2 => 0.5),
        ))
        child3 = DiscreteChildNode(:D, Dict(
            [:c1] => Dict(:d1 => 0.5, :d2 => 0.5),
            [:c2] => Dict(:d1 => 0.5, :d2 => 0.5),
        ))
        net = EnhancedBayesianNetwork([root, child1, child2, child3])
        add_child!(net, root, child1)
        add_child!(net, child1, child2)
        add_child!(net, child2, child3)
        add_child!(net, child3, child1)
        @test_throws ErrorException("network is cyclic!") order!(net)
    end

    @testset "Markov Envelopes" begin
        Y1 = DiscreteRootNode(:y1, Dict(:yy1 => 0.5, :yn1 => 0.5), Dict(:yy1 => [Parameter(0.5, :y1)], :yn1 => [Parameter(0.8, :y1)]))
        X1 = ContinuousRootNode(:x1, Normal())
        X2 = ContinuousRootNode(:x2, Normal())
        X3 = ContinuousRootNode(:x3, Normal())

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