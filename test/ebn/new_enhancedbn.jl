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

        @test_throws ErrorException("functional nodes gf must have at least one continuous parent") EnhancedBayesianNetworks._verify_node(grass_functional, [sprinkler])

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

        adj_matrix = sparse(Matrix([0 1.0 1.0 0; 0 0 0 1.0; 0 0 0 1.0; 0 0 0 0]))

        order_net!(net)

        @test net.adj_matrix == adj_matrix
        @test net.topology_dict == Dict(:w => 1, :s => 2, :g => 4, :r => 3)
        @test net.nodes == [weather, sprinkler, rain, grass]

        @test isnothing(EnhancedBayesianNetworks._verify_net(net))

        # @test EnhancedBayesianNetworks._get_edges(net.adj_matrix) == [(1, 2), (1, 3), (2, 4), (3, 4)]

        grass_parents = ([2, 3], [:s, :r], [sprinkler, rain])

        @test get_parents(net, 4) == grass_parents
        @test get_parents(net, :g) == grass_parents
        @test get_parents(net, grass) == grass_parents
    end
end