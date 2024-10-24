@testset "Discretization structs" begin
    @testset "Exact Discretization" begin
        interval = [-1, 0, 3, 1]
        @test_throws ErrorException("interval values [-1, 0, 3, 1] are not sorted") ExactDiscretization(interval)
        interval = [-1, 0, 1, 3]
        exact_interval = ExactDiscretization([-1, 0, 1, 3])
        @test exact_interval.intervals == interval
        @test isequal(exact_interval, ExactDiscretization(interval))
    end

    @testset "Approximated Discretization" begin
        interval = [-1, 0, 3, 1]
        sigma = 2
        @test_throws ErrorException("interval values [-1, 0, 3, 1] are not sorted") ApproximatedDiscretization(interval, sigma)
        interval = [-1, 0, 1, 3]
        sigma = -1
        @test_throws ErrorException("variance must be positive") ApproximatedDiscretization(interval, sigma)
        sigma = 10
        @test_logs (:warn, "Selected variance values $sigma can be too big, and the approximation not realistic") ApproximatedDiscretization(interval, sigma)
        sigma = 2
        approx_interval = ApproximatedDiscretization([-1, 0, 1, 3], 2)
        @test approx_interval.intervals == interval
        @test isequal(approx_interval, ApproximatedDiscretization(interval, sigma))
    end

    @testset "" begin
        v = DiscreteRootNode(:V, Dict(:yesV => 0.01, :noV => 0.99))
        s = DiscreteRootNode(:S, Dict(:yesS => 0.5, :noS => 0.5))
        t = DiscreteChildNode(:T, [v], Dict(
            [:yesV] => Dict(:yesT => 0.05, :noT => 0.95),
            [:noV] => Dict(:yesT => 0.01, :noT => 0.99))
        )

        l = DiscreteChildNode(:L, [s], Dict(
            [:yesS] => Dict(:yesL => 0.1, :noL => 0.9),
            [:noS] => Dict(:yesL => 0.01, :noL => 0.99))
        )

        b = DiscreteChildNode(:B, [s], Dict(
            [:yesS] => Dict(:yesB => 0.6, :noB => 0.4),
            [:noS] => Dict(:yesB => 0.3, :noB => 0.7))
        )

        e = DiscreteChildNode(:E, [t, l], Dict(
            [:yesT, :yesL] => Dict(:yesE => 1, :noE => 0),
            [:yesT, :noL] => Dict(:yesE => 1, :noE => 0),
            [:noT, :yesL] => Dict(:yesE => 1, :noE => 0),
            [:noT, :noL] => Dict(:yesE => 0, :noE => 01))
        )

        d = DiscreteChildNode(:D, [b, e], Dict(
            [:yesB, :yesE] => Dict(:yesD => 0.9, :noD => 0.1),
            [:yesB, :noE] => Dict(:yesD => 0.8, :noD => 0.2),
            [:noB, :yesE] => Dict(:yesD => 0.7, :noD => 0.3),
            [:noB, :noE] => Dict(:yesD => 0.1, :noD => 0.9))
        )

        x = DiscreteChildNode(:X, [e], Dict(
            [:yesE] => Dict(:yesX => 0.98, :noX => 0.02),
            [:noE] => Dict(:yesX => 0.05, :noX => 0.95))
        )

        nodes = [v, t, e, l, b, s, d, x]

        @test EnhancedBayesianNetworks._order_node(nodes) == [v, s, t, l, b, e, d, x]
        adj = [[0 0 1.0 0 0 0 0 0]
            [0 0 0 1.0 1.0 0 0 0]
            [0 0 0 0 0 1.0 0 0]
            [0 0 0 0 0 1.0 0 0]
            [0 0 0 0 0 0 1.0 0]
            [0 0 0 0 0 0 1.0 1.0]
            [0 0 0 0 0 0 0 0]
            [0 0 0 0 0 0 0 0]]
        adj_matrix = EnhancedBayesianNetworks.get_adj_matrix(nodes)
        @test adj_matrix == adj

        edges_net = [(1, 3), (2, 4), (2, 5), (3, 6), (4, 6), (5, 7), (6, 7), (6, 8)]
        @test EnhancedBayesianNetworks._get_edges(adj_matrix) == edges_net

        positions = [
            [-5.0162312711825985, -1.637492264678324],
            [3.27462979382084, 0.6379696646052927],
            [-3.021902965179569, -1.3266032499014442],
            [1.7962522153008156, -0.7208981551254021],
            [2.259651672069386, 2.0331082548578956],
            [-0.5030477948969894, -0.993862255852705],
            [0.48032282870613446, 1.0823902863377433],
            [-0.5352580271773205, -3.0317538305301897]
        ]
        @test EnhancedBayesianNetworks._get_position(nodes) == positions
    end
end