@testset "Credal Networks" begin

    F = DiscreteNode(:F, DataFrame(:F => [:Ft, :Ff], :Π => [0.5, 0.5]))
    B = DiscreteNode(:B, DataFrame(:B => [:Bt, :Bf], :Π => [0.5, 0.5]))
    L = DiscreteNode(:L, DataFrame(:F => [:Ft, :Ft, :Ft, :Ff, :Ff, :Ff], :L => [:Lt, :Lf, :L2, :Lt, :Lf, :L2], :Π => [0.3, 0.4, 0.3, 0.05, 0.85, 0.1]))
    D = DiscreteNode(:D, DataFrame(:F => [:Ft, :Ft, :Ft, :Ft, :Ff, :Ff, :Ff, :Ff], :B => [:Bt, :Bt, :Bf, :Bf, :Bt, :Bt, :Bf, :Bf], :D => [:Dt, :Df, :Dt, :Df, :Dt, :Df, :Dt, :Df], :Π => [0.8, 0.2, 0.1, 0.9, 0.1, 0.9, 0.7, 0.3]))
    H = DiscreteNode(:B, DataFrame(:D => [:Dt, :Dt, :Df, :Df], :B => [:Ht, :Hf, :Ht, :Hf], :Π => [0.6, 0.4, 0.3, 0.7]))
    @test_throws ErrorException("network nodes names must be unique") CredalNetwork([F, B, L, D, H])

    H = DiscreteNode(:H, DataFrame(:D => [:Dt, :Dt, :Df, :Df], :H => [:Bt, :Bf, :Bt, :Bf], :Π => [0.6, 0.4, 0.3, 0.7]))

    @test_throws ErrorException("network nodes states must be unique") CredalNetwork([F, B, L, D, H])

    H = DiscreteNode(:H, DataFrame(:D => [:Dt, :Dt, :Df, :Df], :H => [:Ht, :Hf, :Ht, :Hf], :Π => [0.6, 0.4, 0.3, 0.7]))

    @test_throws ErrorException("all nodes are precise. Use BayesianNetwork structure!") CredalNetwork([F, B, L, D, H])

    H = DiscreteNode(:H, DataFrame(:D => [:Dt, :Dt, :Df, :Df], :H => [:Ht, :Hf, :Ht, :Hf], :Π => [0.6, 0.4, 0.3, 0.7]), Dict(:Ht => [Parameter(1, :H)], :Hf => [Parameter(0, :H)]))

    I = ContinuousNode{UnivariateDistribution}(:I, DataFrame(:Π => Normal()))

    @test_throws ErrorException("node/s [:I] are continuous. Use EnhancedBayesianNetwork structure!") CredalNetwork([F, B, L, D, H, I])

    H = DiscreteNode(:H, DataFrame(:D => [:Dt, :Dt, :Df, :Df], :H => [:Ht, :Hf, :Ht, :Hf], :Π => [[0.6, 0.8], [0.2, 0.4], [0.2, 0.3], [0.7, 0.8]]))

    cn = CredalNetwork([F, B, L, D, H])
    @test cn.adj_matrix == sparse(zeros(5, 5))
    @test cn.topology_dict == Dict(:F => 1, :H => 5, :D => 4, :B => 2, :L => 3)
    @test issetequal(cn.nodes, [F, B, L, D, H])

    add_child!(cn, F, L)
    add_child!(cn, F, D)
    add_child!(cn, B, D)
    add_child!(cn, D, H)
    order!(cn)

    @test cn.adj_matrix == sparse([
        0.0 0.0 1.0 1.0 0.0;
        0.0 0.0 0.0 1.0 0.0;
        0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 1.0;
        0.0 0.0 0.0 0.0 0.0])
    @test cn.topology_dict == Dict(:F => 1, :H => 5, :D => 4, :B => 2, :L => 3)
    @test issetequal(cn.nodes, [F, B, L, D, H])

    ebn = EnhancedBayesianNetwork([F, B, L, D, H])
    add_child!(ebn, F, L)
    add_child!(ebn, F, D)
    add_child!(ebn, B, D)
    add_child!(ebn, D, H)
    order!(ebn)

    @test cn == CredalNetwork(ebn)
end