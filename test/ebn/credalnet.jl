@testset "Credal Networks" begin

    F = DiscreteRootNode(:F, Dict(:Ft => 0.5, :Ff => 0.5))

    B = DiscreteRootNode(:B, Dict(:Bt => 0.5, :Bf => 0.5))

    L = DiscreteChildNode(:L, Dict(
        [:Ft] => Dict(:Lt => 0.3, :Lf => 0.4, :L2 => 0.3),
        [:Ff] => Dict(:Lt => 0.05, :Lf => 0.85, :L2 => 0.1)
    ))
    D = DiscreteChildNode(:D, Dict(
        [:Ft, :Bt] => Dict(:Dt => 0.8, :Df => 0.2),
        [:Ft, :Bf] => Dict(:Dt => 0.1, :Df => 0.9),
        [:Ff, :Bt] => Dict(:Dt => 0.1, :Df => 0.9),
        [:Ff, :Bf] => Dict(:Dt => 0.7, :Df => 0.3)
    ))

    H = DiscreteChildNode(:H, Dict(
        [:Dt] => Dict(:Ht => 0.6, :Hf => 0.4),
        [:Df] => Dict(:Ht => 0.3, :Hf => 0.7)
    ))

    @test_throws ErrorException("networks nodes are all precise. Use BayesianNetwork structure!") CredalNetwork([F, B, L, D, H])

    H = DiscreteChildNode(:H, Dict(
            [:Dt] => Dict(:Ht => 0.6, :Hf => 0.4),
            [:Df] => Dict(:Ht => 0.3, :Hf => 0.7)
        ), Dict(:Ht => [Parameter(1, :H)], :Hf => [Parameter(0, :H)]))

    I = ContinuousRootNode(:I, Normal())

    @test_throws ErrorException("node/s [:I] are continuous. Use EnhancedBayesianNetwork structure!") CredalNetwork([F, B, L, D, H, I])

    H = DiscreteChildNode(:H, Dict(
        [:Dt] => Dict(:Ht => [0.6, 0.8], :Hf => [0.2, 0.4]),
        [:Df] => Dict(:Ht => [0.2, 0.3], :Hf => [0.7, 0.8])
    ))

    cn = CredalNetwork([F, B, L, D, H])
    @test cn.adj_matrix == sparse(zeros(5, 5))
    @test cn.topology_dict == Dict(:F => 1, :H => 5, :D => 4, :B => 2, :L => 3)
    @test issetequal(cn.nodes, [F, B, L, D, H])

    add_child!(cn, F, L)
    add_child!(cn, F, D)
    add_child!(cn, B, D)
    add_child!(cn, D, H)
    order_net!(cn)

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
    order_net!(ebn)

    @test cn == CredalNetwork(ebn)
end