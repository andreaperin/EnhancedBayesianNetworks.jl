@testset "Credal Networks" begin

    F = DiscreteRootNode(:F, Dict(:Ft => 0.5, :Ff => 0.5))

    B = DiscreteRootNode(:B, Dict(:Bt => 0.5, :Bf => 0.5))

    L = DiscreteChildNode(:L, [F], Dict(
        [:Ft] => Dict(:Lt => 0.3, :Lf => 0.4, :L2 => 0.3),
        [:Ff] => Dict(:Lt => 0.05, :Lf => 0.85, :L2 => 0.1)
    ))

    D = DiscreteChildNode(:D, [F, B], Dict(
        [:Ft, :Bt] => Dict(:Dt => 0.8, :Df => 0.2),
        [:Ft, :Bf] => Dict(:Dt => 0.1, :Df => 0.9),
        [:Ff, :Bt] => Dict(:Dt => 0.1, :Df => 0.9),
        [:Ff, :Bf] => Dict(:Dt => 0.7, :Df => 0.3)
    ))

    H = DiscreteChildNode(:H, [D], Dict(
        [:Dt] => Dict(:Ht => 0.6, :Hf => 0.4),
        [:Df] => Dict(:Ht => 0.3, :Hf => 0.7)
    ))

    @test_throws ErrorException("When all nodes are precise use BayesNetwork structure") CredalNetwork([F, B, L, D, H])

    F = DiscreteRootNode(:F, Dict(:Ft => [0.4, 0.5], :Ff => [0.5, 0.6]))

    H = ContinuousChildNode(:H, [D], Dict(
        [:Dt] => Normal(1, 1),
        [:Df] => Normal()
    ))

    @test_throws ErrorException("Credal Network allows discrete node only!") CredalNetwork([F, B, L, D, H])

    H = DiscreteChildNode(:H, [D], Dict(
        [:Dt] => Dict(:Ht => 0.6, :Hf => 0.4),
        [:Df] => Dict(:Ht => 0.3, :Hf => 0.7)
    ))

    cn = CredalNetwork([F, B, L, D, H])
    badj = [Int64[], Int64[], [2], [1, 2], [4]]
    fadj = [[4], [3, 4], Int64[], [5], Int64[]]
    ne = 4
    dag = SimpleDiGraph{Int64}(ne, fadj, badj)
    name_index = Dict(:F => 2, :H => 5, :D => 4, :B => 1, :L => 3)
    @test isequal(cn.dag, dag)
    @test isequal(cn.name_to_index, name_index)
    @test issetequal(cn.nodes, [F, B, L, D, H])
end