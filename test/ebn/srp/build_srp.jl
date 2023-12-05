@testset "Build SRP" begin
    root1 = DiscreteRootNode(:X1, Dict(:y => 0.2, :n => 0.8), Dict(:y => [Parameter(1, :X1)], :n => [Parameter(0, :X1)]))
    root3 = ContinuousRootNode(:Y1, Normal(), ExactDiscretization([0, 0.2, 1]))

    functional1_parents = [root1, root3]
    model1 = [Model(df -> (df.X1 .^ 2) ./ 2 .- df.Y1, :fun1)]
    simulation1 = MonteCarlo(300)
    performance = df-> df.fun1 - 1
    functional1_node = DiscreteFunctionalNode(:F1, functional1_parents, model1, performance, simulation1)

    srp = EnhancedBayesianNetworks._build_structuralreliabilityproblem_node(functional1_node)

    srps = Dict(
        [:n] => EnhancedBayesianNetworks.StructuralReliabilityProblemPMF(model1, [Parameter(0, :X1), RandomVariable(Normal(), :Y1)], performance, simulation1),
        [:y] => EnhancedBayesianNetworks.StructuralReliabilityProblemPMF(model1, [Parameter(1, :X1), RandomVariable(Normal(), :Y1)], performance, simulation1),
    )

    @test srp.name == functional1_node.name
    @test issetequal(srp.parents, discrete_ancestors(functional1_node))
    @test isequal(srp.srps, srps)
end