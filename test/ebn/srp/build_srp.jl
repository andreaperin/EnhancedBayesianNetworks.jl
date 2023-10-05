@testset "Build Structural Reliability Problem" begin
    root1 = DiscreteRootNode(:X1, Dict(:y => 0.2, :n => 0.8), Dict(:y => [Parameter(1, :X1)], :n => [Parameter(0, :X1)]))
    root3 = ContinuousRootNode(:Y1, Normal(), ExactDiscretization([0, 0.2, 1]))

    functional1_parents = [root1, root3]
    disc_D = ApproximatedDiscretization([-1.1, 0, 0.11], 2)
    model1 = [Model(df -> (df.X1 .^ 2) ./ 2 .- df.Y1, :fun1)]
    simulation1 = MonteCarlo(300)
    functional1_node = ContinuousFunctionalNode(:F1, functional1_parents, model1, simulation1, disc_D)

    srp = EnhancedBayesianNetworks._build_structuralreliabilityproblem_node(functional1_node)

    srps = Dict(
        [:n] => EnhancedBayesianNetworks.StructuralReliabilityProblemPDF(model1, [Parameter(0, :X1), RandomVariable(Normal(), :Y1)], simulation1),
        [:y] => EnhancedBayesianNetworks.StructuralReliabilityProblemPDF(model1, [Parameter(1, :X1), RandomVariable(Normal(), :Y1)], simulation1),
    )

    @test srp.name == functional1_node.name
    @test isequal(srp.discretization, functional1_node.discretization)
    @test issetequal(srp.parents, get_discrete_ancestors(functional1_node))
    @test isequal(srp.srps, srps)
end