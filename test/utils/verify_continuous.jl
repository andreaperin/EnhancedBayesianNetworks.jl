@testset "Verify Continuous" begin
    cpt1 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
    cpt1[] = Normal()
    discretization = ApproximatedDiscretization([-1, 0, 1], 2)
    @test_throws ErrorException("Root node must have ExactDiscretization as discretization structure, provided discretization is $discretization and node cpt is $cpt1") EnhancedBayesianNetworks._verify_discretization(cpt1, discretization)

    cpt2 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}([:x])
    cpt2[:x=>:yesx] = Normal()
    cpt2[:x=>:nox] = Normal(1, 2)
    discretization = ExactDiscretization([-1, 0, 1])

    @test_throws ErrorException("Child node must have ExactDiscretization as discretization structure, provided discretization is $discretization and node cpt is $cpt2") EnhancedBayesianNetworks._verify_discretization(cpt2, discretization)

end