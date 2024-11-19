@testset "Continuous Nodes" begin

    @testset "Root Node" begin
        name = :A
        cpt1 = DataFrame(:Prob => Normal())
        node1 = ContinuousNode{UnivariateDistribution}(name, cpt1)
        node2 = ContinuousNode{UnivariateDistribution}(name, cpt1, ExactDiscretization([-1, 0, 1]))

        @test node1.name == name
        @test node2.name == name
        @test node1.cpt == cpt1
        @test node2.cpt == cpt1
        @test isempty(node1.discretization.intervals)
        @test node2.discretization == ExactDiscretization([-1, 0, 1])
        @test isempty(node1.additional_info)
        @test isempty(node2.additional_info)

        cpt2 = DataFrame(:Prob => (1, 5))
        node3 = ContinuousNode{Tuple{Real,Real}}(name, cpt2)
        @test node3.name == name
        @test node3.cpt == cpt2
        @test isempty(node3.discretization.intervals)
        @test isempty(node3.additional_info)

        cpt3 = DataFrame(:Prob => UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)]))
        node3 = ContinuousNode{UnamedProbabilityBox}(name, cpt2)
        @test node3.name == name
        @test node3.cpt == cpt2
        @test isempty(node3.discretization.intervals)
        @test isempty(node3.additional_info)



    end
    @testset "Child Node" begin
        name = :B
        cpt_b = DataFrame(:G => [:g1, :g2], :Prob => [Normal(0, 1), Normal(1, 1)])
        node1 = ContinuousNode{UnivariateDistribution}(name, cpt_b)
        node2 = ContinuousNode{UnivariateDistribution}(name, cpt_b, ExactDiscretization([-1, 0, 1]))

    end
    # name = :B
    # cpt_b = DataFrame(:G => [:g1, :g2], :Prob => [Normal(0, 1), Normal(1, 1)])
    # node2 = ContinuousNode{UnivariateDistribution}(name, cpt_b)

    # evidence = Evidence(:G => :g1)
    # EnhancedBayesianNetworks._continuous_input(node2, evidence)
    # EnhancedBayesianNetworks._is_precise(node2)

    # name = :B
    # cpt_b = DataFrame(:G => [:g1, :g2], :Prob => [(0, 1), (1, 2)])
    # node2 = ContinuousNode{Tuple{Real,Real}}(name, cpt_b)

    # evidence = Evidence()
    # EnhancedBayesianNetworks._continuous_input(node2)
    # EnhancedBayesianNetworks._is_precise(node2)

    # name = :B
    # normal_pbox1 = UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)])
    # normal_pbox2 = UnamedProbabilityBox{Uniform}([Interval(-2, -1, :a), Interval(1, 2, :b)])

    # cpt_b = DataFrame(:G => [:g1, :g2], :Prob => [normal_pbox1, normal_pbox2])

    # node2 = ContinuousNode{UnamedProbabilityBox}(name, cpt_b)

    # evidence = Evidence(:G => :g1)
    # EnhancedBayesianNetworks._continuous_input(node2, evidence)


    # A = DiscreteNode(:A, DataFrame(:A => [:a1, :a2], :Prob => [0.5, 0.5]))
    # B = DiscreteNode(:B, DataFrame(:B => [:b1, :b2], :Prob => [0.1, 0.2]))
    # cpt_c = DataFrame(:B => [:b1, :b1, :b1, :b1, :b2, :b2, :b2, :b2], :A => [:a1, :a1, :a2, :a2, :a1, :a1, :a2, :a2], :C => [:c1, :c2, :c1, :c2, :c1, :c2, :c1, :c2], :Prob => [0.8, 0.2, 0.6, 0.4, 0.8, 0.2, 0.6, 0.4])
    # C = DiscreteNode(:C, cpt_c, Dict(:c1 => [Parameter(1, :C)], :c2 => [Parameter(2, :C)]))
end