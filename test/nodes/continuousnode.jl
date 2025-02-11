@testset "Continuous Nodes" begin

    @testset "Root Node" begin
        @testset "Nodes and Verifications" begin
            name = :A
            cpt1 = DataFrame(:Π => Normal())
            discretization = ApproximatedDiscretization([-1, 0, 1], 2)
            @test_throws ErrorException("Root node must have ExactDiscretization as discretization structure, provided discretization is $discretization and node cpt is $cpt1") ContinuousNode{UnivariateDistribution}(name, cpt1, discretization)
            node1 = ContinuousNode{UnivariateDistribution}(name, cpt1)
            node2 = ContinuousNode(name, cpt1, ExactDiscretization([-1, 0, 1]))

            @test node1.name == name
            @test node2.name == name
            @test node1.cpt == cpt1
            @test node2.cpt == cpt1
            @test isempty(node1.discretization.intervals)
            @test node2.discretization == ExactDiscretization([-1, 0, 1])
            @test isempty(node1.additional_info)
            @test isempty(node2.additional_info)

            cpt2 = DataFrame(:Π => (1, 5))
            node3 = ContinuousNode{Tuple{Real,Real}}(name, cpt2)
            @test node3.name == name
            @test node3.cpt == cpt2
            @test isempty(node3.discretization.intervals)
            @test isempty(node3.additional_info)

            cpt3 = DataFrame(:Π => UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)]))
            node3 = ContinuousNode{UnamedProbabilityBox}(name, cpt3)
            @test node3.name == name
            @test node3.cpt == cpt3
            @test isempty(node3.discretization.intervals)
            @test isempty(node3.additional_info)

            @test_throws ErrorException("continuous node A has a parameter UnamedProbabilityBox not coherent with the type of distribution Tuple{Real, Real}") ContinuousNode{UnamedProbabilityBox}(name, cpt2)
        end

        @testset "functions" begin
            name = :A
            cpt1 = DataFrame(:Π => Normal())
            cpt2 = DataFrame(:Π => (1, 5))
            cpt3 = DataFrame(:Π => UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)]))

            node1 = ContinuousNode{UnivariateDistribution}(name, cpt1)
            node2 = ContinuousNode{Tuple{Real,Real}}(name, cpt2)
            node3 = ContinuousNode{UnamedProbabilityBox}(name, cpt3)

            @test EnhancedBayesianNetworks._distributions(cpt1) == [Normal()]
            @test EnhancedBayesianNetworks._distributions(cpt2) == [(1, 5)]
            @test EnhancedBayesianNetworks._distributions(cpt3) == [UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)])]
            @test EnhancedBayesianNetworks._distributions(node1) == [Normal()]
            @test EnhancedBayesianNetworks._distributions(node2) == [(1, 5)]
            @test EnhancedBayesianNetworks._distributions(node3) == [UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)])]

            @test isempty(EnhancedBayesianNetworks._scenarios(cpt1))
            @test isempty(EnhancedBayesianNetworks._scenarios(cpt2))
            @test isempty(EnhancedBayesianNetworks._scenarios(cpt3))
            @test isempty(EnhancedBayesianNetworks._scenarios(node1))
            @test isempty(EnhancedBayesianNetworks._scenarios(node2))
            @test isempty(EnhancedBayesianNetworks._scenarios(node3))

            evidence = Evidence()
            @test EnhancedBayesianNetworks._continuous_input(node1, evidence) == [RandomVariable(Normal(), :A)]
            @test EnhancedBayesianNetworks._continuous_input(node2, evidence) == [Interval(1, 5, :A)]
            @test EnhancedBayesianNetworks._continuous_input(node3, evidence)[1].name == :A
            @test EnhancedBayesianNetworks._continuous_input(node3, evidence)[1].lb == -Inf
            @test EnhancedBayesianNetworks._continuous_input(node3, evidence)[1].ub == Inf
            @test EnhancedBayesianNetworks._continuous_input(node3, evidence)[1].parameters == [Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)]

            @test EnhancedBayesianNetworks._continuous_input(node1) == [RandomVariable(Normal(), :A)]
            @test EnhancedBayesianNetworks._continuous_input(node2) == [Interval(1, 5, :A)]
            @test EnhancedBayesianNetworks._continuous_input(node3)[1].name == :A
            @test EnhancedBayesianNetworks._continuous_input(node3)[1].lb == -Inf
            @test EnhancedBayesianNetworks._continuous_input(node3)[1].ub == Inf
            @test EnhancedBayesianNetworks._continuous_input(node3)[1].parameters == [Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)]

            dist1 = Normal()
            dist2 = (1, 5)
            dist3 = UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)])

            @test EnhancedBayesianNetworks._distribution_bounds(dist1) == [-Inf, Inf]
            @test EnhancedBayesianNetworks._distribution_bounds(dist2) == [1, 5]
            @test EnhancedBayesianNetworks._distribution_bounds(dist3) == [-Inf, Inf]

            @test EnhancedBayesianNetworks._distribution_bounds(node1) == [-Inf, Inf]
            @test EnhancedBayesianNetworks._distribution_bounds(node2) == [1, 5]
            @test EnhancedBayesianNetworks._distribution_bounds(node3) == [-Inf, Inf]

            @test EnhancedBayesianNetworks._truncate(dist1, [-1, 1]) == truncated(Normal(), -1, 1)
            @test EnhancedBayesianNetworks._truncate(dist2, [-1, 1]) == (-1, 1)
            @test EnhancedBayesianNetworks._truncate(dist3, [-1, 1]).lb == -1
            @test EnhancedBayesianNetworks._truncate(dist3, [-1, 1]).ub == 1

            @test EnhancedBayesianNetworks._is_precise(node1)
            @test EnhancedBayesianNetworks._is_precise(node2) == false
            @test EnhancedBayesianNetworks._is_precise(node3) == false

            @test EnhancedBayesianNetworks._is_continuous_root(cpt1)
            @test EnhancedBayesianNetworks._is_continuous_root(cpt2)
            @test EnhancedBayesianNetworks._is_continuous_root(cpt3)

            @test EnhancedBayesianNetworks._is_root(node1)
            @test EnhancedBayesianNetworks._is_root(node2)
            @test EnhancedBayesianNetworks._is_root(node3)
        end
    end

    @testset "Child Node" begin
        @testset "Nodes and Verification" begin
            name = :B
            cpt1 = DataFrame(:G => [:g1, :g2], :Π => [Normal(0, 1), Normal(1, 1)])
            discretization = ExactDiscretization([-1, 0, 1])
            @test_throws ErrorException("Child node must have ExactDiscretization as discretization structure, provided discretization is $discretization and node cpt is $cpt1") ContinuousNode{UnivariateDistribution}(name, cpt1, discretization)

            discretization = ApproximatedDiscretization([-1, 0, 1], 2)
            node1 = ContinuousNode{UnivariateDistribution}(name, cpt1)
            node2 = ContinuousNode(name, cpt1, discretization)
            @test node1.name == name
            @test node2.name == name
            @test node1.cpt == cpt1
            @test node2.cpt == cpt1
            @test isempty(node1.discretization.intervals)
            @test node2.discretization == discretization
            @test isempty(node1.additional_info)
            @test isempty(node2.additional_info)

            cpt2 = DataFrame(:G => [:g1, :g2], :Π => [(0, 1), (1, 1)])
            node3 = ContinuousNode{Tuple{Real,Real}}(name, cpt2)
            @test node3.name == name
            @test node3.cpt == cpt2
            @test isempty(node3.discretization.intervals)
            @test isempty(node3.additional_info)

            pb1 = UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)])
            pb2 = UnamedProbabilityBox{Normal}([Interval(-1, 0, :μ), Interval(1, 2, :σ)])
            cpt3 = DataFrame(:G => [:g1, :g2], :Π => [pb1, pb2])
            node3 = ContinuousNode{UnamedProbabilityBox}(name, cpt3)
            @test node3.name == name
            @test node3.cpt == cpt3
            @test isempty(node3.discretization.intervals)
            @test isempty(node3.additional_info)
        end

        @testset "functions" begin
            name = :A
            cpt1 = DataFrame(:G => [:g1, :g2], :Π => [Normal(0, 1), Normal(1, 1)])
            cpt2 = DataFrame(:G => [:g1, :g2], :Π => [(0, 1), (1, 1)])
            pb1 = UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)])
            pb2 = UnamedProbabilityBox{Normal}([Interval(-1, 0, :μ), Interval(1, 2, :σ)])
            cpt3 = DataFrame(:G => [:g1, :g2], :Π => [pb1, pb2])

            node1 = ContinuousNode{UnivariateDistribution}(name, cpt1)
            node2 = ContinuousNode{Tuple{Real,Real}}(name, cpt2)
            node3 = ContinuousNode{UnamedProbabilityBox}(name, cpt3)

            @test EnhancedBayesianNetworks._distributions(cpt1) == [Normal(), Normal(1, 1)]
            @test EnhancedBayesianNetworks._distributions(cpt2) == [(0, 1), (1, 1)]
            @test EnhancedBayesianNetworks._distributions(cpt3) == [pb1, pb2]
            @test EnhancedBayesianNetworks._distributions(node1) == [Normal(), Normal(1, 1)]
            @test EnhancedBayesianNetworks._distributions(node2) == [(0, 1), (1, 1)]
            @test EnhancedBayesianNetworks._distributions(node3) == [pb1, pb2]

            @test EnhancedBayesianNetworks._scenarios(cpt1) == [Dict(:G => :g1), Dict(:G => :g2)]
            @test EnhancedBayesianNetworks._scenarios(cpt2) == [Dict(:G => :g1), Dict(:G => :g2)]
            @test EnhancedBayesianNetworks._scenarios(cpt3) == [Dict(:G => :g1), Dict(:G => :g2)]
            @test EnhancedBayesianNetworks._scenarios(node1) == [Dict(:G => :g1), Dict(:G => :g2)]
            @test EnhancedBayesianNetworks._scenarios(node2) == [Dict(:G => :g1), Dict(:G => :g2)]
            @test EnhancedBayesianNetworks._scenarios(node3) == [Dict(:G => :g1), Dict(:G => :g2)]

            @test EnhancedBayesianNetworks._continuous_input(node1) == [RandomVariable(Normal(), :A), RandomVariable(Normal(1, 1), :A)]
            @test EnhancedBayesianNetworks._continuous_input(node2) == [Interval(0, 1, :A), Interval(1, 1, :A)]
            @test EnhancedBayesianNetworks._continuous_input(node3)[1].name == :A
            @test EnhancedBayesianNetworks._continuous_input(node3)[1].lb == -Inf
            @test EnhancedBayesianNetworks._continuous_input(node3)[1].ub == Inf
            @test EnhancedBayesianNetworks._continuous_input(node3)[1].parameters == [Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)]

            evidence1 = Evidence()
            @test EnhancedBayesianNetworks._continuous_input(node1, evidence1) == [RandomVariable(Normal(), :A), RandomVariable(Normal(1, 1), :A)]
            @test EnhancedBayesianNetworks._continuous_input(node2, evidence1) == [Interval(0, 1, :A), Interval(1, 1, :A)]
            @test EnhancedBayesianNetworks._continuous_input(node3, evidence1)[1].name == :A
            @test EnhancedBayesianNetworks._continuous_input(node3, evidence1)[1].lb == -Inf
            @test EnhancedBayesianNetworks._continuous_input(node3, evidence1)[1].ub == Inf
            @test EnhancedBayesianNetworks._continuous_input(node3, evidence1)[1].parameters == [Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)]

            evidence2 = Evidence(:G => :g1)
            @test EnhancedBayesianNetworks._continuous_input(node3, evidence1)[2].name == :A
            @test EnhancedBayesianNetworks._continuous_input(node3, evidence1)[2].lb == -Inf
            @test EnhancedBayesianNetworks._continuous_input(node3, evidence1)[2].ub == Inf
            @test EnhancedBayesianNetworks._continuous_input(node3, evidence1)[2].parameters == [Interval(-1, 0, :μ), Interval(1, 2, :σ)]

            @test EnhancedBayesianNetworks._continuous_input(node1, evidence2) == [RandomVariable(Normal(), :A)]
            @test EnhancedBayesianNetworks._continuous_input(node2, evidence2) == [Interval(0, 1, :A)]
            @test EnhancedBayesianNetworks._continuous_input(node3, evidence2)[1].name == :A
            @test EnhancedBayesianNetworks._continuous_input(node3, evidence2)[1].lb == -Inf
            @test EnhancedBayesianNetworks._continuous_input(node3, evidence2)[1].ub == Inf
            @test EnhancedBayesianNetworks._continuous_input(node3, evidence2)[1].parameters == [Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)]

            @test EnhancedBayesianNetworks._distribution_bounds(node1) == [-Inf, Inf]
            @test EnhancedBayesianNetworks._distribution_bounds(node2) == [0, 1]
            @test EnhancedBayesianNetworks._distribution_bounds(node3) == [-Inf, Inf]

            @test EnhancedBayesianNetworks._is_precise(node1)
            @test EnhancedBayesianNetworks._is_precise(node2) == false
            @test EnhancedBayesianNetworks._is_precise(node3) == false

            @test EnhancedBayesianNetworks._is_root(node1) == false
            @test EnhancedBayesianNetworks._is_root(node2) == false
            @test EnhancedBayesianNetworks._is_root(node3) == false
        end
    end
end