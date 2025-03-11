@testset "Continuous Nodes" begin

    @testset "Root Node" begin
        name = :A
        cpt1 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
        cpt1[] = Normal()
        discretization = ApproximatedDiscretization([-1, 0, 1], 2)
        add_info = Dict{Vector{Symbol},Dict}()
        @test_throws ErrorException("Root node must have ExactDiscretization as discretization structure, provided discretization is $discretization and node cpt is $cpt1") ContinuousNode(name, cpt1, discretization, add_info)
        node1 = ContinuousNode(name, cpt1)
        node2 = ContinuousNode(name, cpt1, ExactDiscretization([-1, 0, 1]))

        @test node1.name == name
        @test node2.name == name
        @test node1.cpt == cpt1
        @test node2.cpt == cpt1
        @test isempty(node1.discretization.intervals)
        @test node2.discretization == ExactDiscretization([-1, 0, 1])
        @test isempty(node1.additional_info)
        @test isempty(node2.additional_info)

        @test distributions(node2) == node2.cpt.data.Π
        @test scenarios(node2) == Any[]
        @test isprecise(node2)
        @test isroot(node2)

        cpt2_1 = DataFrame(:Π => (1, 5))
        cpt2 = ContinuousConditionalProbabilityTable{ImpreciseContinuousInput}(cpt2_1)
        node3 = ContinuousNode(name, cpt2)
        @test node3.name == name
        @test node3.cpt == cpt2
        @test isempty(node3.discretization.intervals)
        @test isempty(node3.additional_info)

        cpt3_1 = DataFrame(:Π => UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)]))
        cpt3 = ContinuousConditionalProbabilityTable{ImpreciseContinuousInput}(cpt3_1)

        node4 = ContinuousNode(name, cpt3)
        @test node4.name == name
        @test node4.cpt == cpt3
        @test isempty(node4.discretization.intervals)
        @test isempty(node4.additional_info)

        @test distributions(node4) == node4.cpt.data.Π
        @test scenarios(node4) == Any[]
        @test !isprecise(node4)
        @test isroot(node4)

    end

    @testset "Child Node" begin
        name = :A
        cpt1 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}(:x)
        cpt1[:x=>:x1] = Normal(1, 1)
        cpt1[:x=>:x2] = Normal(1, 2)
        discretization = ExactDiscretization([-1, 0, 1])
        add_info = Dict{Vector{Symbol},Dict}()
        @test_throws ErrorException("Child node must have ApproximatedDiscretization as discretization structure, provided discretization is $discretization and node cpt is $cpt1") ContinuousNode(name, cpt1, discretization, add_info)

        node1 = ContinuousNode(name, cpt1)
        node2 = ContinuousNode(name, cpt1, ApproximatedDiscretization([-1, 0, 1], 2))

        @test node1.name == name
        @test node2.name == name
        @test node1.cpt == cpt1
        @test node2.cpt == cpt1
        @test isempty(node1.discretization.intervals)
        @test node2.discretization == ApproximatedDiscretization([-1, 0, 1], 2)
        @test isempty(node1.additional_info)
        @test isempty(node2.additional_info)

        @test distributions(node2) == node2.cpt.data.Π
        @test scenarios(node2) == [Dict(:x => :x1), Dict(:x => :x2)]
        @test isprecise(node2)
        @test !isroot(node2)

        cpt2_1 = DataFrame(:x => [:x1, :x2], :Π => [(1, 5), (2, 6)])
        cpt2 = ContinuousConditionalProbabilityTable{ImpreciseContinuousInput}(cpt2_1)
        node3 = ContinuousNode(name, cpt2)
        @test node3.name == name
        @test node3.cpt == cpt2
        @test isempty(node3.discretization.intervals)
        @test isempty(node3.additional_info)

        cpt3_1 = DataFrame(:x => [:x1, :x2], :Π => [UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)]), UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(4, 5, :σ)])])
        cpt3 = ContinuousConditionalProbabilityTable{ImpreciseContinuousInput}(cpt3_1)

        node4 = ContinuousNode(name, cpt3)
        @test node4.name == name
        @test node4.cpt == cpt3
        @test isempty(node4.discretization.intervals)
        @test isempty(node4.additional_info)

        @test distributions(node4) == node4.cpt.data.Π
        @test scenarios(node4) == [Dict(:x => :x1), Dict(:x => :x2)]
        @test !isprecise(node4)
        @test !isroot(node4)
    end

    @testset "uq inputs" begin

        cpt0 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}([:g, :s])
        cpt0[:g=>:g1, :s=>:s1] = Normal(1, 2)
        cpt0[:g=>:g1, :s=>:s2] = Normal(2, 3)
        cpt0[:g=>:g2, :s=>:s1] = Normal(1, 2)
        cpt0[:g=>:g2, :s=>:s2] = Normal(1, 4)
        node0 = ContinuousNode(:a, cpt0)

        cpt1 = ContinuousConditionalProbabilityTable{Tuple{Real,Real}}([:g, :s])
        cpt1[:g=>:g1, :s=>:s1] = (1, 2)
        cpt1[:g=>:g1, :s=>:s2] = (2, 3)
        cpt1[:g=>:g2, :s=>:s1] = (1, 2)
        cpt1[:g=>:g2, :s=>:s2] = (1, 4)
        node1 = ContinuousNode(:a, cpt1)

        cpt2 = ContinuousConditionalProbabilityTable{UnamedProbabilityBox}([:g, :s])
        cpt2[:g=>:g1, :s=>:s1] = UnamedProbabilityBox{Normal}([Interval(1, 2, :μ), Interval(1, 2, :σ)])
        cpt2[:g=>:g1, :s=>:s2] = UnamedProbabilityBox{Normal}([Interval(2, 3, :μ), Interval(3, 4, :σ)])
        cpt2[:g=>:g2, :s=>:s1] = UnamedProbabilityBox{Normal}([Interval(1, 2, :μ), Interval(3, 4, :σ)])
        cpt2[:g=>:g2, :s=>:s2] = UnamedProbabilityBox{Normal}([Interval(1, 4, :μ), Interval(3, 4, :σ)])
        node2 = ContinuousNode(:a, cpt2)

        evidence1 = Evidence(:g => :g1)
        evidence2 = Evidence(:g => :g1, :s => :s1, :h => :h1)

        @test issetequal(EnhancedBayesianNetworks._uq_inputs(node0, evidence1), [RandomVariable(Normal(1, 2), :a), RandomVariable(Normal(2, 3), :a)])
        @test EnhancedBayesianNetworks._uq_inputs(node0, evidence2) == [RandomVariable(Normal(1, 2), :a)]

        @test issetequal(EnhancedBayesianNetworks._uq_inputs(node1, evidence1), [Interval(1, 2, :a), Interval(2, 3, :a)])

        @test EnhancedBayesianNetworks._uq_inputs(node1, evidence2) == [Interval(1, 2, :a)]

        @test isa(EnhancedBayesianNetworks._uq_inputs(node2, evidence1), Vector{ProbabilityBox{Normal}})
        @test isa(EnhancedBayesianNetworks._uq_inputs(node2, evidence2), Vector{ProbabilityBox{Normal}})

        cpt3 = ContinuousConditionalProbabilityTable{ImpreciseContinuousInput}([:g, :s])
        cpt3[:g=>:g1, :s=>:s1] = (1, 2)
        cpt3[:g=>:g1, :s=>:s2] = (2, 3)
        cpt3[:g=>:g2, :s=>:s1] = (1, 2)
        cpt3[:g=>:g2, :s=>:s2] = (1, 4)
        node3 = ContinuousNode(:a, cpt3)
        @test EnhancedBayesianNetworks._uq_inputs(node3, evidence1) == EnhancedBayesianNetworks._uq_inputs(node1, evidence1)

        @test issetequal(EnhancedBayesianNetworks._uq_inputs(node3), [Interval(1, 2, :a), Interval(2, 3, :a), Interval(1, 2, :a), Interval(1, 4, :a)])
    end

end