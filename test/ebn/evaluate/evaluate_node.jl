@testset "Evaluation Node" begin
    root1 = DiscreteRootNode(:A, Dict(:a1 => 0.5, :a2 => 0.5), Dict(:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]))
    root2 = ContinuousRootNode(:B, Normal())

    @testset "Continuous Node" begin
        model = Model(df -> df.A .+ df.B, :C)
        sim = MonteCarlo(100_000)
        performance = df -> 2 .- df.C
        cont_functional = ContinuousFunctionalNode(:C, [model], sim)

        net = EnhancedBayesianNetwork([root1, root2, cont_functional])
        add_child!(net, root1, cont_functional)
        add_child!(net, root2, cont_functional)
        order!(net)

        evaluated = EnhancedBayesianNetworks._evaluate_node(net, cont_functional)

        @test isa(evaluated, ContinuousChildNode)
        @test evaluated.name == :C
        @test isa(evaluated.distribution[[:a1]], EmpiricalDistribution)
        @test isa(evaluated.distribution[[:a2]], EmpiricalDistribution)
        @test issetequal(get_parents(net, evaluated)[3], [root1, root2])
        @test evaluated.discretization == cont_functional.discretization
        @test isa(evaluated.additional_info[[:a1]], Dict{Symbol,DataFrame})
        @test size(evaluated.additional_info[[:a1]][:samples]) == (sim.n, 3)
        @test isa(evaluated.additional_info[[:a2]], Dict{Symbol,DataFrame})
        @test size(evaluated.additional_info[[:a2]][:samples]) == (sim.n, 3)

        model = Model(df -> df.B .+ 1, :C)
        sim = MonteCarlo(100_000)
        performance = df -> 2 .- df.C
        cont_functional = ContinuousFunctionalNode(:C, [model], sim)
        net = EnhancedBayesianNetwork([root1, root2, cont_functional])
        add_child!(net, root2, cont_functional)
        order!(net)
        evaluated = EnhancedBayesianNetworks._evaluate_node(net, cont_functional)

        @test isa(evaluated, ContinuousRootNode)
        @test evaluated.name == :C
        @test isa(evaluated.distribution, EmpiricalDistribution)
        @test isa(evaluated.distribution, EmpiricalDistribution)
        @test evaluated.discretization == ExactDiscretization(cont_functional.discretization.intervals)

        @testset "Imprecise Parents" begin
            ### ROOT
            r1 = DiscreteRootNode(:A, Dict(:a1 => 0.5, :a2 => 0.5), Dict(:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]))
            r2 = ContinuousRootNode(:B, (0.1, 0.3))
            r3 = ContinuousRootNode(:P, Uniform(-10, 10))
            m = Model(df -> df.A .+ df.B, :C)
            s = MonteCarlo(100_000)
            cont_f = ContinuousFunctionalNode(:C, [m], s)

            net = EnhancedBayesianNetwork([r1, r2, r3, cont_f])
            add_child!(net, r1, cont_f)
            add_child!(net, r2, cont_f)
            add_child!(net, r3, cont_f)
            order!(net)

            @test_throws ErrorException("node C is a continuousfunctionalnode with at least one parent with Interval or p-boxes in its distributions. No method for extracting failure probability p-box have been implemented yet") EnhancedBayesianNetworks._evaluate_node(net, cont_f)
        end
    end

    @testset "Discrete Node" begin

        model = Model(df -> df.A .+ df.B, :C)
        sim = MonteCarlo(100_000)
        performance = df -> 2 .- df.C
        disc_functional = DiscreteFunctionalNode(:C, [model], performance, sim)
        net = EnhancedBayesianNetwork([root1, root2, disc_functional])
        add_child!(net, root1, disc_functional)
        add_child!(net, root2, disc_functional)
        order!(net)

        evaluated = EnhancedBayesianNetworks._evaluate_node(net, disc_functional)

        @test evaluated.name == :C
        @test isapprox(evaluated.states[[:a1]][:safe_C], 0.84, atol=0.02)
        @test isapprox(evaluated.states[[:a1]][:fail_C], 0.16, atol=0.02)
        @test isapprox(evaluated.states[[:a2]][:safe_C], 0.5, atol=0.02)
        @test isapprox(evaluated.states[[:a2]][:fail_C], 0.5, atol=0.02)
        @test issetequal(get_parents(net, evaluated)[3], [root1, root2])
        @test evaluated.parameters == disc_functional.parameters
        @test isa(evaluated.additional_info[[:a1]], Dict{Symbol,Any})
        @test size(evaluated.additional_info[[:a1]][:samples]) == (sim.n, 3)
        @test isa(evaluated.additional_info[[:a1]][:cov], Real)
        @test isa(evaluated.additional_info[[:a2]], Dict{Symbol,Any})
        @test size(evaluated.additional_info[[:a2]][:samples]) == (sim.n, 3)
        @test isa(evaluated.additional_info[[:a1]][:cov], Real)

        model = Model(df -> df.B .+ 1, :C)
        sim = MonteCarlo(100_000)
        performance = df -> 2 .- df.C
        disc_functional = DiscreteFunctionalNode(:C, [model], performance, sim)
        nodes = [root1, root2, disc_functional]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, root2, disc_functional)
        order!(net)

        evaluated = EnhancedBayesianNetworks._evaluate_node(net, disc_functional)

        @test isa(evaluated, DiscreteRootNode)
        @test evaluated.name == :C
        @test isapprox(evaluated.states[:safe_C], 0.84; atol=0.1)
        @test isapprox(evaluated.states[:fail_C], 0.15; atol=0.1)
        @test evaluated.parameters == disc_functional.parameters

        ##FORM
        model = Model(df -> 1 .+ df.B, :C)
        sim = FORM()
        performance = df -> 2 .- df.C
        disc_functional = DiscreteFunctionalNode(:C, [model], performance, sim)
        nodes = [root1, root2, disc_functional]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, root2, disc_functional)
        order!(net)

        evaluated = EnhancedBayesianNetworks._evaluate_node(net, disc_functional)

        @test isa(evaluated, DiscreteRootNode)
        @test evaluated.name == :C
        @test isapprox(evaluated.states[:safe_C], 0.84; atol=0.1)
        @test isapprox(evaluated.states[:fail_C], 0.15; atol=0.1)
        @test evaluated.parameters == disc_functional.parameters

        @testset "Imprecise Parents" begin
            ### ROOT
            interval = (0.10, 1.30)
            root1 = DiscreteRootNode(:A, Dict(:a1 => 0.5, :a2 => 0.5), Dict(:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]))
            root2 = ContinuousRootNode(:B, interval)
            root3 = ContinuousRootNode(:P, Uniform(-10, 10))
            model = Model(df -> df.A .+ df.B .+ df.P, :C)
            sim = DoubleLoop(MonteCarlo(100_000))
            performance = df -> 2 .- df.C
            disc_functional = DiscreteFunctionalNode(:C, [model], performance, sim)
            nodes = [root1, root2, root3, disc_functional]
            net = EnhancedBayesianNetwork(nodes)
            add_child!(net, root1, disc_functional)
            add_child!(net, root2, disc_functional)
            add_child!(net, root3, disc_functional)
            order!(net)
            evaluated = EnhancedBayesianNetworks._evaluate_node(net, disc_functional)

            @test evaluated.name == :C
            @test issetequal(get_parents(net, evaluated)[3], [root1, root2, root3])
            @test isempty(evaluated.additional_info[[:a1]])
            @test isempty(evaluated.additional_info[[:a2]])
            @test evaluated.parameters == disc_functional.parameters
            @test isapprox(evaluated.states[[:a1]][:safe_C][1], 0.482, atol=0.2)
            @test isapprox(evaluated.states[[:a1]][:safe_C][2], 0.496, atol=0.2)
            @test isapprox(evaluated.states[[:a2]][:safe_C][1], 0.436, atol=0.2)
            @test isapprox(evaluated.states[[:a2]][:safe_C][2], 0.445, atol=0.2)
            @test isapprox(evaluated.states[[:a1]][:fail_C][1], 0.518, atol=0.2)
            @test isapprox(evaluated.states[[:a1]][:fail_C][2], 0.504, atol=0.2)
            @test isapprox(evaluated.states[[:a2]][:fail_C][1], 0.564, atol=0.2)
            @test isapprox(evaluated.states[[:a2]][:fail_C][2], 0.555, atol=0.2)

            ss_infinity_adaptive = SubSetInfinityAdaptive(2000, 0.1, 10, 10)
            mc = MonteCarlo(10^4)
            sim = RandomSlicing(ss_infinity_adaptive, mc)
            performance = df -> 2 .- df.C
            disc_functional = DiscreteFunctionalNode(:C, [model], performance, sim)
            nodes = [root1, root2, root3, disc_functional]
            net = EnhancedBayesianNetwork(nodes)
            add_child!(net, root1, disc_functional)
            add_child!(net, root2, disc_functional)
            add_child!(net, root3, disc_functional)
            order!(net)

            evaluated = EnhancedBayesianNetworks._evaluate_node(net, disc_functional)

            @test evaluated.name == :C
            @test issetequal(get_parents(net, evaluated)[3], [root1, root2, root3])
            @test evaluated.parameters == disc_functional.parameters
            @test isa(evaluated.additional_info[[:a1]][:lb], Tuple{Float64,DataFrame})
            @test isa(evaluated.additional_info[[:a1]][:ub], Tuple{Float64,DataFrame})
            @test isa(evaluated.additional_info[[:a2]][:lb], Tuple{Float64,DataFrame})
            @test isa(evaluated.additional_info[[:a2]][:ub], Tuple{Float64,DataFrame})
            @test isapprox(evaluated.states[[:a1]][:safe_C][1], 0.482, atol=0.2)
            @test isapprox(evaluated.states[[:a1]][:safe_C][2], 0.496, atol=0.2)
            @test isapprox(evaluated.states[[:a2]][:safe_C][1], 0.436, atol=0.2)
            @test isapprox(evaluated.states[[:a2]][:safe_C][2], 0.445, atol=0.2)
            @test isapprox(evaluated.states[[:a1]][:fail_C][1], 0.518, atol=0.2)
            @test isapprox(evaluated.states[[:a1]][:fail_C][2], 0.504, atol=0.2)
            @test isapprox(evaluated.states[[:a2]][:fail_C][1], 0.564, atol=0.2)
            @test isapprox(evaluated.states[[:a2]][:fail_C][2], 0.555, atol=0.2)

            ### CHILD
            root1 = DiscreteRootNode(:A, Dict(:a1 => 0.5, :a2 => 0.5), Dict(:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]))
            root2 = ContinuousRootNode(:B, Normal())
            root3 = DiscreteRootNode(:D, Dict(:d1 => 0.5, :d2 => 0.5), Dict(:d1 => [Parameter(1, :D)], :d2 => [Parameter(2, :D)]))
            states = Dict(
                [:a1] => (0.1, 0.3),
                [:a2] => (0.7, 0.8)
            )
            child = ContinuousChildNode(:C1, states)

            model = Model(df -> df.D .+ df.C1 .+ df.B, :C2)
            performance = df -> 2 .- df.C2
            parents = [root3, root2, child]
            model_node = DiscreteFunctionalNode(:F1, [model], performance, sim)

            nodes = [root1, root2, root3, child, model_node]
            net = EnhancedBayesianNetwork(nodes)
            add_child!(net, root1, child)
            add_child!(net, root2, model_node)
            add_child!(net, root3, model_node)
            add_child!(net, child, model_node)
            order!(net)

            evaluated = EnhancedBayesianNetworks._evaluate_node(net, model_node)

            @test evaluated.name == :F1
            @test issetequal(get_parents(net, evaluated)[3], [root2, root3, child])
            @test evaluated.parameters == disc_functional.parameters

            @test isapprox(evaluated.states[[:d1, :a1]][:safe_F1][1], 0.755, atol=0.2)
            @test isapprox(evaluated.states[[:d1, :a1]][:safe_F1][2], 0.818, atol=0.2)
            @test isapprox(evaluated.states[[:d1, :a2]][:safe_F1][1], 0.577, atol=0.2)
            @test isapprox(evaluated.states[[:d1, :a2]][:safe_F1][2], 0.623, atol=0.2)
            @test isapprox(evaluated.states[[:d2, :a1]][:safe_F1][1], 0.380, atol=0.2)
            @test isapprox(evaluated.states[[:d2, :a1]][:safe_F1][2], 0.462, atol=0.2)
            @test isapprox(evaluated.states[[:d2, :a2]][:safe_F1][1], 0.209, atol=0.2)
            @test isapprox(evaluated.states[[:d2, :a2]][:safe_F1][2], 0.242, atol=0.2)

            @test isapprox(evaluated.states[[:d1, :a1]][:fail_F1][1], 0.245, atol=0.2)
            @test isapprox(evaluated.states[[:d1, :a1]][:fail_F1][2], 0.182, atol=0.2)
            @test isapprox(evaluated.states[[:d1, :a2]][:fail_F1][1], 0.423, atol=0.2)
            @test isapprox(evaluated.states[[:d1, :a2]][:fail_F1][2], 0.377, atol=0.2)
            @test isapprox(evaluated.states[[:d2, :a1]][:fail_F1][1], 0.620, atol=0.2)
            @test isapprox(evaluated.states[[:d2, :a1]][:fail_F1][2], 0.538, atol=0.2)
            @test isapprox(evaluated.states[[:d2, :a2]][:fail_F1][1], 0.791, atol=0.2)
            @test isapprox(evaluated.states[[:d2, :a2]][:fail_F1][2], 0.758, atol=0.2)
        end
    end
end
