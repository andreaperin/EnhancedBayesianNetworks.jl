@testset "Evaluation Node" begin
    cpt_root1 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:A])
    cpt_root1[:A=>:a1] = 0.5
    cpt_root1[:A=>:a2] = 0.5
    root1 = DiscreteNode(:A, cpt_root1, Dict(:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]))

    cpt_root2 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
    cpt_root2[] = Normal()
    root2 = ContinuousNode(:B, cpt_root2)

    cpt_root3 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:C])
    cpt_root3[:C=>:c1] = 0.5
    cpt_root3[:C=>:c2] = 0.5
    root3 = DiscreteNode(:C, cpt_root3, Dict(:c1 => [Parameter(1, :C)], :c2 => [Parameter(2, :C)]))

    @testset "Continuous Node" begin
        model = Model(df -> df.A .+ df.B .- df.C, :D)
        sim = MonteCarlo(100_000)
        performance = df -> 2 .- df.D
        cont_functional = ContinuousFunctionalNode(:D, [model], sim)

        net = EnhancedBayesianNetwork([root1, root2, root3, cont_functional])
        add_child!(net, root1, cont_functional)
        add_child!(net, root2, cont_functional)
        add_child!(net, root3, cont_functional)
        order!(net)

        evaluated = EnhancedBayesianNetworks._evaluate_node(net, cont_functional)

        @test isa(evaluated, ContinuousNode)
        @test isroot(evaluated) == false
        @test evaluated.name == :D
        @test typeof.(evaluated.cpt.data[!, :Π]) == [EmpiricalDistribution, EmpiricalDistribution, EmpiricalDistribution, EmpiricalDistribution]

        @test issetequal(parents(net, evaluated)[3], [root1, root2, root3])
        @test evaluated.discretization == cont_functional.discretization
        @test isa(evaluated.additional_info[[:a1, :c1]], Dict{Symbol,DataFrame})
        @test isa(evaluated.additional_info[[:a1, :c2]], Dict{Symbol,DataFrame})
        @test isa(evaluated.additional_info[[:a2, :c1]], Dict{Symbol,DataFrame})
        @test isa(evaluated.additional_info[[:a2, :c2]], Dict{Symbol,DataFrame})

        @test size(evaluated.additional_info[[:a1, :c1]][:samples]) == (sim.n, 4)
        @test size(evaluated.additional_info[[:a1, :c2]][:samples]) == (sim.n, 4)
        @test size(evaluated.additional_info[[:a2, :c1]][:samples]) == (sim.n, 4)
        @test size(evaluated.additional_info[[:a2, :c2]][:samples]) == (sim.n, 4)

        model = Model(df -> df.B .+ 1, :C)
        sim = MonteCarlo(100_000)
        performance = df -> 2 .- df.C
        cont_functional = ContinuousFunctionalNode(:C, [model], sim)
        net = EnhancedBayesianNetwork([root2, cont_functional])
        add_child!(net, root2, cont_functional)
        order!(net)
        evaluated = EnhancedBayesianNetworks._evaluate_node(net, cont_functional)

        @test isa(evaluated, ContinuousNode)
        @test isroot(evaluated)
        @test evaluated.name == :C
        @test typeof.(evaluated.cpt.data[!, :Π]) == [EmpiricalDistribution]
        @test evaluated.discretization == ExactDiscretization(cont_functional.discretization.intervals)

        @testset "Imprecise Parents" begin
            ### ROOT
            cpt_r1 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:A)
            cpt_r1[:A=>:a1] = 0.5
            cpt_r1[:A=>:a2] = 0.5
            r1 = DiscreteNode(:A, cpt_r1, Dict(:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]))

            cpt_r2 = ContinuousConditionalProbabilityTable{Tuple{<:Real,<:Real}}()
            cpt_r2[] = (0.1, 0.3)
            r2 = ContinuousNode(:B, cpt_r2)

            cpt_r3 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
            cpt_r3[] = Uniform(-10, 10)
            r3 = ContinuousNode(:P, cpt_r3)

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
        @test evaluated.cpt.data[!, :A] == [:a1, :a1, :a2, :a2]
        @test evaluated.cpt.data[!, :C] == [:C_fail, :C_safe, :C_fail, :C_safe]
        @test isapprox(evaluated.cpt.data[!, :Π], [0.16, 0.84, 0.5, 0.5]; atol=0.02)

        @test issetequal(parents(net, evaluated)[3], [root1, root2])
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
        nodes = [root2, disc_functional]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, root2, disc_functional)
        order!(net)

        evaluated = EnhancedBayesianNetworks._evaluate_node(net, disc_functional)

        @test isa(evaluated, DiscreteNode)
        @test isroot(evaluated)
        @test evaluated.name == :C

        @test evaluated.cpt.data[!, :C] == [:C_fail, :C_safe]
        @test isapprox(evaluated.cpt.data[!, :Π], [0.1587, 0.8413]; atol=0.1)
        @test evaluated.parameters == disc_functional.parameters

        ##FORM
        model = Model(df -> 1 .+ df.B, :C)
        sim = FORM()
        performance = df -> 2 .- df.C
        disc_functional = DiscreteFunctionalNode(:C, [model], performance, sim)
        nodes = [root2, disc_functional]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, root2, disc_functional)
        order!(net)

        evaluated = EnhancedBayesianNetworks._evaluate_node(net, disc_functional)

        @test isa(evaluated, DiscreteNode)
        @test isroot(evaluated)
        @test evaluated.name == :C

        @test evaluated.cpt.data[!, :C] == [:C_fail, :C_safe]
        @test isapprox(evaluated.cpt.data[!, :Π], [0.1587, 0.8413]; atol=0.1)
        @test evaluated.parameters == disc_functional.parameters

        @testset "Imprecise Parents" begin
            ### ROOT
            interval = (0.10, 1.30)
            cpt_root1 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:A)
            cpt_root1[:A=>:a1] = 0.5
            cpt_root1[:A=>:a2] = 0.5
            root1 = DiscreteNode(:A, cpt_root1, Dict(:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]))

            cpt_root2 = ContinuousConditionalProbabilityTable{Tuple{Real,Real}}()
            cpt_root2[] = interval
            root2 = ContinuousNode(:B, cpt_root2)

            cpt_root3 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
            cpt_root3[] = Uniform(-10, 10)
            root3 = ContinuousNode(:P, cpt_root3)

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
            @test issetequal(parents(net, evaluated)[3], [root1, root2, root3])
            @test isempty(evaluated.additional_info[[:a1]])
            @test isempty(evaluated.additional_info[[:a2]])
            @test evaluated.parameters == disc_functional.parameters

            @test evaluated.cpt.data[!, :A] == [:a1, :a1, :a2, :a2]
            @test evaluated.cpt.data[!, :C] == [:C_fail, :C_safe, :C_fail, :C_safe]
            @test all(isapprox.(evaluated.cpt.data[!, :Π][1], (0.45191, 0.51868); atol=0.2))
            @test all(isapprox.(evaluated.cpt.data[!, :Π][2], (0.48132, 0.54809); atol=0.2))
            @test all(isapprox.(evaluated.cpt.data[!, :Π][3], (0.50418, 0.56792); atol=0.2))
            @test all(isapprox.(evaluated.cpt.data[!, :Π][4], [0.43208, 0.49582]; atol=0.2))

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
            @test issetequal(parents(net, evaluated)[3], [root1, root2, root3])
            @test evaluated.parameters == disc_functional.parameters
            @test isa(evaluated.additional_info[[:a1]][:lb], Tuple{Float64,DataFrame})
            @test isa(evaluated.additional_info[[:a1]][:ub], Tuple{Float64,DataFrame})
            @test isa(evaluated.additional_info[[:a2]][:lb], Tuple{Float64,DataFrame})
            @test isa(evaluated.additional_info[[:a2]][:ub], Tuple{Float64,DataFrame})

            @test all(isapprox.(evaluated.cpt.data[!, :Π][1], (0.45191, 0.51868); atol=0.2))
            @test all(isapprox.(evaluated.cpt.data[!, :Π][2], (0.48132, 0.54809); atol=0.2))
            @test all(isapprox.(evaluated.cpt.data[!, :Π][3], (0.50418, 0.56792); atol=0.2))
            @test all(isapprox.(evaluated.cpt.data[!, :Π][4], (0.43208, 0.49582); atol=0.2))

            ### CHILD
            cpt_root1 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:A)
            cpt_root1[:A=>:a1] = 0.5
            cpt_root1[:A=>:a2] = 0.5
            root1 = DiscreteNode(:A, cpt_root1, Dict(:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]))

            cpt_root2 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
            cpt_root2[] = Normal()
            root2 = ContinuousNode(:B, cpt_root2)

            cpt_root3 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:D)
            cpt_root3[:D=>:d1] = 0.5
            cpt_root3[:D=>:d2] = 0.5
            root3 = DiscreteNode(:D, cpt_root3, Dict(:d1 => [Parameter(1, :D)], :d2 => [Parameter(2, :D)]))

            cpt_child = ContinuousConditionalProbabilityTable{Tuple{Real,Real}}(:A)
            cpt_child[:A=>:a1] = (0.1, 0.3)
            cpt_child[:A=>:a2] = (0.7, 0.8)
            child = ContinuousNode(:C1, cpt_child)

            model = Model(df -> df.D .+ df.C1 .+ df.B, :C2)
            performance = df -> 2 .- df.C2
            model_node = DiscreteFunctionalNode(:F1, [model], performance, sim)

            nodes = [root1, root2, root3, child, model_node]
            net = EnhancedBayesianNetwork(nodes)
            add_child!(net, root1, child)
            add_child!(net, root2, model_node)
            add_child!(net, root3, model_node)
            add_child!(net, child, model_node)
            order!(net)

            #! todo check the error here
            evaluated = EnhancedBayesianNetworks._evaluate_node(net, model_node)

            @test evaluated.name == :F1
            @test issetequal(parents(net, evaluated)[3], [root2, root3, child])
            @test evaluated.parameters == disc_functional.parameters

            @test evaluated.cpt.data[!, :A] == [:a1, :a1, :a2, :a2, :a1, :a1, :a2, :a2]
            @test evaluated.cpt.data[!, :D] == [:d1, :d1, :d1, :d1, :d2, :d2, :d2, :d2]
            @test evaluated.cpt.data[!, :F1] == [:F1_fail, :F1_safe, :F1_fail, :F1_safe, :F1_fail, :F1_safe, :F1_fail, :F1_safe]
            @test all(isapprox.(evaluated.cpt.data[!, :Π][1], (0.198, 0.2406); atol=0.2))
            @test all(isapprox.(evaluated.cpt.data[!, :Π][2], (0.759, 0.802); atol=0.2))
            @test all(isapprox.(evaluated.cpt.data[!, :Π][3], (0.371, 0.4248); atol=0.2))
            @test all(isapprox.(evaluated.cpt.data[!, :Π][4], (0.5752, 0.629); atol=0.2))
            @test all(isapprox.(evaluated.cpt.data[!, :Π][5], (0.537, 0.6224); atol=0.2))
            @test all(isapprox.(evaluated.cpt.data[!, :Π][6], (0.377, 0.463); atol=0.2))
            @test all(isapprox.(evaluated.cpt.data[!, :Π][7], (0.749, 0.7869); atol=0.2))
            @test all(isapprox.(evaluated.cpt.data[!, :Π][8], (0.213, 0.251); atol=0.2))
        end
    end
end