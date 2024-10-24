@testset "Enhanced Bayesian Networks" begin
    @testset "DiGraphFunctions" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
        root2 = DiscreteRootNode(:y, Dict(:y => 0.4, :n => 0.6))
        root3 = ContinuousRootNode(:z, Normal())

        name = :child
        parents = [root1, root2]
        distribution = Dict([:yes, :y] => Normal(), [:no, :y] => Normal(1, 1), [:yes, :n] => Normal(2, 1), [:no, :n] => Normal(3, 1))
        child_node = ContinuousChildNode(name, parents, distribution)

        nodes = [root1, root2, child_node]

        @test EnhancedBayesianNetworks._build_digraph(nodes) == SimpleDiGraph{Int64}(2, [[3], [3], Int64[]], [Int64[], Int64[], [1, 2]])

        dag, nodes, name_to_index = EnhancedBayesianNetworks._topological_ordered_dag(nodes)

        @test dag == SimpleDiGraph{Int64}(2, [[3], [3], Int64[]], [Int64[], Int64[], [1, 2]])

        @test issetequal(nodes, [root2, root1, child_node])

        @test name_to_index == Dict(:y => 1, :x => 2, :child => 3)
    end

    @testset "EnhancedBayesianNetwork" begin
        root1 = DiscreteRootNode(:x, Dict(:yes => 0.5, :no => 0.5))
        root1_1 = DiscreteRootNode(:x, Dict(:y => 0.5, :n => 0.5))
        root1_2 = DiscreteRootNode(:p, Dict(:yes => 0.5, :no => 0.5))

        states_child1 = Dict([:yes] => Dict(:a => 0.5, :b => 0.5), [:no] => Dict(:a => 0.5, :b => 0.5))
        child1 = DiscreteChildNode(:child1, [root1], states_child1, Dict(:a => [Parameter(3, :child1)], :b => [Parameter(0, :child1)]))

        distribution_child2 = Dict([:a] => Normal(), [:b] => Normal(2, 2))
        child2 = ContinuousChildNode(:child2, [child1], distribution_child2)

        model = Model(df -> sqrt.(df.child1 .^ 2 + df.child2 .^ 2), :value1)
        df -> 1 .- 2 .* df.v
        models = [model]
        simulation = MonteCarlo(200)
        performance = df -> 1 .- 2 .* df.v
        functional = DiscreteFunctionalNode(:functional, [child1, child2], models, performance, simulation)

        badjlist = Vector{Vector{Int}}([[], [1], [2], [2, 3]])
        fadjlist = Vector{Vector{Int}}([[2], [3, 4], [4], []])

        nodes = [root1, child1, child2, functional]
        @test_throws ErrorException("nodes must have different names") EnhancedBayesianNetwork([root1, root1_1, child1, child2, functional])

        @test_throws ErrorException("nodes state must have different symbols") EnhancedBayesianNetwork([root1, root1_2, child1, child2, functional])

        ebn = EnhancedBayesianNetwork(nodes)

        @test ebn.dag == EnhancedBayesianNetwork(DiGraph(4, fadjlist, badjlist), nodes, Dict(:x => 1, :child1 => 2, :child2 => 3, :functional => 4)).dag

        @test issetequal(ebn.nodes, EnhancedBayesianNetwork(DiGraph(4, fadjlist, badjlist), nodes, Dict(:x => 1, :child1 => 2, :child2 => 3, :functional => 4)).nodes)

        @test ebn.name_to_index == EnhancedBayesianNetwork(DiGraph(4, fadjlist, badjlist), nodes, Dict(:x => 1, :child1 => 2, :child2 => 3, :functional => 4)).name_to_index
    end

    @testset "Nodes Parental Logic" begin
        root1 = DiscreteRootNode(:x, Dict(:y => 0.5, :n => 0.5))
        root2 = DiscreteRootNode(:z, Dict(:yes => 0.2, :no => 0.8), Dict(:yes => [Parameter(3, :z)], :no => [Parameter(0, :z)]))

        states_child1 = Dict([:y] => Dict(:a => 0.5, :b => 0.5), [:n] => Dict(:a => 0.5, :b => 0.5))
        child1 = DiscreteChildNode(:child1, [root1], states_child1, Dict(:a => [Parameter(3, :child1)], :b => [Parameter(0, :child1)]))

        distribution_child2 = Dict([:a] => Normal(), [:b] => Normal(2, 2))
        child2 = ContinuousChildNode(:child2, [child1], distribution_child2)

        model = Model(df -> sqrt.(df.child1 .^ 2 + df.child2 .- df.z .^ 2), :value1)
        models = [model]
        simulation = MonteCarlo(400)
        performance = df -> 1 .- 2 .* df.v
        functional = DiscreteFunctionalNode(:functional, [child1, child2, root2], models, performance, simulation)

        ebn = EnhancedBayesianNetwork([root1, root2, child1, child2, functional])

        @test issetequal(get_parents(ebn, child1), [root1])
        @test issetequal(get_children(ebn, child2), [functional])
        @test issetequal(get_neighbors(ebn, child2), [child1, functional])
    end
    @testset "Markov Blanket" begin
        x1 = DiscreteRootNode(:x1, Dict(:x1y => 0.5, :x1n => 0.5))
        x2 = DiscreteRootNode(:x2, Dict(:x2y => 0.5, :x2n => 0.5))
        x4 = DiscreteRootNode(:x4, Dict(:x4y => 0.5, :x4n => 0.5))
        x8 = DiscreteRootNode(:x8, Dict(:x8y => 0.5, :x8n => 0.5))
        x3_states = Dict(
            [:x1y] => Dict(:x3y => 0.5, :x3n => 0.5),
            [:x1n] => Dict(:x3y => 0.5, :x3n => 0.5)
        )
        x3 = DiscreteChildNode(:x3, [x1], x3_states)
        x5_states = Dict(
            [:x2y] => Dict(:x5y => 0.5, :x5n => 0.5),
            [:x2n] => Dict(:x5y => 0.5, :x5n => 0.5)
        )
        x5 = DiscreteChildNode(:x5, [x2], x5_states)
        x7_states = Dict(
            [:x4y] => Dict(:x7y => 0.5, :x7n => 0.5),
            [:x4n] => Dict(:x7y => 0.5, :x7n => 0.5)
        )
        x7 = DiscreteChildNode(:x7, [x4], x7_states)
        x11_states = Dict(
            [:x8y] => Dict(:x11y => 0.5, :x11n => 0.5),
            [:x8n] => Dict(:x11y => 0.5, :x11n => 0.5)
        )
        x11 = DiscreteChildNode(:x11, [x8], x11_states)
        x6_states = Dict(
            [:x4y, :x3y] => Dict(:x6y => 0.5, :x6n => 0.5),
            [:x4y, :x3n] => Dict(:x6y => 0.5, :x6n => 0.5),
            [:x4n, :x3y] => Dict(:x6y => 0.5, :x6n => 0.5),
            [:x4n, :x3n] => Dict(:x6y => 0.5, :x6n => 0.5)
        )
        x6 = DiscreteChildNode(:x6, [x4, x3], x6_states)
        x6_states = Dict(
            [:x4y, :x3y] => Dict(:x6y => 0.5, :x6n => 0.5),
            [:x4y, :x3n] => Dict(:x6y => 0.5, :x6n => 0.5),
            [:x4n, :x3y] => Dict(:x6y => 0.5, :x6n => 0.5),
            [:x4n, :x3n] => Dict(:x6y => 0.5, :x6n => 0.5)
        )
        x6 = DiscreteChildNode(:x6, [x4, x3], x6_states)
        x9_states = Dict(
            [:x6y, :x5y] => Dict(:x9y => 0.5, :x9n => 0.5),
            [:x6y, :x5n] => Dict(:x9y => 0.5, :x9n => 0.5),
            [:x6n, :x5y] => Dict(:x9y => 0.5, :x9n => 0.5),
            [:x6n, :x5n] => Dict(:x9y => 0.5, :x9n => 0.5)
        )
        x9 = DiscreteChildNode(:x9, [x6, x5], x9_states)
        x10_states = Dict(
            [:x6y, :x8y] => Dict(:x10y => 0.5, :x10n => 0.5),
            [:x6y, :x8n] => Dict(:x10y => 0.5, :x10n => 0.5),
            [:x6n, :x8y] => Dict(:x10y => 0.5, :x10n => 0.5),
            [:x6n, :x8n] => Dict(:x10y => 0.5, :x10n => 0.5)
        )
        x10 = DiscreteChildNode(:x10, [x6, x8], x10_states)
        x12_states = Dict(
            [:x9y] => Dict(:x12y => 0.5, :x12n => 0.5),
            [:x9n] => Dict(:x12y => 0.5, :x12n => 0.5)
        )
        x12 = DiscreteChildNode(:x12, [x9], x12_states)
        x13_states = Dict(
            [:x10y] => Dict(:x13y => 0.5, :x13n => 0.5),
            [:x10n] => Dict(:x13y => 0.5, :x13n => 0.5)
        )
        x13 = DiscreteChildNode(:x13, [x10], x13_states)
        nodes = [x1, x2, x4, x8, x5, x7, x11, x3, x6, x9, x10, x12, x13]
        ebn = EnhancedBayesianNetwork(nodes)

        @test issetequal(markov_blanket(ebn, x6), [x3, x4, x5, x8, x9, x10])
    end
    @testset "Markov Blanket" begin
        Y1 = DiscreteRootNode(:y1, Dict(:yy1 => 0.5, :yn1 => 0.5), Dict(:yy1 => [Parameter(0.5, :y1)], :yn1 => [Parameter(0.8, :y1)]))
        X1 = ContinuousRootNode(:x1, Normal())
        X2 = ContinuousRootNode(:x2, Normal())
        X3 = ContinuousRootNode(:x3, Normal())

        model = Model(df -> df.y1 .+ df.x1, :y2)
        models = [model]
        simulation = MonteCarlo(200)
        performance = df -> df.y2
        Y2 = DiscreteFunctionalNode(:y2, [Y1, X1], models, performance, simulation)

        model = Model(df -> df.x1 .+ df.x2, :y3)
        models = [model]
        simulation = MonteCarlo(200)
        performance = df -> df.y3
        Y3 = DiscreteFunctionalNode(:y3, [X1, X2], models, performance, simulation)

        model = Model(df -> df.x3 .+ df.x2, :y4)
        models = [model]
        simulation = MonteCarlo(200)
        performance = df -> df.y4
        Y4 = DiscreteFunctionalNode(:y4, [X3, X2], models, performance, simulation)

        model = Model(df -> df.x3, :y5)
        models = [model]
        simulation = MonteCarlo(200)
        performance = df -> df.y5
        parameter = Dict(:fail_y5 => [Parameter(1, :y5)], :fail_y5 => [Parameter(0, :y5)])
        Y5 = DiscreteFunctionalNode(:y5, [X3], models, performance, simulation, parameter)

        model = Model(df -> df.y3, :x4)
        models = [model]
        simulation = MonteCarlo(200)
        X4 = ContinuousFunctionalNode(:x4, [Y5], models, simulation)

        model = Model(df -> df.x4, :y6)
        models = [model]
        simulation = MonteCarlo(200)
        performance = df -> df.y6
        Y6 = DiscreteFunctionalNode(:y6, [X4], models, performance, simulation)
        nodes = [X1, X2, X3, Y1, Y2, Y3, Y4, Y5, X4, Y6]
        ebn = EnhancedBayesianNetwork(nodes)

        @test issetequal(EnhancedBayesianNetworks._markov_envelope_continuous_nodes_group(ebn, Y5), [Y5, X4, X3])

        envelopes = markov_envelope(ebn)
        @test issetequal(envelopes[1], [X1, X2, X3, Y1, Y2, Y3, Y4, Y5])
        @test issetequal(envelopes[2], [Y6, Y5, X4])
    end

    @testset "Straub-Example" begin
        using .MathConstants: γ

        Uᵣ = ContinuousRootNode(:Uᵣ, Normal())
        μ_gamma = 60
        cov_gamma = 0.2
        α, θ = distribution_parameters(μ_gamma, μ_gamma * cov_gamma, Gamma)
        V = ContinuousRootNode(:V, Gamma(α, θ))

        μ_gumbel = 50
        cov_gumbel = 0.4
        μ_loc, β = distribution_parameters(μ_gumbel, cov_gumbel * μ_gumbel, Gumbel)
        H = ContinuousRootNode(:H, Gumbel(μ_loc, β))

        function plastic_moment_capacities(uᵣ)
            ρ = 0.5477
            μ = 150
            cov = 0.2

            λ, ζ = distribution_parameters(μ, μ * cov, LogNormal)

            normal_μ = λ + ρ * ζ * uᵣ
            normal_std = sqrt((1 - ρ^2) * ζ^2)
            exp(rand(Normal(normal_μ, normal_std)))
        end

        parents = [Uᵣ]
        model1 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r1)
        model2 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r2)
        model3 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r3)
        model4 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r4)
        model5 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r5)

        R1 = ContinuousFunctionalNode(:R1, parents, [model1], MonteCarlo(10^6))
        R2 = ContinuousFunctionalNode(:R2, parents, [model2], MonteCarlo(10^6))
        R3 = ContinuousFunctionalNode(:R3, parents, [model3], MonteCarlo(10^6))
        R4 = ContinuousFunctionalNode(:R4, parents, [model4], MonteCarlo(10^6))
        R5 = ContinuousFunctionalNode(:R5, parents, [model5], MonteCarlo(10^6))


        function frame_model(r1, r2, r3, r4, r5, v, h)
            g1 = r1 + r2 + r4 + r5 - 5 * h
            g2 = r2 + 2 * r3 + r4 - 5 * v
            g3 = r1 + 2 * r3 + 2 * r4 + r5 - 5 * h - 5 * v
            return minimum([g1, g2, g3])
        end

        model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.r4, df.r5, df.V, df.H), :G)
        performance = df -> df.G
        simulation = MonteCarlo(10^6)
        frame = DiscreteFunctionalNode(:E, [R1, R2, R3, R4, R5, V, H], [model], performance, simulation)


        nodes = [Uᵣ, V, H, R1, R2, R3, R4, R5, frame]
        ebn = EnhancedBayesianNetwork(nodes)

        eebn = EnhancedBayesianNetworks.evaluate(ebn)

        @test isapprox(eebn.nodes[end].states[:safe_E], 0.973871; atol=0.01)
        @test isapprox(eebn.nodes[end].states[:fail_E], 0.026129; atol=0.01)

        n = 1000
        discretization1 = ApproximatedDiscretization(collect(range(50, 250, 21)), 1)
        discretization2 = ApproximatedDiscretization(collect(range(50.01, 250.01, 21)), 1)
        R4 = ContinuousFunctionalNode(:R4, parents, [model4], MonteCarlo(n), discretization1)
        R5 = ContinuousFunctionalNode(:R5, parents, [model5], MonteCarlo(n), discretization2)

        model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.R4, df.R5, df.V, df.H), :G)
        performance = df -> df.G
        simulation = MonteCarlo(1000)
        frame = DiscreteFunctionalNode(:E, [R1, R2, R3, R4, R5, V, H], [model], performance, simulation)

        nodes = [Uᵣ, V, H, R1, R2, R3, R4, R5, frame]
        ebn = EnhancedBayesianNetwork(nodes)
        eebn = @suppress EnhancedBayesianNetworks.evaluate(ebn)

        evidence2 = Dict(
            :R4_d => Symbol([140.0, 150.0]),
            :R5_d => Symbol([90.01, 100.01])
        )
        ϕ2 = infer(eebn, :E, evidence2)

        @test all(isapprox.(ϕ2.potential, [0.965, 0.035], atol=0.05))
    end
end
