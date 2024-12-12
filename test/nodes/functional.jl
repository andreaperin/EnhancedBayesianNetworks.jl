@testset "Functional Nodes" begin
    @testset "ContinuousFunctionalNode" begin

        name = :functional
        model = Model(df -> sqrt.(df.z .^ 2 + df.z .^ 2), :value1)
        models = [model]
        simulation = MonteCarlo(200)
        discretization = ApproximatedDiscretization()

        node = ContinuousFunctionalNode(name, models, simulation)
        @test ContinuousFunctionalNode(name, models, simulation, discretization) == node
        @test EnhancedBayesianNetworks._is_root(node) == false
    end

    @testset "DiscreteFunctionalNode" begin

        name = :functional
        model = Model(df -> sqrt.(df.z .^ 2 + df.z .^ 2), :value1)
        models = [model]
        simulation = MonteCarlo(200)
        performances = df -> 1 .- 2 .* df.value1
        parameters = Dict{Symbol,Vector{Parameter}}()

        node = DiscreteFunctionalNode(name, models, performances, simulation, parameters)
        @test DiscreteFunctionalNode(name, models, performances, simulation) == node
        @test EnhancedBayesianNetworks._is_root(node) == false
    end

    @testset "Wrap Model" begin
        name = :functional
        model = Model(df -> sqrt.(df.z .^ 2 + df.z .^ 2), :value1)
        simulation = MonteCarlo(200)
        performances = df -> 1 .- 2 .* df.value1
        parameters = Dict{Symbol,Vector{Parameter}}()
        node = DiscreteFunctionalNode(name, [model], performances, simulation, parameters)

        @test node == DiscreteFunctionalNode(name, model, performances, simulation, parameters)
        @test node == DiscreteFunctionalNode(name, model, performances, simulation)

        name = :functional
        model = Model(df -> sqrt.(df.z .^ 2 + df.z .^ 2), :value1)
        simulation = MonteCarlo(200)
        discretization = ApproximatedDiscretization()
        node = ContinuousFunctionalNode(name, [model], simulation, discretization)

        @test node == ContinuousFunctionalNode(name, model, simulation, discretization)
        @test node == ContinuousFunctionalNode(name, model, simulation)
    end
end
