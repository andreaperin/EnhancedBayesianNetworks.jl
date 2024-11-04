@testset "Functional Nodes" begin
    @testset "ContinuousFunctionalNode" begin

        name = :functional
        model = Model(df -> sqrt.(df.z .^ 2 + df.z .^ 2), :value1)
        models = [model]
        simulation = MonteCarlo(200)
        discretization = ApproximatedDiscretization()

        @test ContinuousFunctionalNode(name, models, simulation, discretization) == ContinuousFunctionalNode(name, models, simulation)
    end

    @testset "DiscreteFunctionalNode" begin

        name = :functional
        model = Model(df -> sqrt.(df.z .^ 2 + df.z .^ 2), :value1)
        models = [model]
        simulation = MonteCarlo(200)
        performances = df -> 1 .- 2 .* df.value1
        parameters = Dict{Symbol,Vector{Parameter}}()

        @test DiscreteFunctionalNode(name, models, performances, simulation) == DiscreteFunctionalNode(name, models, performances, simulation, parameters)
    end
end
