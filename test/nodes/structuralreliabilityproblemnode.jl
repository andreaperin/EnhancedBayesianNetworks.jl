@testset "StructuralReliabilityProblem Nodes" begin
    @testset "ContinuousStructuralReliabilitylProblem Node" begin

        root1 = DiscreteRootNode(:x, Dict(:yes_1 => 0.2, :no_1 => 0.8), Dict(:yes_1 => [Parameter(2.2, :x)], :no_1 => [Parameter(5.5, :x)]))
        root2 = DiscreteRootNode(:y, Dict(:yes_2 => 0.4, :no_2 => 0.6), Dict(:yes_2 => [Parameter(2.2, :y)], :no_2 => [Parameter(5.5, :y)]))
        root3 = ContinuousRootNode(:z, Normal())

        name = :srp
        parents = [root1, root2, root3]
        model = Model(df -> sqrt.(df.z .^ 2 + df.z .^ 2), :value1)
        models = [model]
        simulation = MonteCarlo(200)

        pdf = EnhancedBayesianNetworks.StructuralReliabilityProblemPDF(models, [get_randomvariable(root3)], simulation)

        @test pdf.models == models
        @test pdf.inputs == [get_randomvariable(root3)]
        @test pdf.simulation == simulation

        srps = Dict(
            [:yes_1, :yes_2] => pdf,
            [:yes_1, :no_2] => pdf,
            [:no_1, :yes_2] => pdf,
            [:no_1, :yes_2] => pdf
        )

        node = EnhancedBayesianNetworks.ContinuousStructuralReliabilityProblemNode(name, [root1, root2, root3], srps, ApproximatedDiscretization())

        @test node.name == name
        @test issetequal(node.parents, [root1, root2, root3])
        @test node.srps == srps
        @test isequal(node.discretization, ApproximatedDiscretization())
    end

    @testset "DiscreteStructuralReliabilityProblem Node" begin
        root1 = DiscreteRootNode(:x, Dict(:yes_1 => 0.2, :no_1 => 0.8), Dict(:yes_1 => [Parameter(2.2, :x)], :no_1 => [Parameter(5.5, :x)]))
        root2 = DiscreteRootNode(:y, Dict(:yes_2 => 0.4, :no_2 => 0.6), Dict(:yes_2 => [Parameter(2.2, :y)], :no_2 => [Parameter(5.5, :y)]))
        root3 = ContinuousRootNode(:z, Normal())

        name = :srp
        parents = [root1, root2, root3]
        model = Model(df -> sqrt.(df.z .^ 2 + df.z .^ 2), :value1)
        models = [model]
        performance = df -> 1 .- 2 .* df.value1
        simulation = MonteCarlo(200)

        pmf = EnhancedBayesianNetworks.StructuralReliabilityProblemPMF(models, [get_randomvariable(root3)], performance, simulation)

        @test pmf.models == models
        @test pmf.inputs == [get_randomvariable(root3)]
        @test pmf.performance == performance
        @test pmf.simulation == simulation

        srps = Dict(
            [:yes_1, :yes_2] => pmf,
            [:yes_1, :no_2] => pmf,
            [:no_1, :yes_2] => pmf,
            [:no_1, :yes_2] => pmf
        )

        node = EnhancedBayesianNetworks.DiscreteStructuralReliabilityProblemNode(name, [root1, root2, root3], srps, Dict{Symbol,Vector{Parameter}}())

        @test node.name == name
        @test issetequal(node.parents, [root1, root2, root3])
        @test node.srps == srps
        @test node.parameters == Dict{Symbol,Vector{Parameter}}()
    end
end
