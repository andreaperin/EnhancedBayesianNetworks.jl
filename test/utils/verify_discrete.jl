@testset "verify single scenario" begin

    @testset "precise" begin

        sub_cpt1 = DataFrame(:x => [:yesx, :yesx], :y => [:yesy, :noy], :Π => [0.3, 0.7])
        sub_cpt2_1 = DataFrame(:x => [:nox, :nox], :y => [:yesy, :noy], :Π => [0.19999, 0.8])
        sub_cpt2_2 = DataFrame(:x => [:nox, :nox], :y => [:yesy, :noy], :Π => [0.19999, 0.8])
        sub_cpt3 = DataFrame(:x => [:nox, :nox], :y => [:yesy, :noy], :Π => [0.1, 0.8])

        @test [0.3, 0.7] == EnhancedBayesianNetworks._verify_single_scenario!(sub_cpt1)

        @test_logs (:warn, "total probability should be one, but the evaluated value is 0.99999, and will be normalized") EnhancedBayesianNetworks._verify_single_scenario!(sub_cpt2_1)

        @suppress isapprox.(
            EnhancedBayesianNetworks._verify_single_scenario!(sub_cpt2_2), [0.1999919999199992, 0.8000080000800008])

        @test_throws ErrorException("states are not exhaustive and mutually exclusives for the following cpt: $sub_cpt3") EnhancedBayesianNetworks._verify_single_scenario!(sub_cpt3)
    end

    @testset "imprecise" begin

        sub_cpt1 = DataFrame(:x => [:yesx, :yesx], :y => [:yesy, :noy], :Π => [(0.3, 0.4), (0.6, 0.7)])
        sub_cpt2 = DataFrame(:x => [:nox, :nox], :y => [:yesy, :noy], :Π => [(0.1, 0.2), (0.4, 0.5)])
        sub_cpt3 = DataFrame(:x => [:nox, :nox], :y => [:yesy, :noy], :Π => [(0.3, 0.4), (0.8, 0.9)])

        @test [(0.3, 0.4), (0.6, 0.7)] == EnhancedBayesianNetworks._verify_single_scenario!(sub_cpt1)

        @test_throws ErrorException("sum of intervals upper bounds is smaller than 1: $sub_cpt2") EnhancedBayesianNetworks._verify_single_scenario!(sub_cpt2)

        @test_throws ErrorException("sum of intervals lower bounds is bigger than 1: $sub_cpt3") EnhancedBayesianNetworks._verify_single_scenario!(sub_cpt3)
    end
end

@testset "verify probabilities" begin

    @testset "precise" begin
        cpt0 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:a])
        cpt0[:a=>:yes] = -0.5
        cpt0[:a=>:no] = 0.5
        @test_throws ErrorException("probabilities must be non-negative: -0.5") EnhancedBayesianNetworks._verify_probabilities!(cpt0, :a)

        cpt1 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:a])
        cpt1[:a=>:yes] = 1.5
        cpt1[:a=>:no] = 0.5
        @test_throws ErrorException("probabilities must be lower or equal than 1: 1.5") EnhancedBayesianNetworks._verify_probabilities!(cpt1, :a)

        cpt2 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:x, :y])
        cpt2[:x=>:yesx, :y=>:yesy] = 0.3
        cpt2[:x=>:yesx, :y=>:noy] = 0.7
        cpt2[:x=>:nox, :y=>:yesy] = 0.4
        cpt2[:x=>:nox, :y=>:noy] = 0.6
        @test [0.3, 0.7, 0.4, 0.6] == EnhancedBayesianNetworks._verify_probabilities!(cpt2, :y)

        cpt3 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:x, :y])
        cpt3[:x=>:yesx, :y=>:yesy] = 0.2999
        cpt3[:x=>:yesx, :y=>:noy] = 0.7
        cpt3[:x=>:nox, :y=>:yesy] = 0.4
        cpt3[:x=>:nox, :y=>:noy] = 0.6

        @test_logs (:warn, "total probability should be one, but the evaluated value is 0.9999, and will be normalized") EnhancedBayesianNetworks._verify_probabilities!(cpt3, :y)

        cpt4 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:x, :y])
        cpt4[:x=>:yesx, :y=>:yesy] = 0.2999
        cpt4[:x=>:yesx, :y=>:noy] = 0.7
        cpt4[:x=>:nox, :y=>:yesy] = 0.4
        cpt4[:x=>:nox, :y=>:noy] = 0.6

        @test @suppress all(isapprox.(
            EnhancedBayesianNetworks._verify_probabilities!(cpt4, :y), [0.2999299929992999, 0.7000700070007, 0.4, 0.6]))
    end

    @testset "imprecise" begin

        cpt0 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}([:a])
        cpt0[:a=>:yes] = (0.5, 0.4)
        cpt0[:a=>:no] = (0.6, 0.8)

        @test_throws ErrorException("interval probabilities must have a lower bound smaller than upper bound. $(cpt0.data)") EnhancedBayesianNetworks._verify_probabilities!(cpt0, :a)

        cpt1 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}([:a])
        cpt1[:a=>:yes] = (0.2, 0.4)
        cpt1[:a=>:no] = (0.6, 0.8)

        @test [(0.2, 0.4), (0.6, 0.8)] == EnhancedBayesianNetworks._verify_probabilities!(cpt1, :a)

        cpt2 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}([:a])
        cpt2[:a=>:yes] = (0.2, 0.4)
        cpt2[:a=>:no] = (0.6, 1.1)
        @test_throws ErrorException("probabilities must be lower or equal than 1: 1.1") EnhancedBayesianNetworks._verify_probabilities!(cpt2, :a)

        cpt3 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}([:a])
        cpt3[:a=>:yes] = (-0.2, 0.4)
        cpt3[:a=>:no] = (0.6, 0.8)
        @test_throws ErrorException("probabilities must be non-negative: -0.2") EnhancedBayesianNetworks._verify_probabilities!(cpt3, :a)
    end
end

@testset "verify parameters" begin
    cpt1 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:x, :y])
    cpt1[:x=>:yesx, :y=>:yesy] = 0.3
    cpt1[:x=>:yesx, :y=>:noy] = 0.7
    cpt1[:x=>:nox, :y=>:yesy] = 0.4
    cpt1[:x=>:nox, :y=>:noy] = 0.6

    parameters = Dict(:yesy => [Parameter(1, :y)], :noy => [Parameter(0, :y)])

    @test isnothing(EnhancedBayesianNetworks._verify_parameters(cpt1, parameters, :y))

    @test_throws ErrorException("parameters keys [:yesy, :noy] must be coherent with states [:yesx, :nox]") EnhancedBayesianNetworks._verify_parameters(cpt1, parameters, :x)
end