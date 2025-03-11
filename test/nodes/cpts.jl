@testset "CPTs" begin
    @testset "Discrete CPT" begin
        cpt1 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:x])
        @test names(cpt1.data) == ["x", "Π"]
        @test typeof(cpt1).parameters[1] == PreciseDiscreteProbability
        cpt2 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:x)
        @test cpt1 == cpt2
        cpt3 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}([:x])
        @test names(cpt3.data) == ["x", "Π"]
        @test typeof(cpt3).parameters[1] == ImpreciseDiscreteProbability
        cpt4 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}(:x)
        @test cpt3 == cpt4

        cpt5 = DataFrame(:x => [:yesx, :yesx, :nox, :nox], :y => [:yesy, :noy, :yesy, :noy], :Π => [0.1, 0.9, 0.2, 0.8])
        cpt = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(cpt5)
        @test cpt.data == cpt5

        @test isroot(cpt1)
        @test !isroot(cpt)
    end
    @testset "Continuous CPT" begin
        cpt1 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}([:x])
        @test names(cpt1.data) == ["x", "Π"]
        @test typeof(cpt1).parameters[1] == PreciseContinuousInput
        cpt2 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}(:x)
        @test cpt1 == cpt2
        cpt3 = ContinuousConditionalProbabilityTable{ImpreciseContinuousInput}([:x])
        @test names(cpt3.data) == ["x", "Π"]
        @test typeof(cpt3).parameters[1] == ImpreciseContinuousInput
        cpt4 = ContinuousConditionalProbabilityTable{ImpreciseContinuousInput}(:x)
        @test cpt3 == cpt4
        cpt_root = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
        @test names(cpt_root.data) == ["Π"]

        cpt5 = DataFrame(:x => [:yesx, :nox], :Π => [Normal(), Normal(2, 1)])
        cpt = ContinuousConditionalProbabilityTable{PreciseContinuousInput}(cpt5)
        @test cpt.data == cpt5

        cpt1 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
        cpt1[] = Normal()
        @test isroot(cpt1)
        @test !isroot(cpt)
    end

    @testset "CPT functions" begin
        @testset "setindex" begin
            cpt = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:x, :y])
            cpt[:x=>:yesx, :y=>:yesy] = 0.3
            @test cpt.data == DataFrame(:x => :yesx, :y => :yesy, :Π => 0.3)
            ## testing overwriting
            cpt[:x=>:yesx, :y=>:yesy] = 0.7
            @test cpt.data == DataFrame(:x => :yesx, :y => :yesy, :Π => 0.7)
            @test_throws ErrorException("Cannot set index with [:x] into a CPT initialized with [:x, :y]") cpt[:x=>:yesx] = 0.2
            push!(cpt.data, Dict(:x => :yesx, :y => :yesy, :Π => 0.2))
            ## testing assertion error
            @test_throws AssertionError("size(cp, 1) == 1") cpt[:x=>:yesx, :y=>:yesy] = 0.9
        end
        @testset "getindex" begin
            cpt = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:x, :y])
            cpt[:x=>:yesx, :y=>:yesy] = 0.3
            cpt[:x=>:yesx, :y=>:noy] = 0.5
            @test cpt[:x=>:yesx, :y=>:yesy] == 0.3
            @test_throws AssertionError("size(cp, 1) == 1") cpt[:x=>:yesx]
        end
        @testset "states & scenarios" begin
            cpt_child = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:x, :y])
            cpt_child[:x=>:yesx, :y=>:yesy] = 0.3
            cpt_child[:x=>:yesx, :y=>:noy] = 0.5
            cpt_child[:x=>:nox, :y=>:yesy] = 0.3
            cpt_child[:x=>:nox, :y=>:noy] = 0.5
            @test issetequal(states(cpt_child, :y), [:yesy, :noy])
            @test issetequal(scenarios(cpt_child, :y), [Dict(:x => :yesx), Dict(:x => :nox)])
            cpt_root1 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
            cpt_root1[] = Normal()
            @test EnhancedBayesianNetworks._scenarios_cpt(cpt_root1, :y) == [DataFrame(:Π => Normal())]

            cpt_root2 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:x)
            cpt_root2[:x=>:yesx] = 0.5
            cpt_root2[:x=>:nox] = 0.5
            @test EnhancedBayesianNetworks._scenarios_cpt(cpt_root2, :y) == [DataFrame(:x => [:yesx, :nox], :Π => [0.5, 0.5])]
        end
        @testset "isprecise" begin
            cpt1 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
            cpt2 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:x)
            @test isprecise(cpt1)
            @test isprecise(cpt2)
            cpt1 = ContinuousConditionalProbabilityTable{ImpreciseContinuousInput}()
            cpt2 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}(:x)
            @test !isprecise(cpt1)
            @test !isprecise(cpt2)
        end
    end
end