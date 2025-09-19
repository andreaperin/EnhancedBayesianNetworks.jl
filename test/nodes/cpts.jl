@testset "CPTs" begin
    @testset "Discrete CPT" begin
        cpt = ConditionalProbabilityTable{DiscreteProbability}([:x])
        @test names(cpt.data) == ["x", "Π"]
        @test typeof(cpt).parameters[1] == DiscreteProbability
        @test eltype(cpt.data.Π) == DiscreteProbability
        cpt = ConditionalProbabilityTable{DiscreteProbability}(:x)
        @test typeof(cpt).parameters[1] == DiscreteProbability
        @test eltype(cpt.data.Π) == DiscreteProbability
        cpt = ConditionalProbabilityTable{DiscreteProbability}([:x, :y])
        cpt[:x=>:yesx, :y=>:yesy] = 0.1
        cpt[:x=>:yesx, :y=>:noy] = 0.9
        cpt[:x=>:nox, :y=>:yesy] = 0.2
        cpt[:x=>:nox, :y=>:noy] = 0.8

        @test cpt.data.Π == [0.1, 0.9, 0.2, 0.8]

        @test cpt[:x=>:yesx, :y=>:yesy] == 0.1
        @test cpt[:x=>:yesx, :y=>:noy] == 0.9
        @test cpt[:x=>:nox, :y=>:yesy] == 0.2
        @test cpt[:x=>:nox, :y=>:noy] == 0.8
    end
    @testset "Continuous CPT" begin
        cpt = ConditionalProbabilityTable{ContinuousProbability}(:x)
        @test names(cpt.data) == ["x", "Π"]
        @test typeof(cpt).parameters[1] == ContinuousProbability
        @test eltype(cpt.data.Π) == ContinuousProbability
        cpt = ConditionalProbabilityTable{ContinuousProbability}([:x])
        @test names(cpt.data) == ["x", "Π"]
        @test typeof(cpt).parameters[1] == ContinuousProbability
        @test eltype(cpt.data.Π) == ContinuousProbability
        cpt = ConditionalProbabilityTable{ContinuousProbability}([:x])
        @test eltype(cpt.data.Π) == ContinuousProbability
        cpt[:x=>:yesx] = Normal()
        cpt[:x=>:nox] = Normal(2, 1)

        @test cpt.data.Π == [Normal(), Normal(2, 1)]

        @test cpt[:x=>:yesx] == Normal()
        @test cpt[:x=>:nox] == Normal(2, 1)

        cpt = ConditionalProbabilityTable{ContinuousProbability}(Symbol[])
        @test names(cpt.data) == ["Π"]
        @test typeof(cpt).parameters[1] == ContinuousProbability
        @test eltype(cpt.data.Π) == ContinuousProbability

        cpt[] = Normal()
        @test cpt.data.Π == [Normal()]
        @test cpt[] == Normal()
    end
end