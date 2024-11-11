@testset "Probabilities Verification" begin
    states = Dict(:a => 0.2, :b => 0.8)
    @test isnothing(EnhancedBayesianNetworks._verify_probabilities(states))

    states = Dict(:a => 0.2, :b => 0.1)
    @test_throws ErrorException("states [:a, :b] are not exhaustives and mutually exclusive. Their probabilities [0.2, 0.1] does not sum up to 1") EnhancedBayesianNetworks._verify_probabilities(states)

    states = Dict(:a => 0.2, :b => 0.7999999999)
    @test_logs (:warn, "total probaility should be one, but the evaluated value is 0.9999999999 , and will be normalized") EnhancedBayesianNetworks._verify_probabilities(states)

    states = Dict{Symbol,AbstractVector{Real}}(:a => [0.1, 0.2], :b => [0.2, 0.3], :c => [0.5, 0.7])
    @test isnothing(EnhancedBayesianNetworks._verify_probabilities(states))

    states = Dict{Symbol,AbstractVector{Real}}(:a => [-0.1, 0.2], :b => [0.2, 0.3], :c => [0.5, 0.7])
    @test_throws ErrorException("probabilities must be non-negative: Real[-0.1, 0.2, 0.2, 0.3, 0.5, 0.7]") EnhancedBayesianNetworks._verify_probabilities(states)

    states = Dict{Symbol,AbstractVector{Real}}(:a => [0.1, 1.2], :b => [0.2, 0.3], :c => [0.5, 0.7])
    @test_throws ErrorException("probabilities must be lower or equal than 1: Real[0.1, 1.2, 0.2, 0.3, 0.5, 0.7]") EnhancedBayesianNetworks._verify_probabilities(states)

    states = Dict{Symbol,AbstractVector{Real}}(:a => [0.1, 0.2], :b => [0.9, 0.99], :c => [0.5, 0.7])
    @test_throws ErrorException("sum of intervals lower bounds is bigger than 1: Real[0.1, 0.2, 0.9, 0.99, 0.5, 0.7]") EnhancedBayesianNetworks._verify_probabilities(states)

    states = Dict{Symbol,AbstractVector{Real}}(:a => [0.1, 0.2], :b => [0.1, 0.2], :c => [0.1, 0.2])
    @test_throws ErrorException("sum of intervals upper bounds is smaller than 1: Real[0.1, 0.2, 0.1, 0.2, 0.1, 0.2]") EnhancedBayesianNetworks._verify_probabilities(states)

    states = Dict(:a => 0.2, :b => 0.1)
    parameters = Dict(:a => [Parameter(1, :A)], :b => [Parameter(2, :A)])
    @test isnothing(EnhancedBayesianNetworks._verify_parameters(states, parameters))
    parameters = Dict(:c => [Parameter(1, :A)], :b => [Parameter(2, :A)])
    @test_throws ErrorException("parameters keys [:b, :c] must be coherent with states [:a, :b]") EnhancedBayesianNetworks._verify_parameters(states, parameters)
end