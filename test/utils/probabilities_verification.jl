@testset "Verify Probabilities" begin
    @testset "Precise Nodes" begin
        states = Dict(:yes => -0.5, :no => 0.5)
        @test_throws ErrorException("probabilities must be non-negative") EnhancedBayesianNetworks._verify_probabilities(states)

        states = Dict(:yes => 0.5, :no => 1.5)
        @test_throws ErrorException("probabilities must be lower or equal than 1") EnhancedBayesianNetworks._verify_probabilities(states)

        states = Dict(:yes => 0.8, :no => 0.8)
        @test_throws ErrorException("states [:yes, :no] are exhaustives and mutually exclusive. Their probabilities [0.8, 0.8] does not sum up to 1") EnhancedBayesianNetworks._verify_probabilities(states)

        states = Dict(:yes => 0.4999, :no => 0.4999)
        @test_logs (:warn, "total probaility should be one, but the evaluated value is 0.9998 , and will be normalized") EnhancedBayesianNetworks._verify_probabilities(states)

        @test EnhancedBayesianNetworks._normalize_states!(states) == Dict(:yes => 0.5, :no => 0.5)

        parameters = Dict(:yes => [Parameter(1, :A)], :n => [Parameter(2, :A)])
        @test_throws ErrorException("parameters keys [:n, :yes] must be coherent with states [:yes, :no]") EnhancedBayesianNetworks._verify_parameters(states, parameters)

        states = Dict(
            [:yes] => Dict(:ye => 0.1, :no => 0.9),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test_throws ErrorException("non coherent definition of states over scenarios: [([:no, :ye], [0.9, 0.1]), ([:yes, :no], [0.2, 0.8])]") EnhancedBayesianNetworks._check_child_states!(states)

        states = Dict(
            [:yes] => Dict(:yes => 0.1, :no => 0.9),
            [:no] => Dict(:yes => [0.2, 0.3], :no => [0.7, 0.8])
        )
        @test_throws ErrorException("mixed interval and single value states probabilities are not allowed") EnhancedBayesianNetworks._check_child_states!(states)

        states = Dict(:yes => 0.2, :no => 0.8)
        @test EnhancedBayesianNetworks._check_root_states!(states) == Dict{Symbol,Real}(:yes => 0.2, :no => 0.8)

        states = Dict(
            [:yes] => Dict(:yes => 0.1, :no => 0.9),
            [:no] => Dict(:yes => 0.2, :no => 0.8)
        )
        @test EnhancedBayesianNetworks._check_child_states!(states) == states
    end

    @testset "Imprecise Nodes" begin
        states = Dict(:yes => [0.1, 0.15], :no => [0.8, 0.1, 0.1])
        @test_throws ErrorException("interval probabilities must be defined with a 2-values vector") EnhancedBayesianNetworks._verify_probabilities(states)

        states = Dict(:yes => [-0.5, 0.6], :no => [0.7, 0.9])
        @test_throws ErrorException("probabilities must be non-negative: [-0.5, 0.6, 0.7, 0.9]") EnhancedBayesianNetworks._verify_probabilities(states)

        states = Dict(:yes => [1.5, 0.6], :no => [0.7, 0.9])
        @test_throws ErrorException("probabilities must be lower or equal than 1: [1.5, 0.6, 0.7, 0.9]") EnhancedBayesianNetworks._verify_probabilities(states)

        states = Dict(:yes => [0.5, 0.6], :no => [0.7, 0.9])
        @test_throws ErrorException("sum of intervals lower bounds is bigger than 1: [0.5, 0.6, 0.7, 0.9]") EnhancedBayesianNetworks._verify_probabilities(states)

        states = Dict(:yes => [0.1, 0.15], :no => [0.7, 0.8])
        @test_throws ErrorException("sum of intervals upper bounds is smaller than 1: [0.1, 0.15, 0.7, 0.8]") EnhancedBayesianNetworks._verify_probabilities(states)

        states = Dict(:yes => [0.3, 0.4], :no => [0.6, 0.7])
        @test EnhancedBayesianNetworks._normalize_states!(states) == states
    end
end