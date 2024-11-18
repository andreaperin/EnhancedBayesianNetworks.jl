@testset "Discrete Nodes" begin

    @testset "Root Nodes" begin

        @testset "cpt verification" begin
            name = :a
            cpt1 = DataFrame(:x => [:yes, :no], :Prob => [0.5, 0.5])
            cpt2 = DataFrame(:x => [:yes, :no], :Prob => [0.5, [0.4, 0.3]])
            cpt3 = DataFrame(:x => [:yes, :no], :Prob => [[0.3, 0.4], [0.6, 0.7]])
            @test isnothing(EnhancedBayesianNetworks._verify_cpt_coherence(cpt1))
            @test isnothing(EnhancedBayesianNetworks._verify_cpt_coherence(cpt3))
            @test_throws ErrorException("Mixed precise and imprecise probabilities values $cpt2") isnothing(EnhancedBayesianNetworks._verify_cpt_coherence(cpt2))

            cpt4 = DataFrame(:x => [:yes, :no], :Prob => [-0.5, 0.5])
            cpt5 = DataFrame(:x => [:yes, :no], :Prob => [1.5, 0.5])
            @test isnothing(EnhancedBayesianNetworks._verify_precise_probabilities_values(cpt1))
            @test_throws ErrorException("probabilities must be non-negative: $cpt4") EnhancedBayesianNetworks._verify_precise_probabilities_values(cpt4)
            @test_throws ErrorException("probabilities must be lower or equal than 1: $cpt5") EnhancedBayesianNetworks._verify_precise_probabilities_values(cpt5)

            cpt6 = DataFrame(:x => [:yes, :no], :Prob => [[0.6, 0.4], [0.6, 0.7]])
            cpt7 = DataFrame(:x => [:yes, :no], :Prob => [[0.1, 0.4, 0.1], [0.6, 0.7]])
            @test isnothing(EnhancedBayesianNetworks._verify_imprecise_probabilities_values(cpt3))
            @test_throws ErrorException("interval probabilities must lower bound smaller than upper bound. $cpt6") EnhancedBayesianNetworks._verify_imprecise_probabilities_values(cpt6)
            @test_throws ErrorException("interval probabilities must be defined with a 2-values vector. $cpt7") EnhancedBayesianNetworks._verify_imprecise_probabilities_values(cpt7)

            cpt8 = DataFrame(:x => [:yes, :no], :Prob => [[0.2, 0.4], [0.2, 0.3]])
            cpt9 = DataFrame(:x => [:yes, :no], :Prob => [[0.5, 0.6], [0.6, 0.7]])
            @test_throws ErrorException("sum of intervals upper bounds is smaller than 1: $cpt8") EnhancedBayesianNetworks._verify_imprecise_exhaustiveness(cpt8)
            @test_throws ErrorException("sum of intervals lower bounds is bigger than 1: $cpt9") EnhancedBayesianNetworks._verify_imprecise_exhaustiveness(cpt9)

            cpt10 = DataFrame(:x => [:yes, :no], :Prob => [0.9, 0.5])
            cpt11 = DataFrame(:x => [:yes, :no], :Prob => [0.1, 0.5])
            cpt12 = DataFrame(:x => [:yes, :no], :Prob => [0.49, 0.49])
            @test_throws ErrorException("states are not exhaustive and mutually exclusives for the following cpt: $cpt10") EnhancedBayesianNetworks._verify_precise_exhaustiveness_and_normalize!(cpt10)
            @test_throws ErrorException("states are not exhaustive and mutually exclusives for the following cpt: $cpt11") EnhancedBayesianNetworks._verify_precise_exhaustiveness_and_normalize!(cpt11)

            @test_logs (:warn, "total probability should be one, but the evaluated value is 0.98, and will be normalized") EnhancedBayesianNetworks._verify_precise_exhaustiveness_and_normalize!(cpt12)
            new_cpt = @suppress EnhancedBayesianNetworks._verify_precise_exhaustiveness_and_normalize!(cpt12)
            @test new_cpt == DataFrame(:x => [:yes, :no], :Prob => [0.5, 0.5])

            cpt12 = DataFrame(:x => [:yes, :no], :Prob => [0.49, 0.49])
            @test EnhancedBayesianNetworks._normalize!(cpt12) == [0.5, 0.5]
        end

        @testset "node verification" begin
            name = :x
            parameters = Dict(:yes => [Parameter(2, :d)], :no => [Parameter(0, :d)])
            cpt = DataFrame(:x => [:yes, :no], :Prob => [-0.5, 0.5])
            @test_throws ErrorException("probabilities must be non-negative: $cpt") EnhancedBayesianNetworks._verify_cpt_and_normalize!(cpt, name)

            cpt = DataFrame(:x => [:yes, :no], :Prob => [0.5, 1.5])
            @test_throws ErrorException("probabilities must be lower or equal than 1: $cpt") EnhancedBayesianNetworks._verify_cpt_and_normalize!(cpt, name)

            cpt = DataFrame(:x => [:yes, :no], :Prob => [0.8, 0.8])
            @test_throws ErrorException("states are not exhaustive and mutually exclusives for the following cpt: $cpt") EnhancedBayesianNetworks._verify_cpt_and_normalize!(cpt, name)

            cpt = DataFrame(:x => [:yes, :no], :Prob => [0.49, 0.49])
            @test_logs (:warn, "total probability should be one, but the evaluated value is 0.98, and will be normalized") EnhancedBayesianNetworks._verify_cpt_and_normalize!(cpt, name)

            cpt6 = DataFrame(:x => [:yes, :no], :Prob => [[0.6, 0.4], [0.6, 0.7]])
            cpt7 = DataFrame(:x => [:yes, :no], :Prob => [[0.1, 0.4, 0.1], [0.6, 0.7]])
            cpt8 = DataFrame(:x => [:yes, :no], :Prob => [[0.2, 0.4], [0.2, 0.3]])
            cpt9 = DataFrame(:x => [:yes, :no], :Prob => [[0.5, 0.6], [0.6, 0.7]])

            @test_throws ErrorException("interval probabilities must lower bound smaller than upper bound. $cpt6") EnhancedBayesianNetworks._verify_cpt_and_normalize!(cpt6, name)
            @test_throws ErrorException("interval probabilities must be defined with a 2-values vector. $cpt7") EnhancedBayesianNetworks._verify_cpt_and_normalize!(cpt7, name)
            @test_throws ErrorException("sum of intervals upper bounds is smaller than 1: $cpt8") EnhancedBayesianNetworks._verify_cpt_and_normalize!(cpt8, name)
            @test_throws ErrorException("sum of intervals lower bounds is bigger than 1: $cpt9") EnhancedBayesianNetworks._verify_cpt_and_normalize!(cpt9, name)

            cpt4 = DataFrame(:x => [:yes, :noo], :Prob => [0.2, 0.8])
            @test isnothing(EnhancedBayesianNetworks._verify_parameters(cpt, parameters, name))
            @test_throws ErrorException("parameters keys [:yes, :no] must be coherent with states [:yes, :noo]") EnhancedBayesianNetworks._verify_parameters(cpt4, parameters, name)

            name1 = :a
            parameters = Dict(:yes => [Parameter(2, :d)], :no => [Parameter(0, :d)])
            cpt = DataFrame(:x => [:yes, :no], :Prob => [-0.5, 0.5])
            @test_throws ErrorException("defined cpt does not contain a column refered to node name a: $cpt") DiscreteNode(name1, cpt, parameters)

            cpt1 = DataFrame(:x => [:yes, :no], :Prob => [0.2, 0.8])
            cpt2 = DataFrame(:x => [:yes, :no], :Prob => [[0.3, 0.4], [0.6, 0.7]])

            node1 = DiscreteNode(name, cpt1, parameters)
            node2 = DiscreteNode(name, cpt1)
            node3 = DiscreteNode(name, cpt2, parameters)
            node4 = DiscreteNode(name, cpt2)

            @test node1.name == name
            @test node1.parameters == parameters
            @test node1.cpt == cpt1
            @test node2.name == name
            @test isempty(node2.parameters)
            @test node2.cpt == cpt1
            @test node3.name == name
            @test node3.parameters == parameters
            @test node3.cpt == cpt2
            @test node4.name == name
            @test isempty(node4.parameters)
            @test node4.cpt == cpt2

            name = :x
            parameters = Dict(:yes => [Parameter(2, :d)], :no => [Parameter(0, :d)])
            cpt = DataFrame(:x => [:yes, :no], :Prob => [-0.5, 0.5])
            @test_throws ErrorException("probabilities must be non-negative: $cpt") DiscreteNode(name, cpt, parameters)

            cpt = DataFrame(:x => [:yes, :no], :Prob => [0.5, 1.5])
            @test_throws ErrorException("probabilities must be lower or equal than 1: $cpt") DiscreteNode(name, cpt, parameters)

            cpt = DataFrame(:x => [:yes, :no], :Prob => [0.8, 0.8])
            @test_throws ErrorException("states are not exhaustive and mutually exclusives for the following cpt: $cpt") DiscreteNode(name, cpt, parameters)

            cpt = DataFrame(:x => [:yes, :no], :Prob => [0.49, 0.49])
            @test_logs (:warn, "total probability should be one, but the evaluated value is 0.98, and will be normalized") DiscreteNode(name, cpt, parameters)

            cpt6 = DataFrame(:x => [:yes, :no], :Prob => [[0.6, 0.4], [0.6, 0.7]])
            cpt7 = DataFrame(:x => [:yes, :no], :Prob => [[0.1, 0.4, 0.1], [0.6, 0.7]])
            cpt8 = DataFrame(:x => [:yes, :no], :Prob => [[0.2, 0.4], [0.2, 0.3]])
            cpt9 = DataFrame(:x => [:yes, :no], :Prob => [[0.5, 0.6], [0.6, 0.7]])

            @test_throws ErrorException("interval probabilities must lower bound smaller than upper bound. $cpt6") DiscreteNode(name, cpt6)
            @test_throws ErrorException("interval probabilities must be defined with a 2-values vector. $cpt7") DiscreteNode(name, cpt7)
            @test_throws ErrorException("sum of intervals upper bounds is smaller than 1: $cpt8") DiscreteNode(name, cpt8)
            @test_throws ErrorException("sum of intervals lower bounds is bigger than 1: $cpt9") DiscreteNode(name, cpt9)
        end

        @testset "node functions" begin
            name = :x
            cpt = DataFrame(:x => [:yes, :no], :Prob => [0.2, 0.8])
            node = DiscreteNode(name, cpt)
            @test issetequal(EnhancedBayesianNetworks._states(cpt, name), [:yes, :no])
            @test issetequal(EnhancedBayesianNetworks._states(node), [:yes, :no])

            @test isempty(EnhancedBayesianNetworks._scenarios(cpt, name))
            @test isempty(EnhancedBayesianNetworks._scenarios(node))

        end
    end
    @testset "Child Nodes" begin

        @testset "node verification" begin
            name = :x
            parameters = Dict(:yes => [Parameter(2, :d)], :no => [Parameter(0, :d)])
            cpt = DataFrame(:a => [:a1, :a1, :a2, :a2], :x => [:yes, :no, :yes, :no], :Prob => [-0.5, 0.5, 0.6, 0.4])
            @test_throws ErrorException("probabilities must be non-negative: $cpt") DiscreteNode(name, cpt)

            cpt = DataFrame(:a => [:a1, :a1, :a2, :a2], :x => [:yes, :no, :yes, :no], :Prob => [0.5, 1.5, 0.6, 0.4])
            @test_throws ErrorException("probabilities must be lower or equal than 1: $cpt") DiscreteNode(name, cpt)

            cpt = DataFrame(:a => [:a1, :a1, :a2, :a2], :x => [:yes, :no, :yes, :no], :Prob => [0.2, 0.5, 0.6, 0.4])
            error_cpt = DataFrame(:a => [:a1, :a1], :x => [:yes, :no], :Prob => [0.2, 0.5])
            @test_throws ErrorException("states are not exhaustive and mutually exclusives for the following cpt: $error_cpt") DiscreteNode(name, cpt)

            cpt = DataFrame(:a => [:a1, :a1, :a2, :a2], :x => [:yes, :no, :yes, :no], :Prob => [0.49, 0.5, 0.6, 0.4])
            @test_logs (:warn, "total probability should be one, but the evaluated value is 0.99, and will be normalized") DiscreteNode(name, cpt)

            cpt6 = DataFrame(:a => [:a1, :a1, :a2, :a2], :x => [:yes, :no, :yes, :no], :Prob => [[0.6, 0.4], [0.6, 0.7], [0.1, 0.2], [0.8, 0.9]])
            cpt7 = DataFrame(:a => [:a1, :a1, :a2, :a2], :x => [:yes, :no, :yes, :no], :Prob => [[0.2, 0.4, 0.1], [0.6, 0.7], [0.1, 0.2], [0.8, 0.9]])
            cpt8 = DataFrame(:a => [:a1, :a1, :a2, :a2], :x => [:yes, :no, :yes, :no], :Prob => [[0.6, 0.7], [0.6, 0.7], [0.1, 0.2], [0.8, 0.9]])
            cpt9 = DataFrame(:a => [:a1, :a1, :a2, :a2], :x => [:yes, :no, :yes, :no], :Prob => [[0.1, 0.2], [0.6, 0.7], [0.1, 0.2], [0.8, 0.9]])

            @test_throws ErrorException("interval probabilities must lower bound smaller than upper bound. $cpt6") DiscreteNode(name, cpt6)
            @test_throws ErrorException("interval probabilities must be defined with a 2-values vector. $cpt7") DiscreteNode(name, cpt7)
            error_cpt = DataFrame(:a => [:a1, :a1], :x => [:yes, :no], :Prob => [[0.6, 0.7], [0.6, 0.7]])
            @test_throws ErrorException("sum of intervals lower bounds is bigger than 1: $error_cpt") DiscreteNode(name, cpt8)
            error_cpt = DataFrame(:a => [:a1, :a1], :x => [:yes, :no], :Prob => [[0.1, 0.2], [0.6, 0.7]])
            @test_throws ErrorException("sum of intervals upper bounds is smaller than 1: $error_cpt") DiscreteNode(name, cpt9)

            cpt4 = DataFrame(:a => [:a1, :a1, :a2, :a2], :x => [:yes, :no, :yes, :noo], :Prob => [0.5, 0.5, 0.6, 0.4])
            @test isnothing(EnhancedBayesianNetworks._verify_parameters(cpt, parameters, name))
            @test_throws ErrorException("parameters keys [:yes, :no] must be coherent with states [:yes, :no, :noo]") EnhancedBayesianNetworks._verify_parameters(cpt4, parameters, name)

            name1 = :a
            parameters = Dict(:yes => [Parameter(2, :d)], :no => [Parameter(0, :d)])
            cpt = DataFrame(:y => [:a1, :a1, :a2, :a2], :x => [:yes, :no, :yes, :no], :Prob => [0.5, 0.5, 0.6, 0.4])
            @test_throws ErrorException("defined cpt does not contain a column refered to node name a: $cpt") DiscreteNode(name1, cpt, parameters)

            cpt1 = DataFrame(:y => [:a1, :a1, :a2, :a2], :x => [:yes, :no, :yes, :no], :Prob => [0.5, 0.5, 0.6, 0.4])
            cpt2 = DataFrame(:y => [:a1, :a1, :a2, :a2], :x => [:yes, :no, :yes, :no], :Prob => [[0.2, 0.3], [0.7, 0.8], [0.4, 0.6], [0.4, 0.6]])

            node1 = DiscreteNode(name, cpt1, parameters)
            node2 = DiscreteNode(name, cpt1)
            node3 = DiscreteNode(name, cpt2, parameters)
            node4 = DiscreteNode(name, cpt2)

            @test node1.name == name
            @test node1.parameters == parameters
            @test node1.cpt == cpt1
            @test node2.name == name
            @test isempty(node2.parameters)
            @test node2.cpt == cpt1
            @test node3.name == name
            @test node3.parameters == parameters
            @test node3.cpt == cpt2
            @test node4.name == name
            @test isempty(node4.parameters)
            @test node4.cpt == cpt2
        end

        @testset "node functions" begin
            name = :x
            cpt = DataFrame(:a => [:a1, :a1, :a2, :a2], :x => [:yes, :no, :yes, :no], :Prob => [[0.2, 0.3], [0.7, 0.8], [0.4, 0.6], [0.4, 0.6]])
            node = DiscreteNode(name, cpt)
            @test issetequal(EnhancedBayesianNetworks._states(cpt, name), [:yes, :no])
            @test issetequal(EnhancedBayesianNetworks._states(node), [:yes, :no])

            scenarios = [Dict(:a => :a1), Dict(:a => :a2)]
            @test EnhancedBayesianNetworks._scenarios(cpt, name) == scenarios
            @test EnhancedBayesianNetworks._scenarios(node) == scenarios
        end
    end
end