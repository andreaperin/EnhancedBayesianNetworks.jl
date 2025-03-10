@testset "Discrete Nodes" begin

    @testset "Root Nodes" begin
        name = :a
        cpt0 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(name)
        cpt0[:a=>:yes] = -0.5
        cpt0[:a=>:no] = 0.5
        @test_throws ErrorException("probabilities must be non-negative: -0.5") DiscreteNode(name, cpt0)

        cpt1 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(name)
        cpt1[:a=>:yes] = 1.5
        cpt1[:a=>:no] = 0.5
        @test_throws ErrorException("probabilities must be lower or equal than 1: 1.5") DiscreteNode(name, cpt1)

        cpt2 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}(name)
        cpt2[:a=>:yes] = (-0.5, 0.1)
        cpt2[:a=>:no] = (0.5, 0.9)
        @test_throws ErrorException("probabilities must be non-negative: -0.5") DiscreteNode(name, cpt2)

        cpt3 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}(name)
        cpt3[:a=>:yes] = (0.6, 1.5)
        cpt3[:a=>:no] = (0.4, 0.5)
        @test_throws ErrorException("probabilities must be lower or equal than 1: 1.5") DiscreteNode(name, cpt3)

        cpt4 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}(name)
        cpt4[:a=>:yes] = (0.6, 0.8)
        cpt4[:a=>:no] = (0.1, 0.15)
        @test_throws ErrorException("sum of intervals upper bounds is smaller than 1: $(cpt4.data)") DiscreteNode(name, cpt4)

        cpt5 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}(name)
        cpt5[:a=>:yes] = (0.6, 0.8)
        cpt5[:a=>:no] = (0.5, 0.6)
        @test_throws ErrorException("sum of intervals lower bounds is bigger than 1: $(cpt5.data)") DiscreteNode(name, cpt5)

        cpt6 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}(name)
        cpt6[:a=>:yes] = (0.3, 0.5)
        cpt6[:a=>:no] = (0.5, 0.7)
        parameters = Dict(:y => [Parameter(1, :a)], :no => [Parameter(2, :a)])
        @test_throws ErrorException("parameters keys [:y, :no] must be coherent with states [:yes, :no]") DiscreteNode(name, cpt6, parameters)

        parameters = Dict(:yes => [Parameter(1, :a)], :no => [Parameter(2, :a)])
        node = DiscreteNode(name, cpt6, parameters)
        @test node.name == name
        @test node.cpt == cpt6
        @test node.parameters == parameters
        @test node.additional_info == Dict{AbstractVector{Symbol},Dict}()

        @suppress @test_throws ErrorException("defined cpt does not contain a column refered to node name x: $cpt6") DiscreteNode(:x, cpt6)

        node = DiscreteNode(name, cpt6)
        @test node.name == name
        @test node.cpt == cpt6
        @test node.parameters == Dict{Symbol,Vector{Parameter}}()
        @test node.additional_info == Dict{AbstractVector{Symbol},Dict}()

        cpt7 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(name)
        cpt7[:a=>:yes] = 0.4999
        cpt7[:a=>:no] = 0.5
        @suppress node = DiscreteNode(name, cpt7)
        @test all(isapprox.(node.cpt.data[:, :Π], [0.50005, 0.49995]))

        @testset "nodes functions" begin
            node = DiscreteNode(name, cpt6, parameters)

            @test states(node) == [:no, :yes]
            @test scenarios(node) == Any[]
            @test EnhancedBayesianNetworks._scenarios_cpt(node) == [cpt6.data]
        end
    end

    @testset "Child Nodes" begin
        cpt0 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:x, :y])
        cpt0[:x=>:yesx, :y=>:yesy] = -0.5
        cpt0[:x=>:yesx, :y=>:noy] = 0.5
        cpt0[:x=>:nox, :y=>:yesy] = 0.5
        cpt0[:x=>:nox, :y=>:noy] = 0.5

        @test_throws ErrorException("probabilities must be non-negative: -0.5") DiscreteNode(:y, cpt0)

        cpt1 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:x, :y])
        cpt1[:x=>:yesx, :y=>:yesy] = 1.5
        cpt1[:x=>:yesx, :y=>:noy] = 0.5
        cpt1[:x=>:nox, :y=>:yesy] = 0.5
        cpt1[:x=>:nox, :y=>:noy] = 0.5

        @test_throws ErrorException("probabilities must be lower or equal than 1: 1.5") DiscreteNode(:y, cpt1)

        cpt2 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}([:x, :y])
        cpt2[:x=>:yesx, :y=>:yesy] = (-0.5, 0.5)
        cpt2[:x=>:yesx, :y=>:noy] = (0.3, 0.7)
        cpt2[:x=>:nox, :y=>:yesy] = (0.5, 0.8)
        cpt2[:x=>:nox, :y=>:noy] = (0.2, 0.5)

        @test_throws ErrorException("probabilities must be non-negative: -0.5") DiscreteNode(:y, cpt2)

        cpt3 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}([:x, :y])
        cpt3[:x=>:yesx, :y=>:yesy] = (0.5, 1.5)
        cpt3[:x=>:yesx, :y=>:noy] = (0.3, 0.7)
        cpt3[:x=>:nox, :y=>:yesy] = (0.5, 0.8)
        cpt3[:x=>:nox, :y=>:noy] = (0.2, 0.5)

        @test_throws ErrorException("probabilities must be lower or equal than 1: 1.5") DiscreteNode(:y, cpt3)

        cpt4 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}([:x, :y])
        cpt4[:x=>:yesx, :y=>:yesy] = (0.5, 0.3)
        cpt4[:x=>:yesx, :y=>:noy] = (0.3, 0.7)
        cpt4[:x=>:nox, :y=>:yesy] = (0.5, 0.8)
        cpt4[:x=>:nox, :y=>:noy] = (0.2, 0.5)

        @test_throws ErrorException("interval probabilities must have a lower bound smaller than upper bound. $(cpt4.data)") DiscreteNode(:y, cpt4)

        cpt5 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}([:x, :y])
        cpt5[:x=>:yesx, :y=>:yesy] = (0.2, 0.3)
        cpt5[:x=>:yesx, :y=>:noy] = (0.3, 0.4)
        cpt5[:x=>:nox, :y=>:yesy] = (0.5, 0.8)
        cpt5[:x=>:nox, :y=>:noy] = (0.2, 0.5)

        res = EnhancedBayesianNetworks._scenarios_cpt(cpt5, :y)[1]
        @test_throws ErrorException("sum of intervals upper bounds is smaller than 1: $res") DiscreteNode(:y, cpt5)

        cpt6 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}([:x, :y])
        cpt6[:x=>:yesx, :y=>:yesy] = (0.8, 0.9)
        cpt6[:x=>:yesx, :y=>:noy] = (0.3, 0.4)
        cpt6[:x=>:nox, :y=>:yesy] = (0.5, 0.8)
        cpt6[:x=>:nox, :y=>:noy] = (0.2, 0.5)

        res = EnhancedBayesianNetworks._scenarios_cpt(cpt6, :y)[1]
        @test_throws ErrorException("sum of intervals lower bounds is bigger than 1: $res") DiscreteNode(:y, cpt6)

        cpt7 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}([:x, :y])
        cpt7[:x=>:yesx, :y=>:yesy] = (0.3, 0.5)
        cpt7[:x=>:yesx, :y=>:noy] = (0.5, 0.7)
        cpt7[:x=>:nox, :y=>:yesy] = (0.5, 0.8)
        cpt7[:x=>:nox, :y=>:noy] = (0.2, 0.5)
        parameters = Dict(:yes => [Parameter(1, :a)], :no => [Parameter(2, :a)])

        @test_throws ErrorException("parameters keys [:yes, :no] must be coherent with states [:yesy, :noy]") DiscreteNode(:y, cpt7, parameters)

        parameters = Dict(:yesy => [Parameter(1, :a)], :noy => [Parameter(2, :a)])
        node = DiscreteNode(:y, cpt7, parameters)
        @test node.name == :y
        @test node.cpt == cpt7
        @test node.parameters == parameters
        @test node.additional_info == Dict{AbstractVector{Symbol},Dict}()

        node = DiscreteNode(:y, cpt7)
        @test node.name == :y
        @test node.cpt == cpt7
        @test node.parameters == Dict{Symbol,Vector{Parameter}}()
        @test node.additional_info == Dict{AbstractVector{Symbol},Dict}()

        cpt8 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:x, :y])
        cpt8[:x=>:yesx, :y=>:yesy] = 0.49999
        cpt8[:x=>:yesx, :y=>:noy] = 0.5
        cpt8[:x=>:nox, :y=>:yesy] = 0.8
        cpt8[:x=>:nox, :y=>:noy] = 0.2
        @suppress node = DiscreteNode(:y, cpt8)
        @test all(isapprox.(node.cpt.data[:, :Π], [0.2, 0.8, 0.500005, 0.499995]))

        @testset "nodes functions" begin
            node = DiscreteNode(:y, cpt7, parameters)

            @test issetequal(states(node), [:noy, :yesy])
            @test issetequal(scenarios(node), [Dict(:x => :yesx), Dict(:x => :nox)])
            @test issetequal(EnhancedBayesianNetworks._scenarios_cpt(node), EnhancedBayesianNetworks._scenarios_cpt(cpt7, :y))

            @test isprecise(node) == false
            node2 = DiscreteNode(:y, cpt8, parameters)
            @test isprecise(node2)
        end
    end

    @testset "Extreme Points Root" begin

        cpt1 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:x)
        cpt1[:x=>:yesx] = 0.2
        cpt1[:x=>:nox] = 0.8
        node1 = DiscreteNode(:x, cpt1)

        cpt2 = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}(:x)
        cpt2[:x=>:yesx] = (0.1, 0.2)
        cpt2[:x=>:nox] = (0.8, 0.9)
        node2 = DiscreteNode(:x, cpt2)

        @test_throws ErrorException("Precise conditional probability table does not have extreme points: $(cpt1.data)") EnhancedBayesianNetworks._extreme_points_probabilities(cpt1.data)

        extreme_point_probabilities = EnhancedBayesianNetworks._extreme_points_probabilities(cpt2.data)
        @test all(isapprox.(extreme_point_probabilities[1], [0.8, 0.2], atol=0.001))
        @test all(isapprox.(extreme_point_probabilities[2], [0.9, 0.1], atol=0.001))

        extreme_point_dfs = EnhancedBayesianNetworks._extreme_points_dfs(cpt2.data)
        @test isapprox(extreme_point_dfs[1][!, :Π][1], 0.8, atol=0.05)
        @test isapprox(extreme_point_dfs[1][!, :Π][2], 0.2, atol=0.05)
        @test isapprox(extreme_point_dfs[2][!, :Π][1], 0.9, atol=0.05)
        @test isapprox(extreme_point_dfs[2][!, :Π][2], 0.1, atol=0.05)

        node = DiscreteNode(:x, cpt2)
        ext_nodes = EnhancedBayesianNetworks._extreme_points(node)

        node1 = DiscreteNode(:x, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(extreme_point_dfs[1]))
        node2 = DiscreteNode(:x, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(extreme_point_dfs[2]))
        @test ext_nodes[1].name == node1.name
        @test ext_nodes[1].cpt.data[!, :x] == node1.cpt.data[!, :x]
        @test all(isapprox.(ext_nodes[1].cpt.data[!, :Π], node1.cpt.data[!, :Π]; atol=0.01))

        @test ext_nodes[2].name == node2.name
        @test ext_nodes[2].cpt.data[!, :x] == node2.cpt.data[!, :x]
        @test all(isapprox.(ext_nodes[2].cpt.data[!, :Π], node2.cpt.data[!, :Π]; atol=0.01))
    end

    @testset "Extreme Points Child" begin end

end
