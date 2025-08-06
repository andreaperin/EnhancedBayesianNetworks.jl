@testset "Parameters Learning - MLE" begin
    df = CSV.read("learning/parameters_learning/sprinkler.csv", DataFrame)
    df_m = CSV.read("learning/parameters_learning/sprinkler_missing.csv", DataFrame)

    bn = BayesianNetwork2be([:Weather, :Sprinkler, :Rain, :Grass])
    add_child!(bn, :Weather, :Sprinkler)
    add_child!(bn, :Weather, :Rain)
    add_child!(bn, :Sprinkler, :Grass)
    add_child!(bn, :Rain, :Grass)
    order!(bn)

    states_space = Dict(
        :Weather => [:Cloudy, :Sunny],
        :Rain => [:Yes, :No],
        :Sprinkler => [:On, :Off],
        :Grass => [:Wet, :Dry]
    )
    states_space2 = Dict(
        :Weather => [:Cloudy, :Sunny],
        :Rain => [:Yes, :No, :Pooring],
        :Sprinkler => [:On, :Off],
        :Grass => [:Wet, :Dry]
    )
    states_space3 = Dict(
        :Weather => [:Cloudy, :Sunny],
        :Rain => [:Yes, :No],
        :Sprinkler => [:On, :Off],
        :Grass => [:Wet, :Dry, :New]
    )

    @testset "weighted probs subgroup" begin

        net = deepcopy(bn)
        node = :Grass
        pars = parents(net, node)[2]
        grouped = groupby(Symbol.(df), [pars...]) |> collect
        res1 = EnhancedBayesianNetworks._weighted_counts_and_probs_subgroup(grouped[1], states_space, node, pars, :weight)
        @test res1[1] == (Sprinkler=:Off, Rain=:No)
        @test res1[2].Rain == [:No, :No]
        @test res1[2].Sprinkler == [:Off, :Off]
        @test res1[2].Grass == [:Dry, :Wet]
        @test res1[2].Π == [261, 0]
        @test res1[3].Rain == [:No, :No]
        @test res1[3].Sprinkler == [:Off, :Off]
        @test res1[3].Grass == [:Dry, :Wet]
        @test res1[3].Π == [1.0, 0.0]

        node = :Weather
        pars = parents(net, node)[2]
        res2 = EnhancedBayesianNetworks._weighted_counts_and_probs_subgroup(grouped[1], states_space, node, pars, :weight)
        @test res2[1] == NamedTuple()
        @test res2[2].Weather == [:Sunny, :Cloudy]
        @test res2[2].Π == [179, 82]
        @test res2[3].Weather == [:Sunny, :Cloudy]
        @test isapprox(res2[3].Π, [0.685823754789272, 0.31417624521072796])

        net = deepcopy(bn)
        node = :Rain
        pars = parents(net, node)[2]
        grouped = groupby(Symbol.(df), [pars...]) |> collect
        res1 = EnhancedBayesianNetworks._weighted_counts_and_probs_subgroup(grouped[1], states_space2, node, pars, :weight)
    end

    @testset "counts and probs" begin
        net = deepcopy(bn)
        node = :Rain
        res1 = EnhancedBayesianNetworks._counts_and_probs(Symbol.(df), node, parents(net, node)[2], states_space)
        @test res1[1].Weather == [:Sunny, :Sunny, :Cloudy, :Cloudy]
        @test res1[1].Rain == [:No, :Yes, :Yes, :No]
        @test res1[1].Π == [394, 94, 421, 91]
        @test res1[2].Weather == [:Sunny, :Sunny, :Cloudy, :Cloudy]
        @test res1[2].Rain == [:No, :Yes, :Yes, :No]
        @test isapprox(res1[2].Π, [0.8073770491803278, 0.19262295081967212, 0.822265625, 0.177734375])

        node = :Grass
        @test_throws ErrorException("provided dataframe does not contain the following states for the following nodes Dict{Any, Any}(:Rain => [:Pooring])") EnhancedBayesianNetworks._counts_and_probs(Symbol.(df), node, parents(net, node)[2], states_space2)
    end

    @testset "Initialize cpts" begin
        @test_throws ErrorException("not complete DataFrame") EnhancedBayesianNetworks._initialize_cpt(df_m, bn, states_space)

        cpts = EnhancedBayesianNetworks._initialize_cpt(df, bn, states_space)
        @test cpts[1][1] == :Weather
        @test isa(cpts[1][2], DiscreteConditionalProbabilityTable)
        @test issetequal(cpts[1][2].data.Weather, [:Sunny, :Cloudy])
        @test isapprox(cpts[1][2].data.Π, [0.488, 0.512], atol=0.001)

        @test cpts[2][1] == :Sprinkler
        @test isa(cpts[2][2], DiscreteConditionalProbabilityTable)
        @test issetequal(cpts[2][2].data.Sprinkler, [:Off, :On, :Off, :On])
        @test isapprox(cpts[2][2].data.Π, [0.461066, 0.53893, 0.910156, 0.0898438], atol=0.001)

        @test cpts[3][1] == :Rain
        @test isa(cpts[3][2], DiscreteConditionalProbabilityTable)
        @test issetequal(cpts[3][2].data.Rain, [:Yes, :No, :Yes, :No])
        @test isapprox(cpts[3][2].data.Π, [0.8073770, 0.192622, 0.822265625, 0.177734375], atol=0.001)

        @test cpts[4][1] == :Grass
        @test isa(cpts[4][2], DiscreteConditionalProbabilityTable)
        @test issetequal(cpts[4][2].data.Grass, [:Dry, :Wet, :Dry, :Wet, :Dry, :Wet, :Dry, :Wet])
        @test isapprox(cpts[4][2].data.Π, [1.0, 0.0, 0.8860465116279069, 0.11395348837209303, 0.84375, 0.15625, 0.9764705882352941, 0.023529411764705882], atol=0.001)

        @test_throws ErrorException("provided dataframe does not contain the following states for the following nodes Dict{Any, Any}(:Rain => [:Pooring])") EnhancedBayesianNetworks._initialize_cpt(df, bn, states_space2)
    end

    @testset "learn_parameters_MLE" begin
        net1 = deepcopy(bn)
        net2 = deepcopy(bn)
        net3 = deepcopy(bn)
        net4 = deepcopy(bn)

        bn2 = BayesianNetwork2be([:Weather, :Sprinkler, :Rain, :River])
        @test_throws ErrorException("nodes provided in the dataframe are not coherent with the ones provided in the network") learn_parameters_MLE(df, bn2)

        bn1 = learn_parameters_MLE(df, net1)
        bn2 = learn_parameters_MLE(df, net2, states_space)
        @test bn1 == bn2

        @test_throws ErrorException("provided dataframe does not contain the following states for the following nodes Dict{Any, Any}(:Rain => [:Pooring])") learn_parameters_MLE(df, net3, states_space2)

        bn3 = learn_parameters_MLE(df, net4, states_space3)
        @test issetequal(states(bn3.nodes[end]), states_space3[:Grass])
    end
end