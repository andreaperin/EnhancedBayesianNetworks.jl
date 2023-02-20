using UncertaintyQuantification
include("../bn.jl")

emission = NamedCategorical([:nothappen, :happen], [0.3, 0.7])
CPD_emission = RootCPD(:emission, emission)
emission_node = StdNode(CPD_emission)

h_distribution = Normal(0.24, 0.01)
CPD_h = RootCPD(:h, h_distribution)
h_node = StdNode(CPD_h)

μ = log(10e9^2 / sqrt(1.6e9^2 + 10e9^2))
σ = sqrt(log(1.6e9^2 / 10e9^2 + 1))
E_distribution = LogNormal(μ, σ)
CPD_E = RootCPD(:E, E_distribution)
E_node = StdNode(CPD_E)

μ = log(5000^2 / sqrt(400^2 + 5000^2))
σ = sqrt(log(400^2 / 5000^2 + 1))
P_distribution = LogNormal(μ, σ)
CPD_P = RootCPD(:P, P_distribution)
P_node = StdNode(CPD_P)

μ = log(600^2 / sqrt(140^2 + 600^2))
σ = sqrt(log(140^2 / 600^2 + 1))
ρ_distribution = LogNormal(μ, σ)
CPD_ρ = RootCPD(:ρ, ρ_distribution)
ρ_node = StdNode(CPD_ρ)




##TODO Continue from here implementing dependency between rvs as an argument of SRP states_dictionary


## Output Node FunctionalCPD
output_target = :output
output_parents = [emission_node, E_node, ρ_node, P_node, h_node]
output_parental_ncat = [2]
l = Parameter(1.8, :l) # length
b = Parameter(0.12, :b) # width
inertia = Model(df -> df.b .* df.h .^ 3 / 12, :I)
displacement = Model(
    df ->
        (df.ρ .* 9.81 .* df.b .* df.h .* df.l .^ 4) ./ (8 .* df.E .* df.I) .+
        (df.P .* df.l .^ 3) ./ (3 .* df.E .* df.I),
    :w,
)
## Scenario1 (Emission_node = 1)
model1 = [inertia, displacement]
performance1 = df -> max_displacement .- df.w
parameters1 = [l, b]
correlated_nodes1 = name.([E_node, ρ_node])
copula1 = GaussianCopula([1 0.8; 0.8 1])
name1 = :jd
correlation1 = [CorrelationCopula(correlated_nodes1, copula1, :jd)]
srp1 = SystemReliabilityProblem(model1, parameters1, performance1, correlation1)
scenario1 = ProbabilityDictionaryFunctional((Dict(name(emission_node) => 1), srp1))
## Scenario1 (Emission_node = 2)
model2 = [inertia, displacement]
performance2 = df -> max_displacement .- df.w
parameters2 = [l, b]
correlated_nodes2 = name.([E_node, ρ_node])
copula2 = GaussianCopula([1 0.85; 0.85 1])
name2 = :jd
correlation2 = [CorrelationCopula(correlated_nodes2, copula2, :jd)]
srp2 = SystemReliabilityProblem(model2, parameters2, performance2, correlation2)
scenario2 = ProbabilityDictionaryFunctional((Dict(name(emission_node) => 2), srp1))
srp2 = SystemReliabilityProblem(model2, parameters2, performance2, correlation2)

scenario2 = ProbabilityDictionaryFunctional((Dict(name(emission_node) => 2), srp2))

prob_dict_output = [scenario1, scenario2]

CPD_output = FunctionalCPD(output_target, name.(output_parents), output_parental_ncat, prob_dict_output)

output_node = FunctionalNode(CPD_output, output_parents, "discrete")





nodes = [output_node, E_node, ρ_node, P_node, h_node, emission_node]
ebn = EnhancedBayesNet(nodes)
show(ebn)




""" Solving EnhancedBayesNet (No undefined continuous parents case) """



function evaluate_rvs(ebn::EnhancedBayesNet, node::FunctionalNode, srp::SystemReliabilityProblem)
    continuous_parents = get_continuous_parents(node)
    non_correlated_continuous_parents = Vector{UQInput}()
    rvs = Vector{UQInput}()
    for copula in srp.correlation
        continuous_parents = filter(n -> n in name.(continuous_parents, copula.nodes))
        if isa(rv.cpd, RootCPD)
            get_rv = (target, x) -> RandomVariable(filter(x -> name(x) == target, x)[1].cpd.distributions, target)
            correlated_distributions = get_rv.(copula.nodes, repeat([continuous_parents], length(copula.nodes)))
            push!(rvs, JointDistribution(correlated_distributions, copula.copula))
        else
            throw(DomainError(copula, "this case needs to be taken into account"))
        end
    end

end



# c1 = GaussianCopula([1 0.8; 0.8 1])
# c2 = GaussianCopula([1 0.7; 0.7 1])
# f1 = (E, ρ) -> JointDistribution([E, ρ], c1)
# fun1 = Dict(
#     ":model" => f1,
#     ":rvs_node" => [E_node, ρ_node]
# )
# f2 = (E, ρ) -> JointDistribution([E, ρ], c2)

# parents_jd = name.([E_node, ρ_node, emission_node])
# target = :jd
# parental_ncategories = [2]
# prob_dict_jd = [ProbabilityDictionaryFunctional((Dict(:emission => 1), SystemReliabilityProblem(fun1))),
#     ProbabilityDictionaryFunctional((Dict(:emission => 2), SystemReliabilityProblem(f2)))]

# CPD_jd = FunctionalCPD(:jd, parents_jd, parental_ncategories, prob_dict_jd)
# jd_node = FunctionalNode(CPD_jd, [E_node, ρ_node, emission_node], "continuous")

# ### Evaluating node_jd Distribution
# nodes = [emission_node, h_node, E_node, P_node, ρ_node, jd_node]
# dag = _build_DiAGraph_from_nodes(nodes)
# ordered_cpds, ordered_nodes, ordered_name_to_index, ordered_dag = _topological_ordered_dag(nodes)
# bn = EnhancedBayesNet(ordered_dag, ordered_nodes, ordered_cpds, ordered_name_to_index)
# show(bn)