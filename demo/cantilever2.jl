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


c1 = GaussianCopula([1 0.8; 0.8 1])
c2 = GaussianCopula([1 0.7; 0.7 1])
f1 = (E, ρ) -> JointDistribution([E, ρ], c1)
f2 = (E, ρ) -> JointDistribution([E, ρ], c2)


parents_jd = name.([E_node, ρ_node, emission_node])
target = :jd
parental_ncategories = [2]
prob_dict_jd = [ProbabilityDictionaryFunctional((Dict(:emission => 1), SystemReliabilityProblem(f1))),
    ProbabilityDictionaryFunctional((Dict(:emission => 2), SystemReliabilityProblem(f2)))]

CPD_jd = FunctionalCPD(:jd, parents_jd, parental_ncategories, prob_dict_jd)
jd_node = FunctionalNode(CPD_jd, [E_node, ρ_node, emission_node], "continuous")

### Evaluating node_jd Distribution
nodes = [emission_node, h_node, E_node, P_node, ρ_node, jd_node]
dag = _build_DiAGraph_from_nodes(nodes)
ordered_cpds, ordered_nodes, ordered_name_to_index, ordered_dag = _topological_ordered_dag(nodes)
bn = EnhancedBayesNet(ordered_dag, ordered_nodes, ordered_cpds, ordered_name_to_index)
show(bn)

