include("../src/bn.jl")

## Uᵣ_node
Uᵣ_distributions = Normal()
Uᵣ_CPD = RootCPD(:Uᵣ, [Uᵣ_distributions])
Uᵣ_node = RootNode(Uᵣ_CPD)

## External forces Nodes
H_distribution = Gumbel(50, 0.4)
H_CPD = RootCPD(:H, [H_distribution])
H_node = RootNode(H_CPD)

## DiscreteNode 1
timescenario = NamedCategorical([:first, :second], [0.9, 0.1])
CPD_timescenario = RootCPD(:timescenario, [timescenario])
node_timescenario = RootNode(CPD_timescenario)


V_parents = name(node_timescenario)
parental_ncategories = [2]
V_distribution = [Gamma(60, 0.2), Gamma(50, 0.2)]
V_CPD = StdCPD(:V, [V_parents], parental_ncategories, V_distribution)
V_node = StdNode(V_CPD, [node_timescenario])
# v1 = [
#     ModelParameters(:E, [:model1], [Parameter(2, :ro)]),
#     ModelParameters(:V, [:model3], [Parameter(5.8, :fu)])
# ]
# v2 = [
#     ModelParameters(:E, [:model2], [Parameter(3, :ro)]),
#     ModelParameters(:V, [:model3, :model4], [Parameter(2,:ro), Parameter(5.8, :fu)])
# ]
# parameters_vector = [v1, v2]

# node_V = StdNode(V_CPD, [node_timescenario], parameters_vector)


## DiscreteNode 2
emission = NamedCategorical([:nothappen, :happen], [0.0, 1.0])
CPD_emission = RootCPD(:emission, [emission])
emission1 = [
    ModelParameters(:E, :failure1, [Parameter(2, :ro), Parameter(3, :i), Parameter(9, :l)]),
    ModelParameters(:PO, :failure3, [Parameter(5.8, :fu)])
]
emission2 = [
    ModelParameters(:E, :failure2, [Parameter(3, :ro), Parameter(3, :i), Parameter(9, :l)]),
    ModelParameters(:PO, :failure4, [Parameter(2, :ro)])
]
parameters_vector = [emission1, emission2]
node_emission = RootNode(CPD_emission, parameters_vector)

ρᵣ = 0.3
λᵣ = 150
cov = 0.2
ζᵣ = cov * λᵣ

function cpd_f_given_ur(df)
    return r -> cdf(Normal(), (log(r) - df.Uᵣ * sqrt(ρᵣ) - λᵣ) / sqrt(ζᵣ^2 - ρᵣ))
end
model_f = ModelWithName(:model_f, [Model(cpd_f_given_ur, :f)])
f_CPD = FunctionalCPD(:f, name.([Uᵣ_node]), [1], [model_f])
f_node = FunctionalNode(f_CPD, [Uᵣ_node], "continuous")

function cpd_o_given_f(f)
    return r -> cdf(Normal(), (log(r) - f * sqrt(ρᵣ) - λᵣ) / sqrt(ζᵣ^2 - ρᵣ))
end
model_o1 = ModelWithName(:model_o1, [Model(cpd_o_given_f, :o1)])
o1_CPD = FunctionalCPD(:o1, name.([f_node]), [1], [model_o1])
o1_node = FunctionalNode(o1_CPD, [f_node], "continuous")

function cpd_o_given_ur(ur)
    return r -> cdf(Normal(), (log(r) - ur * sqrt(ρᵣ) - λᵣ) / sqrt(ζᵣ^2 - ρᵣ))
end
model_o2 = ModelWithName(:model_o2, [Model(cpd_o_given_ur, :o2)])
o2_CPD = FunctionalCPD(:o2, name.([Uᵣ_node]), [1], [model_o2])
o2_node = FunctionalNode(o2_CPD, [Uᵣ_node], "continuous")

## DiscreteNode 2
# emission_ts1 = NamedCategorical([:nothappen, :happen], [0.0, 1.0])
# emission_ts2 = NamedCategorical([:nothappen, :happen], [0.0, 0.1])
# CPD_emission = StdCPD(:emission, [name(node_timescenario)], [2], [emission_ts1, emission_ts2])
# emission1 = [
#     ModelParameters(:E, [:model1], [[Parameter(2, :ro)]]),
#     ModelParameters(:PO, [:model3, :model4], [[Parameter(5.8, :fu)], [Parameter(8, :t)]])
# ]
# emission2 = [
#     ModelParameters(:E, [:model2], [[Parameter(3, :ro)]]),
#     ModelParameters(:PO, [:model4, :model6], [[Parameter(2, :ro)], [Parameter(9, :fu)]])
# ]
# parameters_vector = [emission1, emission2]
# node_emission = StdNode(CPD_emission, [node_timescenario], parameters_vector)

## R Nodes
parentsᵣ = [Uᵣ_node]
parental_ncategoriesᵣ = Vector{Int}()


## Function for returning cpd of node Rᵢ ∀ i in [1;5]
function cpd_r_given_o1(o1)
    return r -> cdf(Normal(), (log(r) - o1 * sqrt(ρᵣ) - λᵣ) / sqrt(ζᵣ^2 - ρᵣ))
end

## node R₁
model₁ = ModelWithName(:model1, [Model(cpd_r_given_o1, :R₁)])
R₁_CPD = FunctionalCPD(:R₁, name.([o1_node]), parental_ncategoriesᵣ, [model₁])
R₁_node = FunctionalNode(R₁_CPD, [o1_node], "continuous")

function cpd_r_given_o2(o2)
    return r -> cdf(Normal(), (log(r) - o2 * sqrt(ρᵣ) - λᵣ) / sqrt(ζᵣ^2 - ρᵣ))
end

model₂ = ModelWithName(:model2, [Model(cpd_r_given_o2, :R₂)])
R₂_CPD = FunctionalCPD(:R₂, name.([o2_node]), parental_ncategoriesᵣ, [model₂])
R₂_node = FunctionalNode(R₂_CPD, [o2_node], "continuous")

## node R₂
function cpd_r_given_u(ur)
    return r -> cdf(Normal(), (log(r) - ur * sqrt(ρᵣ) - λᵣ) / sqrt(ζᵣ^2 - ρᵣ))
end
## node R₃
model₃ = ModelWithName(:model3, [Model(cpd_r_given_u, :R₃)])
R₃_CPD = FunctionalCPD(:R₃, name.(parentsᵣ), parental_ncategoriesᵣ, [model₃])
R₃_node = FunctionalNode(R₃_CPD, parentsᵣ, "continuous")
## node R₁
model₄ = ModelWithName(:model4, [Model(cpd_r_given_u, :R₄)])
R₄_CPD = FunctionalCPD(:R₄, name.(parentsᵣ), parental_ncategoriesᵣ, [model₄])
R₄_node = FunctionalNode(R₄_CPD, parentsᵣ, "continuous")
## node R₁
model₅ = ModelWithName(:model5, [Model(cpd_r_given_u, :R₅)])
R₅_CPD = FunctionalCPD(:R₅, name.(parentsᵣ), parental_ncategoriesᵣ, [model₅])
R₅_node = FunctionalNode(R₅_CPD, parentsᵣ, "continuous")



## Node E - as 1 single model
function failure_1(r₁, r₂, r₅, r₄, emission_par)
    return r₁ + r₂ + r₄ + r₅ - 5 * h - emission_par
end
function failure_2(r₁, r₂, r₅, r₄, emission_par)
    return r₁ - r₂ - r₄ + 4r₅ - 5 * h - emission_par
end

function cpd_r_given_parents1(r₁, r₂, r₃, r₄, r₅, h, v, failure1)
    g1 = failure1(r₁, r₂, r₅, r₄)
    g2 = r₂ + 2 * r₅ + r₄ - 5 * v
    g3 = r₁ + 2 * r₃ + 2 * r₄ + r₅ - 5 * h - 5 * v
    min(g1, g2, g3) ≤ 0 ? output = true : output = false
    return output
end

function cpd_r_given_parents2(r₁, r₂, r₃, r₄, r₅, h, v, failure2)
    g1 = failure2(r₁, r₂, r₅, r₄)
    g2 = r₂ + 2 * r₅ + r₄ - 5 * v
    g3 = r₁ + 2 * r₃ + 2 * r₄ + r₅ - 5 * h - 5 * v
    min(g1, g2, g3) ≤ 0 ? output = true : output = false
    return output
end

parental_ncategoriesₑ = [2]
parentsₑ = [V_node, H_node, R₁_node, R₂_node, R₃_node, R₄_node, R₅_node, node_emission]
failuremodel1 = Model(failure_1, :f1)
outputmodel1 = Model(cpd_r_given_parents1, :E)
modelₑ1 = ModelWithName(:failure1, [failuremodel1, outputmodel1])
failuremodel2 = Model(failure_2, :f1)
outputmodel2 = Model(cpd_r_given_parents2, :E)
modelₑ2 = ModelWithName(:failure2, [failuremodel2, outputmodel2])


E_CPD = FunctionalCPD(:E, name.(parentsₑ), parental_ncategoriesₑ, [modelₑ1, modelₑ2])
E_node = FunctionalNode(E_CPD, parentsₑ, "discrete")

# ebn = EnhancedBayesNet([Uᵣ_node, V_node, H_node, R₁_node, R₂_node, R₃_node, R₄_node, R₅_node, E_node, node_emission, node_timescenario])
ebn = EnhancedBayesNet([Uᵣ_node, f_node, o1_node, o2_node, V_node, H_node, R₁_node, R₂_node, R₃_node, R₄_node, R₅_node, E_node, node_emission, node_timescenario])
show(ebn)
groups = markov_envelopes(ebn)
rdag, dag_names = _reduce_ebn_to_rbn(ebn)
graphplot(
    rdag,
    method=:tree,
    names=dag_names,
    fontsize=9,
    nodeshape=:ellipse,
    markercolor=map(x -> x.type == "discrete" ? "lightgreen" : "orange", filter(x -> x.cpd.target ∈ dag_names, ebn.nodes)),
    linecolor=:darkgrey,
)


empty_srp_table_with_evidence = _build_node_evidence_after_reduction(ebn, rdag, dag_names, E_node)
r_nodes = _get_node_in_rbn(ebn)
# nodes_to_be_evaluated = filter(x -> isa(x, FunctionalNode), r_nodes)
# a, b = _get_ancestors_distribution_4sampling(nodes_to_be_evaluated[1])
# # node = nodes_to_be_evaluated[1]
empty_srp = empty_srp_table_with_evidence[1]
# # parent_node = V_node
# # a = _build_srp_single_node_single_evidence(ebn, empty_srp.evidence, parent_node)
single_struc_table = empty_srp
node = E_node


# uqinputs = _build_uqinputs_vector_single_evidence(ebn, empty_srp, E_node)

a = _functional_node_after_reduction(ebn, empty_srp_table_with_evidence, E_node)
df = UncertaintyQuantification.sample(a[1].srp[2].inputs, 2)

aux = a[1].srp[2].ancestors_models[1]