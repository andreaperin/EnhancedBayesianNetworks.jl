include("../src/bn.jl")

## Uᵣ_node
Uᵣ_distributions = Normal()
Uᵣ_CPD = RootCPD(:Uᵣ, [Uᵣ_distributions])
Uᵣ_node = RootNode(Uᵣ_CPD)

## External forces Nodes
H_distribution = Gumbel(50, 0.4)
H_CPD = RootCPD(:H, [H_distribution])
H_node = RootNode(H_CPD)

V_distribution = Gamma(60, 0.2)
V_CPD = RootCPD(:V, [V_distribution])
V_node = RootNode(V_CPD)


## R Nodes
ρᵣ = 0.3
γᵣ = 150
ζᵣ = 0.2
parentsᵣ = [Uᵣ_node]
parental_ncategoriesᵣ = Vector{Int}()


## Function for returning cpd of node Rᵢ ∀ i in [1;5]
function cpd_r_given_u(uᵣ)
    return (r) -> cdf(Normal(), (ln(r) - uᵣ * sqrt(ρᵣ) - λᵣ) / sqrt(ζᵣ^2 - ρᵣ))
end

## node R₁
model₁ = [Model(cpd_r_given_u, :R₁)]
R₁_CPD = FunctionalCPD(:R₁, name.(parentsᵣ), parental_ncategoriesᵣ, [model₁])
R₁_node = FunctionalNode(R₁_CPD, parentsᵣ, "continuous")
## node R₂
model₂ = [Model(cpd_r_given_u, :R₂)]
R₂_CPD = FunctionalCPD(:R₂, name.(parentsᵣ), parental_ncategoriesᵣ, [model₂])
R₂_node = FunctionalNode(R₂_CPD, parentsᵣ, "continuous")
## node R₃
model₃ = [Model(cpd_r_given_u, :R₃)]
R₃_CPD = FunctionalCPD(:R₃, name.(parentsᵣ), parental_ncategoriesᵣ, [model₃])
R₃_node = FunctionalNode(R₃_CPD, parentsᵣ, "continuous")
## node R₁
model₄ = [Model(cpd_r_given_u, :R₄)]
R₄_CPD = FunctionalCPD(:R₄, name.(parentsᵣ), parental_ncategoriesᵣ, [model₄])
R₄_node = FunctionalNode(R₄_CPD, parentsᵣ, "continuous")
## node R₁
model₅ = [Model(cpd_r_given_u, :R₅)]
R₅_CPD = FunctionalCPD(:R₅, name.(parentsᵣ), parental_ncategoriesᵣ, [model₅])
R₅_node = FunctionalNode(R₅_CPD, parentsᵣ, "continuous")

## Node E - as 1 single model
function failure_1(r₁, r₂, r₅, r₄)
    return r₁ + r₂ + r₄ + r₅ - 5 * h
end

function cpd_r_given_parents(r₁, r₂, r₃, r₄, r₅, h, v, failure1)
    g1 = failure1(r₁, r₂, r₅, r₄)
    g2 = r₂ + 2 * r₅ + r₄ - 5 * v
    g3 = r₁ + 2 * r₃ + 2 * r₄ + r₅ - 5 * h - 5 * v
    min(g1, g2, g3) ≤ 0 ? output = true : output = false
    return output
end

parental_ncategoriesₑ = Vector{Int}()
parentsₑ = [V_node, H_node, R₁_node, R₂_node]
failuremodel = Model(failure_1, :f1)
outputmodel = Model(cpd_r_given_parents, :E)
modelₑ = [failuremodel, outputmodel]
E_CPD = FunctionalCPD(:E, name.(parentsₑ), parental_ncategoriesₑ, [modelₑ])
E_node = FunctionalNode(E_CPD, parentsₑ, "discrete")

ebn = EnhancedBayesNet([Uᵣ_node, V_node, H_node, R₁_node, R₂_node, E_node])
# groups = markov_envelopes(ebn)
node = :Uᵣ
ch = :R₄
ebn1 = ebn

# node_index = ebn.name_to_index[node]
# child_nodes = children(ebn, node)
# new_dag = ebn.dag
# for child in child_nodes
#     new_dag = _invert_nodes_link(new_dag, ebn.name_to_index[node], ebn.name_to_index[child])
# end
# dag = copy(new_dag)
# barren_index = 3
# new_dag = invert_nodes_link(ebn, pr, ch)

new_dag, new_names = _eliminate_node(ebn, node)

graphplot(
    new_dag,
    method=:tree,
    names=new_names,
    fontsize=9,
    nodeshape=:ellipse,
    markercolor=map(x -> x.type == "discrete" ? "lightgreen" : "orange", ebn.nodes),
    linecolor=:darkgrey,
)