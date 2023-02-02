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
prob_dict_jd = [ProbabilityDictionary(((:emission => 1), f1)),
    ProbabilityDictionary(((:emission => 2), f2))]

CPD_jd = FunctionalCPD(:jd, parents_jd, parental_ncategories, prob_dict_jd)

### TODO Build functional nodes (mutable struct) from FunctionalCPD checking: 1) Discrete Parents (same in evidence) 2) Discrete Parents lengh and parental_ncategories



c_distribution = Normal(0, 1)
c_parents = [E_node, ρ_node]
CPD_c = CategoricalCPD(:c, name.(c_parents), [1], [c_distribution])
c_node = StdNode(CPD_c, c_parents)


c = GaussianCopula([1 0.8; 0.8 1])
jd_distribution = JointDistribution([E_distribution ρ_distribution], c)



l = Parameter(1.8, :l) # length
b = Parameter(0.12, :b) # width

h = RandomVariable(Normal(0.24, 0.01), :h) # height

μ = log(10e9^2 / sqrt(1.6e9^2 + 10e9^2))
σ = sqrt(log(1.6e9^2 / 10e9^2 + 1))
E = RandomVariable(LogNormal(μ, σ), :E) # young's modulus

μ = log(5000^2 / sqrt(400^2 + 5000^2))
σ = sqrt(log(400^2 / 5000^2 + 1))
P = RandomVariable(LogNormal(μ, σ), :P) # tip load

μ = log(600^2 / sqrt(140^2 + 600^2))
σ = sqrt(log(140^2 / 600^2 + 1))
ρ = RandomVariable(LogNormal(μ, σ), :ρ) # density

c = GaussianCopula([1 0.8; 0.8 1])
jd = JointDistribution([E ρ], c)

inputs = [l, b, h, P, jd]