using EnhancedBayesianNetworks
using Plots


## Fake node
time = DiscreteRootNode(:t, Dict(:first => 0.5, :second => 0.5), Dict(:first => [Parameter(1, :useless)], :second => [Parameter(1, :useless)]))

## R Nodes Definition as RootNodes
λᵣ = 150
σᵣ = 0.2 * 150
ρᵣ = 0.3
log_λᵣ = log((λᵣ^2) / (sqrt(λᵣ^2 + σᵣ^2)))
log_σᵣ = sqrt(log(1 + σᵣ^2 / λᵣ^2))

R_dist = LogNormal(log_λᵣ, log_σᵣ)

R1 = ContinuousRootNode(:R1, R_dist)
R2 = ContinuousRootNode(:R2, R_dist)
R3 = ContinuousRootNode(:R3, R_dist)
R4 = ContinuousRootNode(:R4, R_dist)
R5 = ContinuousRootNode(:R5, R_dist)


## Correlation Node
# correlationvector1 = [1 ρᵣ ρᵣ ρᵣ ρᵣ]
# correlationvector2 = [ρᵣ 1 ρᵣ ρᵣ ρᵣ]
# correlationvector3 = [ρᵣ ρᵣ 1 ρᵣ ρᵣ]
# correlationvector4 = [ρᵣ ρᵣ ρᵣ 1 ρᵣ]
# correlationvector5 = [ρᵣ ρᵣ ρᵣ ρᵣ 1]
# correlationMatrix = vcat(correlationvector1, correlationvector2, correlationvector3, correlationvector4, correlationvector5)
# c = GaussianCopula(correlationMatrix)
# e = Vector{Tuple{Symbol,AbstractNode}}()
# dist = JointDistribution(get_randomvariable.([R1, R2, R3, R4, R5], [e, e, e, e, e]), c)
# jd_dist = OrderedDict(
#     [:first] => dist,
#     [:second] => dist
# )

## Build model from copula! ask Jasper

# jd_parents = [time, R1, R2, R3, R4, R5]
# jd_node = ContinuousFunctionalNode(:jd, jd_parents, jd_dist)


## External Forces

θ = 0.4 * 50 * √(6) / π
μ = 50 - θ * Base.MathConstants.eulergamma
H = ContinuousRootNode(:H, Gumbel(μ, θ))

θ = (0.2)^2 * 60
α = 60 / θ
V = ContinuousRootNode(:V, Gamma(α, θ))

## Output Node
# Failure Functions
failure_1 = df -> df.R1 .+ df.R2 .+ df.R4 .+ df.R5 - 5 .* df.H
failure_2 = df -> df.R2 .+ 2 .* df.R3 .+ df.R4 - 5 .* df.V
failure_3 = df -> df.R1 .+ 2 .* df.R3 .+ 2 .* df.R4 .+ df.R5 - 5 .* df.H - 5 .* df.V
model1 = Model(failure_1, :f1)
model2 = Model(failure_2, :f2)
model3 = Model(failure_3, :f3)

models = OrderedDict(
    [:first] => [model1, model2, model3],
    [:second] => [model1, model2, model3]
)
performances = OrderedDict(
    [:first] => df -> minimum(hcat(df.f1, df.f2, df.f3), dims=2),
    [:second] => df -> minimum(hcat(df.f1, df.f2, df.f3), dims=2)
)
simulations = OrderedDict(
    [:first] => MonteCarlo(10^5),
    [:second] => MonteCarlo(10^6)
)


E = DiscreteFunctionalNode(:E, [H, V, time, R1, R2, R3, R4, R5], models, performances, simulations)

ebn = EnhancedBayesianNetwork([H, V, R1, R2, R3, R4, R5, time, E])
EnhancedBayesianNetworks.plot(ebn)

rbns = reduce_ebn_markov_envelopes(ebn)

evaluate_rbn(rbns)



### Structural Reliability ProbabilisticGraphicalModel
λᵣ = 150
σᵣ = 0.2 * 150
ρᵣ = 0.3
log_λᵣ = log((λᵣ^2) / (sqrt(λᵣ^2 + σᵣ^2)))
log_σᵣ = sqrt(log(1 + σᵣ^2 / λᵣ^2))
R1 = RandomVariable(LogNormal(log_λᵣ, log_σᵣ), :R1)
R2 = RandomVariable(LogNormal(log_λᵣ, log_σᵣ), :R2)
R3 = RandomVariable(LogNormal(log_λᵣ, log_σᵣ), :R3)
R4 = RandomVariable(LogNormal(log_λᵣ, log_σᵣ), :R4)
R5 = RandomVariable(LogNormal(log_λᵣ, log_σᵣ), :R5)

θ = 0.4 * 50 * √(6) / π
μ = 50 - θ * Base.MathConstants.eulergamma
H = RandomVariable(Gumbel(μ, θ), :H)

θ = (0.2)^2 * 60
α = 60 / θ
V = RandomVariable(Gamma(α, θ), :V)

# correlationvector1 = [1 ρᵣ ρᵣ ρᵣ ρᵣ]
# correlationvector2 = [ρᵣ 1 ρᵣ ρᵣ ρᵣ]
# correlationvector3 = [ρᵣ ρᵣ 1 ρᵣ ρᵣ]
# correlationvector4 = [ρᵣ ρᵣ ρᵣ 1 ρᵣ]
# correlationvector5 = [ρᵣ ρᵣ ρᵣ ρᵣ 1]
# correlationMatrix = vcat(correlationvector1, correlationvector2, correlationvector3, correlationvector4, correlationvector5)
# c = GaussianCopula(correlationMatrix)
# jd = JointDistribution([R1, R2, R3, R4, R5], c)

failure_1 = df -> df.R1 .+ df.R2 .+ df.R4 .+ df.R5 - 5 .* df.H
failure_2 = df -> df.R2 .+ 2 .* df.R3 .+ df.R4 - 5 .* df.V
failure_3 = df -> df.R1 .+ 2 .* df.R3 .+ 2 .* df.R4 .+ df.R5 - 5 .* df.H - 5 .* df.V
model1 = Model(failure_1, :f1)
model2 = Model(failure_2, :f2)
model3 = Model(failure_3, :f3)
perf = Model(df -> minimum([df.f1, df.f2, df.f3]), :output)

model = [model1, model2, model3, perf]
inputs = [H, V, R1, R2, R3, R4, R5]

res = probability_of_failure(model, df -> minimum(hcat(df.f1, df.f2, df.f3), dims=2), inputs, MonteCarlo(10^6))[1]
