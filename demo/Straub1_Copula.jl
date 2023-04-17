include("../src/bn.jl")

## R Nodes Definition as RootNodes
λᵣ = 150
σᵣ = 0.2
ρᵣ = 0.3
log_λᵣ = log(λᵣ^2 / (sqrt(λᵣ^2 + σᵣ^2)))
log_σᵣ = sqrt(log(1 + σᵣ^2 / λᵣ^2))

R_dist = LogNormal(log_λᵣ, log_σᵣ)

R1 = RandomVariable(R_dist, :R1)
R1_cpd = RootCPD(:R1, [R_dist])
R1_node = RootNode(R1_cpd)

R2 = RandomVariable(R_dist, :R2)
R2_cpd = RootCPD(:R2, [R_dist])
R2_node = RootNode(R2_cpd)

R3 = RandomVariable(R_dist, :R3)
R3_cpd = RootCPD(:R3, [R_dist])
R3_node = RootNode(R3_cpd)

R4 = RandomVariable(R_dist, :R4)
R4_cpd = RootCPD(:R4, [R_dist])
R4_node = RootNode(R4_cpd)

R5 = RandomVariable(R_dist, :R5)
R5_cpd = RootCPD(:R5, [R_dist])
R5_node = RootNode(R5_cpd)


## Correlation Node
correlationvector1 = [1 ρᵣ ρᵣ ρᵣ ρᵣ]
correlationvector2 = [ρᵣ 1 ρᵣ ρᵣ ρᵣ]
correlationvector3 = [ρᵣ ρᵣ 1 ρᵣ ρᵣ]
correlationvector4 = [ρᵣ ρᵣ ρᵣ 1 ρᵣ]
correlationvector5 = [ρᵣ ρᵣ ρᵣ ρᵣ 1]
correlationMatrix = vcat(correlationvector1, correlationvector2, correlationvector3, correlationvector4, correlationvector5)
c = GaussianCopula(correlationMatrix)
jd_dist = JointDistribution([R1, R2, R3, R4, R5], c)
jd_parents = [R1_node, R2_node, R3_node, R4_node, R5_node]
jd_CDP = StdCPD(:jd, name.(jd_parents), jd_dist)
jd_node = StdNode(jd_CDP, jd_parents)

## Output Node
# Failure Functions
failure_1 = df -> df.R1 .+ df.R2 .+ df.R4 .+ df.R5 - 5 .* df.H
failure_2 = df -> df.R2 .+ 2 .* df.R3 .+ df.R4 - 5 .* df.V
failure_3 = df -> df.R1 .+ 2 .* df.R3 .+ 2 .* df.R4 .+ df.R5 - 5 .* df.H - 5 .* df.V
performance = df -> minimum([df.f1, df.f2, df.f3])

model1 = Model(failure_1, :f1)
model2 = Model(failure_2, :f2)
model3 = Model(failure_3, :f3)
perf = Model(performance, :output)

models = [model1, model2, model3, perf]

model = ModelWithName(:failuremodes, models)
output_parents = [jd_node]
output_CPD = FunctionalCPD(:out, name.(output_parents), model)

