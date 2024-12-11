using EnhancedBayesianNetworks
using Plots

tampering = DiscreteRootNode(:Tampering, Dict(:NoT => [0.98999, 0.99111], :YesT => [0.00889, 0.01001]))
fire = DiscreteRootNode(:Fire, Dict(:NoF => [0.958978, 0.959989], :YesF => [0.00011, 0.041002]))

alarm_states = Dict(
    [:NoT, :NoF] => Dict(:NoA => [0.999800, 0.999997], :YesA => [0.000003, 0.000200]),
    [:NoT, :YesF] => Dict(:NoA => [0.010000, 0.012658], :YesA => [0.987342, 0.990000]),
    [:YesT, :NoF] => Dict(:NoA => [0.100000, 0.119999], :YesA => [0.880001, 0.900000]),
    [:YesT, :YesF] => Dict(:NoA => [0.400000, 0.435894], :YesA => [0.564106, 0.600000])
)
alarm = DiscreteChildNode(:Alarm, [tampering, fire], alarm_states)

smoke_state = Dict(
    [:NoF] => Dict(:NoS => [0.897531, 0.915557], :YesS => [0.010000, 0.102469]),
    [:YesF] => Dict(:NoS => [0.090000, 0.110000], :YesS => [0.890000, 0.910000])
)
smoke = DiscreteChildNode(:Smoke, [fire], smoke_state)

leaving_state = Dict(
    [:NoA] => Dict(:NoL => [0.585577, 0.599999], :YesL => [0.400001, 0.414423]),
    [:YesA] => Dict(:NoL => [0.100000, 0.129999], :YesL => [0.870001, 0.900000])
)
leaving = DiscreteChildNode(:Leaving, [alarm], leaving_state)

report_state = Dict(
    [:NoL] => Dict(:NoR => [0.809988, 0.828899], :YesR => [0.171101, 0.190012]),
    [:YesL] => Dict(:NoR => [0.240011, 0.250000], :YesR => [0.750000, 0.759989])
)
report = DiscreteChildNode(:Report, [leaving], report_state)

nodes = [tampering, fire, alarm, smoke, leaving, report]

cn = CredalNetwork(nodes)

plt1 = EnhancedBayesianNetworks.plot(cn, :stress, 0.1, 8)
savefig(plt1, "/Users/andreaperin_macos/Documents/PhD/3_Academic/Papers_Presentations/Papers/0_IEEE_ebn/imgs/fig_cn_tolo.png")


evidence = Evidence()
query = [:Smoke]
ϕ1 = infer(cn, query, evidence)
res1 = ϕ1.potential[ϕ1.states_mapping[:Smoke][:YesS]]


evidence = Evidence()
query = [:Report]
ϕ2 = infer(cn, query, evidence)
res2 = ϕ2.potential[ϕ2.states_mapping[:Report][:YesR]]


evidence = Evidence()
query = [:Alarm]
ϕ3 = infer(cn, query, evidence)
res3 = ϕ3.potential[ϕ3.states_mapping[:Alarm][:YesA]]

evidence = Evidence()
query = [:Leaving]
ϕ4 = infer(cn, query, evidence)
res4 = ϕ4.potential[ϕ4.states_mapping[:Leaving][:YesL]]


evidence = Evidence(
    :Fire => :YesF
)
query = [:Leaving]
ϕ5 = infer(cn, query, evidence)
res5 = ϕ5.potential[ϕ5.states_mapping[:Leaving][:YesL]]


evidence = Evidence(
    :Fire => :YesF
)
query = [:Report]
ϕ6 = infer(cn, query, evidence)
res6 = ϕ6.potential[ϕ6.states_mapping[:Report][:YesR]]


evidence = Evidence(
    :Alarm => :YesA
)
query = [:Report]
ϕ7 = infer(cn, query, evidence)
res7 = ϕ7.potential[ϕ7.states_mapping[:Report][:YesR]]



evidence = Evidence(
    :Leaving => :YesL
)
query = [:Fire]
ϕ8 = infer(cn, query, evidence)
res8 = ϕ8.potential[ϕ8.states_mapping[:Fire][:YesF]]

res_no_evi = [res1, res2, res3, res4]
ending_points = map(i -> i[2], res_no_evi)
starting_points = map(i -> i[1], res_no_evi)
ending_points_ref = [0.025, 0.040, 0.030, 0.040]
starting_points_ref = [0.130, 0.444, 0.040, 0.440]

st = hcat(starting_points, starting_points_ref)
en = hcat(ending_points, ending_points_ref)
plot(bar(ending_points, bar_width=1.2, alpha=0.8, color=:red, fillto=starting_points, label="Ebn.jl"), bar([0.130, 0.444, 0.040, 0.440], bar_width=0.8, alpha=1, color=:green, fillto=[0.025, 0.040, 0.030, 0.040], label="Ref"))



res_evi = [res5, res6, res7, res8]
ending_points_evi = map(i -> i[2], res_evi)
starting_points_evi = map(i -> i[1], res_evi)
ending_points_ref_evi = [0.900, 0.700, 0.710, 0.510]
starting_points_ref_evi = [0.850, 0.648, 0.660, 0.030]

st_evi = hcat(starting_points_evi, starting_points_ref_evi)
en_evi = hcat(ending_points_evi, ending_points_ref_evi)
using StatsPlot

prob_names = ["P(YesL|YesF)", "P(YesR|YesF)", "P(YesR|YesA)", "P(YesF|YesL)"]
ctg = repeat(["Ebn.jl", "Ref"], inner=length(prob_names))
nam = repeat(prob_names, outer=2)

plt2 = groupedbar(nam, st_evi, bar_position=:dodge, group=ctg, bar_width=0.7, fillto=en_evi, ylims=(0, 1), grid=true)

savefig(plt2, "/Users/andreaperin_macos/Documents/PhD/3_Academic/Papers_Presentations/Papers/0_IEEE_ebn/imgs/fig_cn_inference.png")