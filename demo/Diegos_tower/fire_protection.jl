using EnhancedBayesianNetworks
using Plots

tampering = DiscreteNode(:Tampering, DataFrame(:Tampering => [:NoT, :YesT], :Prob => [[0.98999, 0.99111], [0.00889, 0.01001]]))
fire = DiscreteNode(:Fire, DataFrame(:Fire => [:NoF, :YesF], :Prob => [[0.958978, 0.959989], [0.00011, 0.041002]]))

alarm_df = DataFrame(:Tampering => [:NoT, :NoT, :NoT, :NoT, :YesT, :YesT, :YesT, :YesT], :Fire => [:NoF, :NoF, :YesF, :YesF, :NoF, :NoF, :YesF, :YesF], :Alarm => [:NoA, :YesA, :NoA, :YesA, :NoA, :YesA, :NoA, :YesA], :Prob => [[0.999800, 0.999997], [0.000003, 0.000200], [0.010000, 0.012658], [0.987342, 0.990000], [0.100000, 0.119999], [0.880001, 0.900000], [0.400000, 0.435894], [0.564106, 0.600000]])
alarm = DiscreteNode(:Alarm, alarm_df)

smoke_df = DataFrame(:Fire => [:NoF, :NoF, :YesF, :YesF], :Smoke => [:NoS, :YesS, :NoS, :YesS], :Prob => [[0.897531, 0.915557], [0.010000, 0.102469], [0.090000, 0.110000], [0.890000, 0.910000]])
smoke = DiscreteNode(:Smoke, smoke_df)

leaving_df = DataFrame(:Alarm => [:NoA, :NoA, :YesA, :YesA], :Leaving => [:NoL, :YesL, :NoL, :YesL], :Prob => [[0.585577, 0.599999], [0.400001, 0.414423], [0.100000, 0.129999], [0.870001, 0.900000]])
leaving = DiscreteNode(:Leaving, leaving_df)

report_df = DataFrame(:Leaving => [:NoL, :NoL, :YesL, :YesL], :Report => [:NoR, :YesR, :NoR, :YesR], :Prob => [[0.809988, 0.828899], [0.171101, 0.190012], [0.240011, 0.250000], [0.750000, 0.759989]])
report = DiscreteNode(:Report, report_df)

nodes = [tampering, fire, alarm, smoke, leaving, report]

cn = CredalNetwork(nodes)
add_child!(cn, :Tampering, :Alarm)
add_child!(cn, :Fire, :Alarm)
add_child!(cn, :Fire, :Smoke)
add_child!(cn, :Alarm, :Leaving)
add_child!(cn, :Leaving, :Report)
order!(cn)

plt1 = gplot(cn; nodesizefactor=0.17)
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