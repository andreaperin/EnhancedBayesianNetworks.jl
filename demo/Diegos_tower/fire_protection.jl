using EnhancedBayesianNetworks
using Plots

tampering = DiscreteNode(:Tampering, DataFrame(:Tampering => [:NoT, :YesT], :Prob => [0.98, 0.02]))
fire = DiscreteNode(:Fire, DataFrame(:Fire => [:NoF, :YesF], :Prob => [[0.98, 0.99], [0.01, 0.02]]))

alarm_df = DataFrame(:Tampering => [:NoT, :NoT, :NoT, :NoT, :YesT, :YesT, :YesT, :YesT], :Fire => [:NoF, :NoF, :YesF, :YesF, :NoF, :NoF, :YesF, :YesF], :Alarm => [:NoA, :YesA, :NoA, :YesA, :NoA, :YesA, :NoA, :YesA], :Prob => [[0.9998, 0.9999], [0.0001, 0.0002], [0.01, 0.015], [0.985, 0.99], [0.1, 0.15], [0.85, 0.90], [0.4, 0.6], [0.4, 0.6]])
alarm = DiscreteNode(:Alarm, alarm_df)

smoke_df = DataFrame(:Fire => [:NoF, :NoF, :YesF, :YesF], :Smoke => [:NoS, :YesS, :NoS, :YesS], :Prob => [[0.9, 0.99], [0.01, 0.1], [0.09, 0.13], [0.87, 0.91]])
smoke = DiscreteNode(:Smoke, smoke_df)

leaving_df = DataFrame(:Alarm => [:NoA, :NoA, :YesA, :YesA], :Leaving => [:NoL, :YesL, :NoL, :YesL], :Prob => [[0.58, 0.999], [0.10, 0.12], [0.001, 0.420], [0.88, 0.9]])
leaving = DiscreteNode(:Leaving, leaving_df)

report_df = DataFrame(:Leaving => [:NoL, :NoL, :YesL, :YesL], :Report => [:NoR, :YesR, :NoR, :YesR], :Prob => [[0.80, 0.99], [0.01, 0.2], [0.24, 0.75], [0.25, 0.76]])

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

# saveplot(plt1, "/Users/andreaperin_macos/Documents/PhD/3_Academic/Papers_Presentations/Presentations/BGE-Slide/imgs/fig_cn_tolo.svg")

evidence = Evidence()
query = [:Smoke]
ϕ1 = infer(cn, query, evidence)

evidence = Evidence()
query = [:Report]
ϕ2 = infer(cn, query, evidence)

evidence = Evidence()
query = [:Alarm]
ϕ3 = infer(cn, query, evidence)

evidence = Evidence(
    :Fire => :YesF
)
query = [:Leaving]
ϕ4 = infer(cn, query, evidence)

evidence = Evidence(
    :Fire => :YesF
)
query = [:Report]
ϕ5 = infer(cn, query, evidence)

evidence = Evidence(
    :Leaving => :YesL
)
query = [:Fire]
ϕ6 = infer(cn, query, evidence)

smoke_pot = ϕ1.potential
reference_smoke = [[0.8838 0.9814], [0.0186, 0.1162]]

report_pot = ϕ2.potential
reference_report = [[0.5547, 0.9719], [0.0281, 0.4453]]

alarm_pot = ϕ3.potential
reference_alarm = [[0.9625 0.9733], [0.0281, 0.0375]]

leaving_pot = ϕ4.potential
reference_leaving = [[0.1085 0.1435], [0.8565, 0.8915]]


### PLOTIING

mn = [
    reference_smoke[1][2],
    reference_smoke[2][2],
    # reference_report[1][2],
    # reference_report[2][2],
    reference_alarm[1][2],
    reference_alarm[2][2],
    reference_leaving[1][2],
    reference_leaving[2][2],
    smoke_pot[1][2],
    smoke_pot[2][2],
    # report_pot[1][2],
    # report_pot[2][2],
    alarm_pot[1][2],
    alarm_pot[2][2],
    leaving_pot[1][2],
    leaving_pot[2][2],]

nm = [
    reference_smoke[1][1],
    reference_smoke[2][1],
    # reference_report[1][1],
    # reference_report[2][1],
    reference_alarm[1][1],
    reference_alarm[2][1],
    reference_leaving[1][1],
    reference_leaving[2][1],
    smoke_pot[1][1],
    smoke_pot[2][1],
    # report_pot[1][1],
    # report_pot[2][1],
    alarm_pot[1][1],
    alarm_pot[2][1],
    leaving_pot[1][1],
    leaving_pot[2][1]
]
sx = repeat(["Ref", "ebn.jl"], inner=6)
nam = repeat(["S=N", "S=Y", "A=N", "A=Y", "L|F=N", "L|F=Y"], outer=2)

# using StatsPlots
fig1 = groupedbar(nam, mn, group=sx, ylabel="Marginal Probabilities", fillto=nm,
    title="Inference In CN - Comparison with reference solutions")

# savefig(fig1, "/Users/andreaperin_macos/Documents/PhD/3_Academic/Papers_Presentations/Presentations/BGE-Slide/imgs/barplot.png")