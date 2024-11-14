using EnhancedBayesianNetworks
name_a = :A
cpt_a = DataFrame(name_a => [:a1, :a2], :Prob => [0.2, 0.8])
A = DiscreteNode(name_a, cpt_a)

name_b = :B
cpt_b = DataFrame(name_b => [:b1, :b2, :b3], :Prob => [0.1, 0.8, 0.1])
B = DiscreteNode(name_b, cpt_b)

name_c = :C
cpt_c = DataFrame(name_c => [:c1, :c2], :Prob => [0.3, 0.7])
C = DiscreteNode(name_c, cpt_c)


cpt = DataFrame(name_c => [:yes, :no, :maybe], :Prob => [[0.4, 0.5], [0.2, 0.4], [0.2, 0.3]])
C = DiscreteNode(name_c, cpt, Dict(:c1 => [Parameter(1, :O)], :c2 => [Parameter(2, :O)]))
