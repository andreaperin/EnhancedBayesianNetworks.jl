using EnhancedBayesianNetworks
name_a = :A
cpt_a = DataFrame(name_a => [:a1, :a2], :Prob => [0.17, 0.8])
node = DiscreteNode(name_a, cpt_a)

name_b = :B
cpt_b = DataFrame(name_a => [:a1, :a1, :a1, :a2, :a2, :a2], name_b => [:b1, :b2, :b3, :b1, :b2, :b3], :Prob => [0.1, 0.8, 0.1, 0.1, 0.76, 0.1])
B = DiscreteNode(name_b, cpt_b, Dict(:b1 => [Parameter(1, :B)], :b4 => [Parameter(1, :B)], :b3 => [Parameter(1, :B)]))














name_c = :C
cpt_c = DataFrame(name_c => [:c1, :c2], :Prob => [[0.2, 0.4], [0.4, 0.41]])
C = DiscreteNode(name_c, cpt_c)


cpt = DataFrame(name_c => [:yes, :no, :maybe], :Prob => [[0.4, 0.5], [0.2, 0.4], [0.2, 0.3]])
C = DiscreteNode(name_c, cpt, Dict(:c1 => [Parameter(1, :O)], :c2 => [Parameter(2, :O)]))


cpt_E = DataFrame(:A => [:a1, :a1, :a2, :a2], :E => [:e1, :e2, :e1, :e2], :Prob => [[0.9, 0.3], [0.1, 0.5], [0.2, 0.5], [0.1, 0.8]])

E = DiscreteNode(:E, cpt_E)

EnhancedBayesianNetworks._verify_probabilities(E)