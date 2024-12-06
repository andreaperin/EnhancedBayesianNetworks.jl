function _are_consistent(a::Evidence, b::Evidence)
    for key in keys(a)
        haskey(b, key) && b[key] != a[key] && error("not consistent evidences: $a; $b")
    end
    return true
end

function _verify_evidence(a::Evidence, bn::BayesianNetwork)
    any(keys(a) .∉ [[i.name for i in bn.nodes]]) && error("all nodes in the $a have to be in the network")
    for (n, s) in a
        s ∉ _states(bn.nodes[bn.topology_dict[n]]) && error("node states in $a must be coherent with the one defined in the network")
    end
end