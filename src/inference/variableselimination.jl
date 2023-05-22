function infer(inf::InferenceState)
    rbn = inf.rbn
    nodes = rbn.nodes
    query = inf.query
    evidence = inf.evidence
    hidden = setdiff([i.name for i in nodes], vcat(query, [j[2].name for j in evidence]))
end
