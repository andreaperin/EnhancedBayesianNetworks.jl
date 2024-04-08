function transfer_continuous(ebn::EnhancedBayesianNetwork)
    transferred_ebn = EnhancedBayesianNetwork(_transfer_continuous!(deepcopy(ebn.nodes)))
    continuous_functional = filter(x -> isa(x, ContinuousFunctionalNode), transferred_ebn.nodes)
    for n in continuous_functional
        continuous_parents = filter(x -> isa(x, ContinuousNode), n.parents)
        if !isempty(filter(x -> !isa(x.distribution, UnivariateDistribution), continuous_parents))
            error("node $(n.name) is a continuousfunctionalnode with at least one parent with Interval or p-boxes in its distributions. No method for extracting failure probability p-box have been implemented yet")
        end
    end
    return transferred_ebn
end
