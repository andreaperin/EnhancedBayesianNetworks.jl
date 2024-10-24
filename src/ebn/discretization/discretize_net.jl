function _discretize(ebn::EnhancedBayesianNetwork)
    return EnhancedBayesianNetwork(_discretize!(deepcopy(ebn.nodes)))
end