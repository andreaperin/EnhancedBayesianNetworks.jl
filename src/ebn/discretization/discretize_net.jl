function discretize(ebn::EnhancedBayesianNetwork)
    return EnhancedBayesianNetwork(_discretize(ebn.nodes))
end