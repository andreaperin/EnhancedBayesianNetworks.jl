function transfer_continuous(ebn::EnhancedBayesianNetwork)
    return EnhancedBayesianNetwork(_transfer_continuous!(deepcopy(ebn.nodes)))
end
