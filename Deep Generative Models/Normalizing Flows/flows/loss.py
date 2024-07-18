def likelihood(X_train, model, device):
    ##########################################################
    # YOUR CODE HERE
    X_train = X_train.to(device)
    log_probs = model.log_prob(X_train)
    loss = - log_probs.mean()
    ##########################################################

    return loss
