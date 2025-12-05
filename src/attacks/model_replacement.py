def model_replacement_attack(global_state, malicious_state, scaling_factor: float):
    replaced = {}
    for k in global_state.keys():
        g = global_state[k].detach()
        m = malicious_state[k].detach()
        replaced[k] = g + scaling_factor * (m - g)
    return replaced
