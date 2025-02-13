using Random

"""
    split_data(n_patients::Int; train_ratio=0.7, val_ratio=0.15)

Restituisce tre vettori di indici per training, validation e test.
"""
function split_data(rng, n_patients::Int; train_ratio=0.3)
    indices = shuffle(rng, 1:n_patients)
    
    n_train = Int(round(train_ratio * n_patients))
    n_val   = Int(round(val_ratio   * n_patients))
    
    train_idx = indices[1:n_train]
    val_idx   = indices[n_train+1:n_train+n_val]
    test_idx  = indices[n_train+n_val+1:end]
    
    return train_idx, val_idx, test_idx
end