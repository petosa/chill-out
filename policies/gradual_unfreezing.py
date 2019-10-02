def get_gradual_unfreezing_policy():
    return [[False]*(8-i) + [True]*i for i in range(1,9)]