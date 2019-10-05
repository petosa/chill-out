def get_gradual_unfreezing_policy(n_layers=8):
    # policy = [
    #     [False]*7 + [True],
    #     [False]*6 + [True]*2,
    #     [False]*5 + [True]*3,
    #     [False]*4 + [True]*4,
    #     [False]*3 + [True]*5,
    #     [False]*2 + [True]*6,
    #     [False]*1 + [True]*7,
    #     [True]*8,
    # ]
    return [[False]*(n_layers-i) + [True]*i for i in range(1,n_layers+1)]