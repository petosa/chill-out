def get_trainable_layer_count(model):
    counter = 0
    for _, (_, w) in enumerate(model.named_children()):
        for _, (_, w1) in enumerate(w.named_children()):
            #These loops go over all nested children in the model architecture (including those without grad updates)
            count = True
            for _, (_, _) in enumerate(w1.named_parameters()):
                #This loop filters out any children that aren't trainable, but we only want to count the layer if theres at least 1 trainable node within it.
                if count == True:
                    counter += 1
                    count = False
    return counter
