from numpy.lib.function_base import average


def average_list(li):
    return sum(li)/len(li) if len(li)>0 else 0

def get_curve_state(last, penultimate):
    if last>penultimate:
        state = "increasing"
    elif last<penultimate:
        state = "decreasing"
    elif int(round(last, 0))==int(round(penultimate, 0)):
        state = "stable"
    else:
        state = "Error"
    return state

def is_decreasing(li):
    return all(earlier>=later for earlier, later in zip(li, li[1:]))

def is_good_fitted(train, val):
    # 1. Close to each other, with validation loss being slightly greater than the training loss
    gaps = []

    for tr, va in zip(train, val):
        gaps.append(va-tr)
    # percent of neg items
    total_neg_values = 0
    for gap in gaps:
        if gap<0:
            total_neg_values += 1
    percentage_neg_values = total_neg_values*100/len(gaps)
    avg_gap = average_list(gaps)
    # pourcentage de valeurs négatives < 10% et le gap moyen ne doit pas dépasser 0.15
    c11 = percentage_neg_values < float(10)
    c12 = avg_gap < 0.15

    c1 = True if c11 and c12 else False

    # 2. Initially decreasing training and validation loss and a pretty flat training and validation loss after some points till the end
    # decreasing at the beginning
    train_thirty_first_percent = train[:int(round(30*len(train)/100, 0))+1]
    validation_thirty_first_percent = val[:int(round(30*len(val)/100, 0))+1]
    
    # flat
    train_fifty_last_percent = train[int(50*len(train)/100):]
    validation_fifty_last_percent = val[int(50*len(val)/100):]
    delta_train = max(train_fifty_last_percent)-min(train_fifty_last_percent)
    delta_val = max(validation_fifty_last_percent)-min(validation_fifty_last_percent)
    c221 = delta_train > 0 and delta_train < 0.1
    c222 = delta_val > 0 and delta_val < 0.1

    c21 = is_decreasing(train_thirty_first_percent) and is_decreasing(validation_thirty_first_percent)
    c22 = True if c221 and c222 else False

    if c21 and c22:
        c2 = True
    elif not c21 and c22:
        c2 = True
    elif c21 and not c22:
        c2 = True
    else:
        c2 = False

    if c1 and c2:
        to_return = "yes"
    elif not c1 and c2:
        to_return = "partially"
    elif c1 and not c2:
        to_return = "partially"
    else:
        to_return = "no"
    
    return to_return

def is_under_fitted(train, val):
    # 1. Increasing training loss
    train_first_tier = train[:int(round(30*len(train)/100, 0))]
    train_second_tier = train[int(round(30*len(train)/100)):int(round(60*len(train)/100))+1]
    train_last_tier = train[int(round(60*len(train)/100, 0))+1:]
    c11 = is_decreasing(train_first_tier)
    c12 = is_decreasing(train_second_tier)
    c13 = is_decreasing(train_last_tier)
    if not c11 and not c12 and not c13:
        c1 = True
    elif c11 and not c12 and not c13:
        c1 = True
    elif not c11 and c12 and not c13:
        c1 = True
    elif not c11 and not c12 and c13:
        c1 = True
    else:
        c1 = False

    print("UNDER FITTING:\n1. Increasing training loss: "+str(c1))

    # 2. Training and validation loss are close to each other at the end
    val_last_tier = val[int(round(60*len(val)/100, 0))+1:]
    gaps = []
    for tr, va in zip(train_last_tier, val_last_tier):
        gaps.append(abs(tr-va))
    avg_gap = average_list(gaps)
    c2 = avg_gap < 0.15

    print("UNDER FITTING\n2. Training and validation loss are close to each other at the end: "+str(c2))

    # 3. Sudden dip in the training and validation at the end
    val_first_tier = val[:int(round(30*len(val)/100, 0))]
    val_first_tier_avg = average_list(val_first_tier)
    train_first_tier_avg = average_list(train_first_tier)
    first_tier_avg = (val_first_tier_avg+train_first_tier_avg)/2

    train_last_tier_avg = average_list(train_last_tier)
    val_last_tier_avg = average_list(val_last_tier)
    last_tier_avg = (train_last_tier_avg+val_last_tier_avg)/2

    # is decreasing at the end ?
    last_train, penultimate_train = train[-1], train[-2]
    last_val, penultimate_val = val[-1], val[-2]


    c31 = first_tier_avg > last_tier_avg
    c32 = (last_train<penultimate_train) and (last_val<penultimate_val)

    c3 = True if c31 and c32 else False

    print("UNDER FITTING\n3. Sudden dip in the training and validation loss at the end: "+str(c3))

    if c1 and c2 and c3:
        to_return = "yes"
    elif not c1 and c2 and c3:
        to_return = "partially"
    elif c1 and not c2 and c3:
        to_return = "partially"
    elif c1 and c2 and not c3:
        to_return = "partially"
    else:
        to_return = "no"

    return to_return

def is_over_fitted(train, val):
    # 1. Training and validation loss are far away from each other
    gaps = []
    for tr, va in zip(train, val):
        gaps.append(abs(tr-va))
    c1 = average_list(gaps)>0.6

    print("OVER FITTING:\nTraining and validation loss are far away from each other: "+str(c1))
    
    # 2. Gradually decreasing validation loss (without flattening)
    val_first_tier = val[:int(round(30*len(val)/100, 0))]
    val_second_tier = val[int(round(30*len(val)/100)):int(round(60*len(val)/100))+1]
    val_last_tier = val[int(round(60*len(val)/100, 0))+1:]
    
    c21 = is_decreasing(val_first_tier)
    c22 = is_decreasing(val_second_tier)
    c23 = is_decreasing(val_last_tier)

    if c21 and c22 and c23:
        c2 = True
    elif not c21 and c22 and c23:
        c2 = True
    elif c21 and not c22 and c23:
        c2 = True
    elif c21 and c22 and not c23:
        c2 = True
    else:
        c2 = False

    print("OVER FITTING:\nGradually decreasing validation loss (without flatenning): "+str(c2))

    # 3. Very low training loss that's very slightly increasing
    delta_train = max(train)-min(train)
    train_last_tier = train[int(round(60*len(train)/100, 0))+1:]
    
    c31 = delta_train < 0.05
    c32 = is_decreasing(train_last_tier)

    if c31 and c32:
        c3 = True
    elif not c31 and c32:
        c3 = True
    elif c31 and not c32:
        c3 = True
    else:
        c3 = False

    print("OVER FITTING:\nVery low training loss that's very slightly increasing "+str(c3))

    # final condition
    if c1 and c2 and c3:
        to_return = "yes"
    elif not c1 and c2 and c3:
        to_return = "partially"
    elif c1 and not c2 and c3:
        to_return = "partially"
    elif c1 and c2 and not c3:
        to_return = "partially"
    else:
        to_return = "no"
    
    return to_return

def percentage_title(input, output):
    titles = ['Mrs.', 'Ms.', 'Miss', 'Dr.', 'Mr.']
    occ_title_in, occ_title_out = 0, 0
    for inp in input:
        inp = inp.split(" ")
        for title in titles:
            if title in inp:
                occ_title_in += 1
    for out in output:
        out = out.split(" ")
        for title in titles:
            if title in out:
                occ_title_out += 1
    return occ_title_in*100/len(input), occ_title_out*100/len(output)