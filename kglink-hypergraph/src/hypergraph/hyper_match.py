def hyper_match(e1, e2):

    s1 = set(e1)
    s2 = set(e2)

    return len(s1 & s2) / len(s1 | s2)