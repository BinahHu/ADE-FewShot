import random
random.seed(73)


def enlarge(x1, x2, bound, ratio):
    l = x2 - x1
    p = int(l * ratio - l)

    d1 = random.randint(0, min(x1, p))
    d2 = p - d1
    if d1 < 0:
        print("d1 < 0 error!")
    if d1 < 0:
        print("d2 < 0 error!")

    x1 -= d1
    x2 += d2

    return [x1, x2]


def add_context(args, box, shape):
    Ratio = args.ratio
    l, r, u, d = box
    H = shape[0]
    W = shape[1]

    boxArea = (r - l) * (d - u)
    xthd = W / (r - l)
    ythd = H / (d - u)
    if H * W / boxArea < Ratio:
        Ratio = H * W / boxArea

    if random.randint(0, 1) == 0:
        xratio = random.uniform(1.0, min(Ratio, xthd))
        yratio = Ratio / xratio
    else:
        yratio = random.uniform(1.0, min(Ratio, ythd))
        xratio = Ratio / yratio

    l, r = enlarge(l, r, W, xratio)
    u, d = enlarge(u, d, H, yratio)
    box = [l, r, u, d]

    return box
