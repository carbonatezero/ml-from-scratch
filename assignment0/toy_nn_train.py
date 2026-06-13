x = [2, 1, 3]
y = 2
lr = 0.001

w1 = [[0.5, -1, 0.2], [1, 0.1, -0.5]]
b1 = [0.1, -0.2]
w2 = [1.5, -1]
b2 = 0.3


def dot(a, b):
    s = 0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


for step in range(20):
    h = [dot(x, w1[0]) + b1[0], dot(x, w1[1]) + b1[1]]
    r = [max(0, h[0]), max(0, h[1])]
    score = dot(r, w2) + b2
    loss = (score - y) ** 2

    dscore = 2 * (score - y)
    dw2 = [dscore * r[0], dscore * r[1]]
    db2 = dscore
    dr = [dscore * w2[0], dscore * w2[1]]
    dh = [dr[0] if h[0] > 0 else 0, dr[1] if h[1] > 0 else 0]

    for neuron in range(2):
        for i in range(3):
            w1[neuron][i] -= lr * dh[neuron] * x[i]
        b1[neuron] -= lr * dh[neuron]
        w2[neuron] -= lr * dw2[neuron]
    b2 -= lr * db2

    print("step", step, "score", round(score, 3), "loss", round(loss, 3))
