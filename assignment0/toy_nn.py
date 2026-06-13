x = [2, 1, 3]

w1 = [[0.5, -1, 0.2], [1, 0.1, -0.5]]
b1 = [0.1, -0.2]

w2 = [1.5, -1]
b2 = 0.3


def dot(a, b):
    s = 0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s


h = []
for neuron in range(2):
    h.append(dot(x, w1[neuron]) + b1[neuron])

r = [max(0, v) for v in h]
score = dot(r, w2) + b2

print("input:", x)
print("hidden:", h)
print("relu:", r)
print("score:", round(score, 2))

