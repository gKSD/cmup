def func(f):
    f.append(1)
    f.append(2)


f = []
func(f)
func(f)
print f


if 1:
    test = 5
if 1:
    test = test + 5
print test
