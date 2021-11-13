import numpy as np

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    #Your code here
    if x <= y:
        return x*y
    else:
        return x/y

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y
    """
    #Your code here
    vfunc = np.vectorize(scalar_function)
    return vfunc(x,y)

def get_sum_metrics(predictions, metrics=None):
    if metrics is None:
        metrics = []
    for i in range(0, 3):
        f = lambda x, i=i: x + i
        metrics.append(f)
    sum_metrics = 0
    for metric in metrics:
        sum_metrics += metric(predictions)
    return sum_metrics

if __name__ == "__main__":
    z = vector_function(np.random.random(5),np.random.random(5))
    print(get_sum_metrics(0))
    print(get_sum_metrics(1))
    print(get_sum_metrics(2))
    print(get_sum_metrics(3, [lambda x: x]))
    print(get_sum_metrics(0))
    print(get_sum_metrics(1))
    print(get_sum_metrics(2))
