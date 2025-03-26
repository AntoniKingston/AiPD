import numpy as np
def timestwo(x):
    x = x * 2
    return x
def main():
    x = np.array([3,5,6])
    x = timestwo(x)
    print(x)

if __name__ == '__main__':
    main()