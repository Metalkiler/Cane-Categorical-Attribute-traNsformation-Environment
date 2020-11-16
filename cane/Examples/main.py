from Examples.functions import *

# with a custom function to apply to data:
if __name__ == "__main__":
    dfNumbers = pd.DataFrame(np.random.randint(0, 100000, size=(100000, 12)), columns=list('ABCDEFGHIJKL'))
    print(cane.scale_data(dfNumbers, n_cores=3, scaleFunc="custom", customfunc=customFunc))
