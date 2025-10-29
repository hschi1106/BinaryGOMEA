import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv('plot_result.csv')
    ell = df['ell']
    min_pop = df['min_pop']
    avg_nfe = df['avg_nfe']
    
    # plot n to ell
    plt.figure(figsize=(10, 6))
    plt.scatter(ell, min_pop, marker='o', color='black', s=10)
    plt.title('n vs ell')
    plt.xlabel('ell')
    plt.ylabel('n')
    plt.grid(True)
    plt.savefig('n2ell.png')
    
    # plot logn to logell
    plt.figure(figsize=(10, 6))
    plt.scatter(ell, min_pop, marker='o', color='black', s=10)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('log n vs log ell')
    plt.xlabel('log ell')
    plt.ylabel('log n')
    plt.grid(True, which="both", ls="--")
    plt.savefig('logn2logell.png')
    
    # plot avg_nfe to ell
    plt.figure(figsize=(10, 6))
    plt.scatter(ell, avg_nfe, marker='o', color='black', s=10)
    plt.title('average nfe vs ell')
    plt.xlabel('ell')
    plt.ylabel('average nfe')
    plt.grid(True)
    plt.savefig('nfe2ell.png')
    
    # plot logavg_nfe to logell
    plt.figure(figsize=(10, 6))
    plt.scatter(ell, avg_nfe, marker='o', color='black', s=10)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('log average nfe vs log ell')
    plt.xlabel('log ell')
    plt.ylabel('log average nfe')
    plt.grid(True, which="both", ls="--")
    plt.savefig('lognfe2logell.png')