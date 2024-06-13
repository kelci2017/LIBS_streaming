import seaborn as sns
import matplotlib.pyplot as plt
from numpy import trapz

def visualize_chunks(x_chunk_list, dx_values=[6, 6, 5, 6, 5, 5], linestyles=['-', '-', '--', '-.', ':', '-.'], labels=['Chunk 0', 'Chunk 1', 'Chunk 2', 'Chunk 3', 'Chunk 4', 'Chunk 5'], filename="chunk_denity.png"):
    for i, dx, linestyle in zip(range(len(x_chunk_list)), dx_values, linestyles):
        sns.distplot(a=trapz(x_chunk_list[i],dx=dx), hist=False, kde_kws={'linestyle':linestyle})

    plt.xlabel("Spectrum area")
    plt.ylabel("Frequency")
    plt.legend(labels=labels, loc="upper right")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()