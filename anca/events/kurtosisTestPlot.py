import mpld3

def kurtosisTestPlot(kvals, kZscore, kpvalue, conformerSize):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.grid(color='white', linestyle='solid')
        x = range(conformerSize);
        lines = ax.plot(x,kvals[0:conformerSize], marker='o')
        plt.title('Kurtosis Test: Zscore, pvalue calculation on per-conformer basis')
        plt.xlabel('Time Frame', fontsize=20);
        plt.ylabel('Kurtosis ($\kappa$)', fontsize=20);
        plt.grid(True);
        plt.legend();

        labels = ['Zscore: {}, pvalue: {}'.format(i,j) for i,j in zip(kZscore[0:conformerSize],kpvalue[0:conformerSize])];
        mpld3.plugins.connect(fig, mpld3.plugins.PointClickableHTMLTooltip(lines[0],labels=labels))
        mpld3.enable_notebook()
