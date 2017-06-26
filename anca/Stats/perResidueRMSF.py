import numpy

def perResidueRMSF(caDevsMDall, dim=None, Na=None):

xx1 = np.std(caDevsMDall, 1);
rmsf_1 = np.mean(xx1.reshape((dim, Na)),0);
plt.plot(rmsf_1, 'k-');
plt.xlabel('Residue number', fontsize=20);
plt.ylabel('RMSF (\AA)', fontsize=20);
plt.show();
