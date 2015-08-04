import numpy as np
import tables as tb
from PCA_Utils import transform_PCA
from PCA_Utils import score_reconstructions

input_node = 'r_hc_features'
label_node = 'label'

data_path = '/projects/francisco/data/caffe/standardized/combined/ad_cn_train.h5'
data_path_test = '/projects/francisco/data/caffe/standardized/combined/ad_cn_test.h5'

data = tb.open_file(data_path, 'r')
train_X = data.get_node('/' + input_node)[:]
train_y = data.get_node('/' + label_node)[:]
data.close()

data = tb.open_file(data_path_test, 'r')
test_X = data.get_node('/' + input_node)[:]
test_y = data.get_node('/' + label_node)[:]
test_fused_X = data.get_node('/' + input_node + '_fused')[:]
test_fused_y = data.get_node('/' + label_node + '_fused')[:]
data.close()


# Try PCA with various num_components and find the best DICE score from the resulting reconstructions.
K = []
V_E = []
K_comps = [4, 16, 64, 128, 256, 328, 512, 1024]
for k in K_comps:
    print 'Fitting PCA with {} components'.format(k)
    pca, X_hat_pca = transform_PCA(k, train_X, test_fused_X)
    D = score_reconstructions(test_fused_X, X_hat_pca)
    K.append(np.mean(D))
    V_E = np.sum(pca.explained_variance_ratio_)

best_ind = np.argmax(K)
print 'Best DICE score {} from {} components'.format(K[best_ind], K_comps[best_ind])
print 'With {} variance explained'.format(V_E[best_ind])
