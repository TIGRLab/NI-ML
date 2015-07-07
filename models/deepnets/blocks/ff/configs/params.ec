[hyperparams]
net_name: EC_ad_cn
side : r
max_iter: 500
base_lr: 0.000775
train_batch : 128
valid_batch : 4828
test_batch : 1024
hidden_units: 32
# sd = m ^ (-1/2), where m is input connections
W_sd:1
W_mu:0
b_sd:1
b_mu:0
dropout_ratio: 0.7
input_dropout_ratio: 0.2
weight_decay: 0.0
max_norm: 1000.0
solver_type: adagrad
fine_tune: True
data_file : /projects/francisco/data/fuel/standardized/EC_r_ad_cn.h5

# To find min job after spearmint has run:
# $ mongo # starts db cmd line
# > use spearmint
# > jobs = db.ff_ljobs # the jobs table
# > jobs.find().sort({values: {main: 1}}).limit(1).pretty()
