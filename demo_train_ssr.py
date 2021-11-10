import sys
import os
import json
import utils
import train_ssr



#Configure the experiment
train_ssr.config = dict(
    dataset='Cifar10',
    optimizer='Adam',
    optimizer_decay_at_epochs=[80, 120],
    optimizer_decay_with_factor=10.0,
    optimizer_learning_rate=0.1,
    optimizer_momentum=0.9,
    optimizer_weight_decay=0.0001,
    batch_size=32,
    num_epochs=50,
    seed=42,
    alpha=0.3, # for smooth loss
    scale=1.0  # fro smooth loss
)


train_ssr.output_dir = 'output/'
if not os.path.exists(train_ssr.output_dir):
    os.makedirs(train_ssr.output_dir)

print(train_ssr.config["dataset"], train_ssr.config["optimizer"],
    train_ssr.config["optimizer_learning_rate"], train_ssr.config["optimizer_momentum"])

# Save the config
with open(os.path.join(train_ssr.output_dir, 'config.json'), 'w') as fp:
    json.dump(train_ssr.config, fp, indent=' ')

# Configure the logging of scalar measurements
logfile = utils.logging.JSONLogger(os.path.join(train_ssr.output_dir, 'metrics.json'))
train_ssr.log_metric = logfile.log_metric

# Train
best_accuracy = train_ssr.main()
