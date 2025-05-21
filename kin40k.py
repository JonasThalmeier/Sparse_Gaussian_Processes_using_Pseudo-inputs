from max_likelihood import SparseGPModel
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import jax
from uci_datasets import Dataset
import time
from sklearn.metrics import mean_squared_error

# Load data
data = Dataset("kin40k")
x_train, y_train, x_test, y_test = data.get_split(split=0)

# Define number of inducing points to test
inducing_points = np.linspace(5, 100, num=15, dtype=int)  # Logarithmic scale for inducing points
D = x_train.shape[1]

# Initialize arrays to store results
mse_scores = []
train_times = []
inference_times = []

# Ensure consistent sizes between training and test sets
N_train = 500  # Size of training set
N_test = 100   # Size of test set

# Slice the data to use consistent sizes
X = jnp.array(x_train[:N_train])
y = jnp.array(y_train[:N_train]).reshape(-1)
X_test = jnp.array(x_test[:N_test])
y_test = jnp.array(y_test[:N_test]).reshape(-1)

for M in inducing_points:
    print(f"Training with {M} inducing points...")
    
    # Measure training time
    model = SparseGPModel(N_train, D, M, margin=0)
    start_time = time.time()
    f_bar, cov_f_bar, X_bar_opt, log_c_opt, log_b_opt, log_sigma_sq_opt, X_bar_init = model.fit(X, y)
    train_time = time.time() - start_time
    train_times.append(train_time)
    
    # Measure inference time
    start_time = time.time()
    f_bar_test,_ = model.predict(X_test)
    inference_time = time.time() - start_time
    inference_times.append(inference_time)
    
    # Calculate MSE
    mse = mean_squared_error(y_test, f_bar_test)
    mse_scores.append(mse)

# save results to disk
results = {
    'inducing_points': inducing_points,
    'mse_scores': np.array(mse_scores),
    'train_times': np.array(train_times),
    'inference_times': np.array(inference_times),
    'parameters': {
        'N_train': N_train,
        'N_test': N_test,
        'D': D
    }
}
np.savez_compressed(f'simulation_results.npz', **results)

# Define consistent plotting parameters
fig_size = (10, 6)
font_size = 12
plt.style.use('default')
plt.rc('font', size=font_size)
plt.rc('axes', titlesize=font_size+2, labelsize=font_size)
plt.rc('xtick', labelsize=font_size)
plt.rc('ytick', labelsize=font_size)
plt.rc('legend', fontsize=font_size)

# Load results from disk
loaded = np.load('simulation_results.npz', allow_pickle=True)
inducing_points = loaded['inducing_points']
mse_scores = loaded['mse_scores']
train_times = loaded['train_times']
inference_times = loaded['inference_times']
parameters = loaded['parameters'].item()  # Convert numpy array to dict

# Print loaded parameters
print("Loaded parameters:")
print(f"N_train: {parameters['N_train']}")
print(f"N_test: {parameters['N_test']}")
print(f"D: {parameters['D']}\n")

# Create and save MSE plot with logarithmic y-axis
plt.figure(figsize=fig_size)
plt.semilogy(inducing_points, mse_scores, 'k-o', color='black')
plt.xlabel('Number of Inducing Points (M)')
plt.ylabel('Mean Squared Error (log scale)')
plt.title('MSE vs Number of Inducing Points')
plt.grid(True, which="both", ls="-")
plt.tight_layout()
plt.savefig('mse_plot_loaded.png', bbox_inches='tight', dpi=300)
plt.close()

# Create and save training time plot
plt.figure(figsize=fig_size)
plt.plot(inducing_points, train_times, 'k-o', color='black')
plt.xlabel('Number of Inducing Points (M)')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time vs Number of Inducing Points')
plt.grid(True)
plt.tight_layout()
plt.savefig('training_time_plot_loaded.png', bbox_inches='tight', dpi=300)
plt.close()

# Create and save inference time plot
plt.figure(figsize=fig_size)
plt.plot(inducing_points, inference_times, 'k-o', color='black')
plt.xlabel('Number of Inducing Points (M)')
plt.ylabel('Inference Time (seconds)')
plt.title('Inference Time vs Number of Inducing Points')
plt.grid(True)
plt.tight_layout()
plt.savefig('inference_time_plot_loaded.png', bbox_inches='tight', dpi=300)
plt.close()

# Print numerical results
print("Numerical Results:")
print("Inducing Points | MSE | Training Time | Inference Time")
print("-" * 55)
for m, mse, t_time, i_time in zip(inducing_points, mse_scores, train_times, inference_times):
    print(f"{m:14d} | {mse:.4f} | {t_time:.2f}s | {i_time:.2f}s")