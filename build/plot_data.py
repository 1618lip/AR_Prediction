import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Helper Functions ---
def read_vector(filename):
    """Read a text file containing one float per line and return as a list of floats."""
    with open(filename, 'r') as f:
        return [float(line.strip()) for line in f if line.strip()]

def read_csv(filename):
    """Read a CSV file (header + numeric rows) and return a dict of lists."""
    data = {}
    with open(filename, 'r') as f:
        header = f.readline().strip().split(',')
        for h in header:
            data[h] = []
        for line in f:
            values = line.strip().split(',')
            for h, v in zip(header, values):
                data[h].append(float(v))
    return data

# --- Plot Error Metrics vs AR Order ---
def plot_error_metrics():
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    orders = [int(x) for x in read_vector("ar_orders.txt")]
    mses = read_vector("ar_mses.txt")
    rmses = read_vector("ar_rmses.txt")
    mapes = read_vector("ar_mapes.txt")

    # Plot MSE vs AR Order
    plt.figure(figsize=(8,5))
    plt.plot(orders, mses, "bo-")
    plt.xlabel("AR Order")
    plt.ylabel("MSE")
    plt.title("MSE vs. AR Order")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "mse_vs_ar_order.png"))
    plt.close()

    # Plot RMSE vs AR Order
    plt.figure(figsize=(8,5))
    plt.plot(orders, rmses, "ro-")
    plt.xlabel("AR Order")
    plt.ylabel("RMSE")
    plt.title("RMSE vs. AR Order")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "rmse_vs_ar_order.png"))
    plt.close()

    # Plot MAPE vs AR Order
    plt.figure(figsize=(8,5))
    plt.plot(orders, mapes, "go-")
    plt.xlabel("AR Order")
    plt.ylabel("MAPE (%)")
    plt.title("MAPE vs. AR Order")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "mape_vs_ar_order.png"))
    plt.close()

    print("Error metric plots saved in 'plots' folder.")

# --- Plot AR Coefficients ---
def plot_ar_coefficients():
    plots_dir = "plots"
    ar_coeffs = read_vector("ar_coefficients.txt")
    plt.figure(figsize=(8,5))
    x = np.arange(1, len(ar_coeffs)+1)
    plt.bar(x, ar_coeffs, color="orange")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Coefficient Value")
    plt.title("AR Model Coefficients (Differenced Data)")
    plt.grid(True, axis="y")
    plt.savefig(os.path.join(plots_dir, "ar_coefficients.png"))
    plt.close()
    print("AR coefficients plot saved.")

# --- Plot Historical and Forecasted Log-Returns ---
def plot_log_returns():
    plots_dir = "plots"
    log_returns = read_vector("log_returns.txt")
    forecasted_returns = read_vector("forecasted_returns.txt")
    plt.figure(figsize=(10,5))
    x1 = np.arange(len(log_returns))
    plt.plot(x1, log_returns, "b-o", label="Training Log-Returns")
    x2 = np.arange(len(log_returns), len(log_returns) + len(forecasted_returns))
    plt.plot(x2, forecasted_returns, "r--", label="Forecasted Log-Returns")
    plt.xlabel("Time Index")
    plt.ylabel("Log-Return")
    plt.title("Historical and Forecasted Log-Returns")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "log_returns_comparison.png"))
    plt.close()
    print("Log-returns comparison plot saved.")

# --- Animate Stock Price Forecast ---
def animate_forecast():
    # Load data from text files.
    train_prices = np.array(read_vector("train_prices.txt"))
    actual_future_prices = np.array(read_vector("actual_future_prices.txt"))
    forecasted_prices = np.array(read_vector("forecasted_prices.txt"))
    train_time = np.array(read_vector("train_time_indices.txt"))
    forecast_time = np.array(read_vector("forecast_time_indices.txt"))
    
    # Total time axis for training + validation.
    full_time = np.concatenate((train_time, forecast_time))
    full_actual = np.concatenate((train_prices, actual_future_prices))
    
    # Setup the figure and axes.
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(train_time, train_prices, "bo-", label="Training Prices")
    ax.plot(forecast_time, actual_future_prices, "g*-", label="Actual Future Prices")
    # We will animate the forecasted prices.
    forecast_line, = ax.plot([], [], "ro-", label="Forecasted Prices", linewidth=2)
    
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Price")
    ax.set_title("Animated Forecast of Stock Prices")
    ax.grid(True)
    ax.legend()
    
    # Animation function: update the forecast_line for each frame.
    def update(frame):
        # frame goes from 0 to len(forecast_time)
        if frame == 0:
            forecast_line.set_data([], [])
        else:
            # Animate forecasted prices up to current frame.
            x = forecast_time[:frame]
            y = forecasted_prices[:frame]
            forecast_line.set_data(x, y)
        return forecast_line,
    
    anim = FuncAnimation(fig, update, frames=len(forecast_time)+1, interval=500, blit=True)
    
    # Save the animation as MP4 (requires ffmpeg) or display interactively.
    anim.save(os.path.join("plots", "forecast_animation.mp4"), writer="ffmpeg", dpi=200)
    plt.close()
    print("Forecast animation saved to 'plots/forecast_animation.mp4'.")

# --- Main Function ---
def main():
    # Create plots folder if not exists.
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Generate all static plots.
    plot_error_metrics()
    plot_ar_coefficients()
    plot_log_returns()
    
    # Animate the forecast.
    animate_forecast()
    
    print("All plots and animation generated.")

if __name__ == "__main__":
    main()



