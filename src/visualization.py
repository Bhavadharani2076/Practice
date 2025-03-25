import matplotlib.pyplot as plt
import io
import base64

def generate_plot(df):
    """Generate a plot of the dataset."""
    plt.figure(figsize=(12, 8))

    # Plot each signal
    for i, signal in enumerate(df.columns[:-1]):  # Exclude the Target_Health_Status column
        plt.subplot(len(df.columns) - 1, 1, i + 1)
        plt.plot(df[signal], label=signal)
        plt.ylabel(signal)
        plt.legend()

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    return plot_url