import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

from sklearn.linear_model import Ridge
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.kernel_ridge import KernelRidge

import gradio as gr
import random


def generate_data(n_samples: int) -> tuple:
    rng = np.random.RandomState(random.randint(0, 1000))
    data = np.linspace(0, 30, num=n_samples).reshape(-1, 1)
    target = np.sin(data).ravel()
    training_sample_indices = rng.choice(
        np.arange(0, int(0.4 * n_samples)), size=int(0.2 * n_samples), replace=False
    )
    training_data = data[training_sample_indices]
    training_noisy_target = (
        target[training_sample_indices] + 0.5 * rng.randn(len(training_sample_indices))
    )

    return data, target, training_data, training_noisy_target


def plot_ridge_and_kernel(n_samples: int) -> plt.figure:
    data, target, training_data, training_noisy_target = generate_data(n_samples)

    ridge = Ridge().fit(training_data, training_noisy_target)
    kernel_ridge = KernelRidge(kernel=ExpSineSquared())
    kernel_ridge.fit(training_data, training_noisy_target)

    fig, ax = plt.subplots(figsize=(8, 4))

    ridge_predictions = ridge.predict(data)
    kernel_ridge_predictions = kernel_ridge.predict(data)

    ax.plot(data, target, label="True signal", linewidth=2)
    ax.scatter(
        training_data,
        training_noisy_target,
        color="black",
        label="Noisy measurements",
    )
    ax.plot(data, ridge_predictions, label="Ridge regression")
    ax.plot(
        data,
        kernel_ridge_predictions,
        label="Kernel ridge",
        linewidth=2,
        linestyle="dashdot",
    )
    ax.fill_between(
        data.ravel(),
        ridge_predictions,
        kernel_ridge_predictions,
        color="lightgrey",
        alpha=0.4,
    )
    ax.legend()
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    _ = ax.set_title("Ridge vs Kernel Ridge with the area between highlighted")

    return fig


def gradio_plot(n_samples: int) -> Image.Image:
    fig = plot_ridge_and_kernel(n_samples)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    im = Image.open(buf)
    return im


inputs = [
    gr.inputs.Slider(minimum=100, maximum=5000, step=100, label="n_samples", default=1000),
]


# Create the Gradio app
title = "Comparison of kernel ridge and Gaussian process regression"
description = "Kernel ridge regression and Gaussian process regression both use the kernel trick to fit data. While kernel ridge regression aims to find a single target function minimizing the loss (mean squared error), Gaussian process regression takes a probabilistic approach, defining a Gaussian posterior distribution over target functions using Bayes' theorem. Essentially, kernel ridge regression seeks one best function, while Gaussian process regression considers a range of probable functions based on prior probabilities and observed data. \n \n link to the official doc https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_compare_gpr_krr.html#sphx-glr-auto-examples-gaussian-process-plot-compare-gpr-krr-py"
iface = gr.Interface(fn=gradio_plot, inputs=inputs, outputs="image", title = title , description = description)
iface.launch()