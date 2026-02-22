# HMM Baum-Welch Visualizer

A high-performance Python + Streamlit dashboard for visualizing the Expectation-Maximization (EM) process of the Baum-Welch algorithm — ideal for learning and analyzing Hidden Markov Models (HMMs) through interactive state transition diagrams.

## Features

| Feature | Description |
| :--- | :--- |
| **Hybrid Rendering Engine** | Seamlessly switches between high-speed, flicker-free Matplotlib playback (during animation) and interactive, draggable Pyvis network graphs (when paused). |
| **Real-time Convergence Dashboards** | Live-updating Plotly subplots tracking Log-Likelihood, Observation Probability $P(O|\lambda)$, Optimization Loss (NLL), and Error Rate. |
| **Custom Media Player Dock** | A sticky, glassmorphism-styled UI dock for controlling animation playback, speed (1x to 10x), and stepping precisely through mathematical iterations. |
| **Dynamic State Diagram** | Animated 3-tier layout (START $\rightarrow$ Hidden States $\rightarrow$ Observations) with dynamic edge widths, "squircle" nodes, and color-coded probability updates. |
| **Fully Configurable** | Easily adjust the number of Hidden States (N), Observation Symbols (M), Max Iterations, and input custom observation sequences on the fly via the sidebar. |

---

## Installation & Setup

To run this project locally, ensure you have Python installed, then follow these steps:

```bash
# Clone the repository
git clone [https://github.com/YOUR-USERNAME/hmm-baum-welch-visualizer.git](https://github.com/YOUR-USERNAME/hmm-baum-welch-visualizer.git)
cd hmm-baum-welch-visualizer

# Install the required dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
