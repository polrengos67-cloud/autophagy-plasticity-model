import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import textwrap

# --- 1. Improved Model with Non-Negativity Constraints ---
def model(t, y, k_decay, k_damage, beta, alpha, A0, sigma, k_maintain, k_atp, k_energy, P_val):
    S, D, A, E = y
    
    # Autophagic Flux (A)
    target_A = A0 / (1 + sigma)
    dAdt = 0.05 * (target_A - A)
    
    # Energy (E): Production - Consumption
    # LOGIC: If E is 0, it cannot drop further.
    production = k_atp * (1 - min(D, 0.9)) # Damage caps production, but doesn't invert it
    consumption = k_energy * S
    dEdt = production - consumption
    if E <= 0 and dEdt < 0: dEdt = 0 # Physical constraint
    
    # Synaptic Strength (S)
    # Maintenance requires E > 0
    maintenance = k_maintain * S * E if E > 0 else 0
    dSdt = -k_decay * S - k_damage * D * S + P_val + maintenance
    
    # Synaptic Damage (D)
    # LOGIC: Damage cannot be negative
    clearance = alpha * A * D
    dDdt = beta - clearance
    if D <= 0 and dDdt < 0: dDdt = 0 # Physical constraint
    
    return [dSdt, dDdt, dAdt, dEdt]

# --- 2. Professional Plotting ---
def create_supergraph(t, S, D, A, E, title):
    fig, axs = plt.subplots(4, 1, figsize=(8.5, 11), sharex=True)
    plt.subplots_adjust(hspace=0.15)

    def style_ax(ax, data, color, label, ylabel, ylim=None):
        ax.plot(t, data, color=color, linewidth=2.5, label=label)
        ax.set_ylabel(ylabel, fontsize=11, weight='bold')
        ax.grid(True, which='major', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ylim: ax.set_ylim(ylim)
        ax.legend(loc='upper right', frameon=False, fontsize=10)

    # Plotting with specific y-limits to look clean
    style_ax(axs[0], S, '#003366', 'Synaptic Strength (S)', 'Strength (a.u.)', ylim=(0, max(S)*1.2))
    style_ax(axs[1], D, '#CC0000', 'Cellular Damage (D)', 'Damage (a.u.)', ylim=(0, max(0.6, max(D)*1.1)))
    style_ax(axs[2], A, '#006600', 'Autophagic Flux (A)', 'Flux (a.u.)', ylim=(0, 1.2))
    style_ax(axs[3], E, '#FF9900', 'Metabolic Energy (E)', 'Energy (a.u.)', ylim=(0, max(E)*1.2))

    axs[0].set_title(title, weight='bold', fontsize=16, pad=20)
    axs[3].set_xlabel('Time (arbitrary units)', fontsize=12, weight='bold')
    
    return fig

# --- 3. Streamlit App ---
st.set_page_config(layout="wide", page_title="Autophagy Model")
st.title("Model Configuration & Export")

col1, col2 = st.columns(2)

# PARAMETERS FOR "FIGURE 1" (CHRONIC STRESS)
# High Sigma, Low Decay
with col1:
    st.header("Figure 1: Chronic Stress")
    if st.button("Generate Figure 1 (Stress)"):
        # Params: High Stress, standard decay
        params = (0.005, 0.8, 0.01, 0.2, 1.0, 4.0, 0.02, 0.3, 0.15)
        # (k_decay, k_damage, beta, alpha, A0, sigma, k_maintain, k_atp, k_energy)
        
        y0 = [1.0, 0.0, 1.0/(1+4.0), 1.0]
        # Simulation...
        # (Run simulation logic here similar to previous code)
        # ...
        # TRICK: Just hardcoding the optimized simulation runner for brevity:
        t_final = 150
        t = np.linspace(0, t_final, 500)
        
        def run_sim(p_val_func):
            res = []
            y_curr = y0
            dt = t[1] - t[0]
            for i in range(len(t)):
                P = p_val_func(t[i])
                dy = model(t[i], y_curr, *params, P)
                y_next = [y_curr[j] + dy[j]*dt for j in range(4)]
                y_curr = y_next
                res.append(y_curr)
            return np.array(res).T

        # Stimulus 30-60
        P_func = lambda x: 1.5 if 30 <= x <= 60 else 0
        res = run_sim(P_func)
        
        fig = create_supergraph(t, res[0], res[1], res[2], res[3], "Suboptimal Condition: Chronic Stress")
        
        # Download
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        st.download_button("Download Figure 1 (Stress).png", buf, "figure1.png", "image/png")
        st.pyplot(fig)

# PARAMETERS FOR "FIGURE 2" (OPTIMIZED)
# Zero Stress, slightly higher maintain, same decay
with col2:
    st.header("Figure 2: Healthy Control")
    if st.button("Generate Figure 2 (Healthy)"):
        # Params: Zero Stress
        params = (0.005, 0.8, 0.01, 0.2, 1.0, 0.0, 0.02, 0.3, 0.15)
        
        y0 = [1.0, 0.0, 1.0, 2.0] # Start with healthy energy
        
        t_final = 150
        t = np.linspace(0, t_final, 500)
        
        def run_sim(p_val_func):
            res = []
            y_curr = y0
            dt = t[1] - t[0]
            for i in range(len(t)):
                P = p_val_func(t[i])
                dy = model(t[i], y_curr, *params, P)
                y_next = [y_curr[j] + dy[j]*dt for j in range(4)]
                y_curr = y_next
                res.append(y_curr)
            return np.array(res).T

        P_func = lambda x: 1.5 if 30 <= x <= 60 else 0
        res = run_sim(P_func)
        
        fig = create_supergraph(t, res[0], res[1], res[2], res[3], "Optimized Condition: Healthy Control")
        
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        st.download_button("Download Figure 2 (Healthy).png", buf, "figure2.png", "image/png")
        st.pyplot(fig)