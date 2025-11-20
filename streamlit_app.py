# streamlit_app.py
import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import base64
import os

# -----------------------
# Model: stress → autophagy → plasticity (extended)
# -----------------------
def model_dynamics(t, y, params, phase='normal'):
    # state: S, D, A, E
    S, D, A, E = y
    # unpack
    k_decay = params['k_decay']
    k_damage = params['k_damage']
    beta = params['beta']
    alpha = params['alpha']
    A0 = params['A0']
    sigma = params['sigma']
    k_maintain = params['k_maintain']
    k_energy = params['k_energy']
    k_atp = params['k_atp']

    # Stress-suppressed baseline autophagy
    A_actual = A0 / (1 + sigma)

    # Optional oscillation to mimic AMPK-mTOR cycles
    ampk_factor = 1 + 0.5 * np.sin(0.1 * t) if phase == 'oscillating' else 1.0
    A_dynamic = A_actual * ampk_factor

    # Energy production reduced by damage
    energy_production = k_atp * (1.0 - D / 2.0)
    energy_cost = k_energy * S
    dEdt = energy_production - energy_cost

    # Maintenance term depends on energy availability (only after stimulus)
    maintenance = k_maintain * S * E if phase == 'post_stim' else 0.0
    dSdt = -k_decay * S - k_damage * D * S + maintenance

    # Damage accumulation and autophagy clearance
    dDdt = beta - alpha * A_dynamic * D

    # Slow autophagy adaptation (homeostatic)
    dAdt = 0.01 * (A_actual - A)

    return [dSdt, dDdt, dAdt, dEdt]

def simulate_protocol(params):
    # initial state S, D, A, E
    y0 = [1.0, 0.1, params['A0'], 0.8]
    t_stim = params['stim_time']
    t_end = 150

    # Baseline
    sol1 = solve_ivp(lambda t, y: model_dynamics(t, y, params, 'normal'),
                     [0, t_stim], y0, t_eval=np.linspace(0, t_stim, 200), method='RK45')

    # Stimulus (instant jump in S)
    y_stim = sol1.y[:, -1].copy()
    y_stim[0] += params['stim_strength']

    # Post stimulus with maintenance dependent on energy
    sol2 = solve_ivp(lambda t, y: model_dynamics(t, y, params, 'post_stim'),
                     [t_stim, t_end], y_stim, t_eval=np.linspace(t_stim, t_end, 600), method='RK45')

    t_all = np.concatenate([sol1.t, sol2.t])
    y_all = np.concatenate([sol1.y, sol2.y], axis=1)
    return t_all, y_all, t_stim

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Autophagy-Plasticity Model", layout="wide")
st.title("Stress → Autophagy → Neural Plasticity (Interactive Model)")
st.write("A simple ODE model demonstrating how stress-modulated autophagy gates synaptic stability and plasticity.")

with st.sidebar:
    st.header("Model Parameters")
    st.subheader("Stress & Autophagy")
    sigma = st.slider("σ (stress level)", 0.0, 10.0, 0.0, 0.1)
    A0 = st.slider("A₀ (baseline autophagy)", 0.2, 2.0, 1.0, 0.1)

    st.subheader("Synaptic Dynamics")
    k_decay = st.slider("k_decay", 0.001, 0.02, 0.005, 0.001)
    k_damage = st.slider("k_damage", 0.05, 1.0, 0.3, 0.05)
    k_maintain = st.slider("k_maintain", 0.0, 0.02, 0.008, 0.001)

    st.subheader("Metabolism")
    k_energy = st.slider("k_energy (energy cost)", 0.01, 0.4, 0.1, 0.01)
    k_atp = st.slider("k_atp (ATP production)", 0.05, 0.6, 0.2, 0.05)

    st.subheader("Damage & Clearance")
    beta = st.slider("β (damage rate)", 0.001, 0.05, 0.02, 0.001)
    alpha = st.slider("α (clearance rate)", 0.01, 0.6, 0.2, 0.01)

    st.subheader("Stimulation Protocol")
    stim_strength = st.slider("LTP stimulus (instant)", 0.0, 2.0, 0.8, 0.05)
    stim_time = st.slider("Stimulus time", 5, 80, 30, 1)

# computed metrics
A_eff = A0 / (1 + sigma)
D_eq = beta / (alpha * A_eff) if (alpha * A_eff) != 0 else np.inf
plasticity_potential = A_eff * max(0.0, 1.0 - D_eq)

# display metrics
col1, col2, col3 = st.columns(3)
col1.metric("Effective Autophagy (A_eff)", f"{A_eff:.3f}")
col2.metric("Damage equilibrium (D_eq)", f"{D_eq:.3f}")
col3.metric("Plasticity potential", f"{plasticity_potential:.3f}")

params = {
    'k_decay': k_decay, 'k_damage': k_damage, 'beta': beta,
    'alpha': alpha, 'A0': A0, 'sigma': sigma, 'k_maintain': k_maintain,
    'k_energy': k_energy, 'k_atp': k_atp, 'stim_strength': stim_strength,
    'stim_time': stim_time
}

t, y, t_stim = simulate_protocol(params)
S, D, A, E = y

# plotting
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
axes[0].plot(t, S, linewidth=2.2, label='Synaptic strength (S)')
axes[0].axvline(t_stim, linestyle='--', label='Stimulus')
axes[0].set_ylabel('S'); axes[0].legend(); axes[0].grid(True)

axes[1].plot(t, D, linewidth=2.2, label='Damage (D)')
axes[1].axhline(D_eq, linestyle=':', label=f'D_eq={D_eq:.3f}')
axes[1].set_ylabel('D'); axes[1].legend(); axes[1].grid(True)

axes[2].plot(t, A, linewidth=2.2, label='Autophagy (A)')
axes[2].axhline(A_eff, linestyle=':', label=f'A_eff={A_eff:.3f}')
axes[2].set_ylabel('A'); axes[2].legend(); axes[2].grid(True)

axes[3].plot(t, E, linewidth=2.2, label='Energy (E)')
axes[3].axhline(0.5, linestyle=':', label='critical ATP ~0.5')
axes[3].set_ylabel('E'); axes[3].set_xlabel('Time'); axes[3].legend(); axes[3].grid(True)

st.pyplot(fig)

# allow downloads: PNGs and PDF
def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import textwrap

# --- Optimized PDF Generation Function ---
def create_pdf_bytes(t, S, D, A, E, params):
    buf = BytesIO()
    # Use a context manager to ensure the file is saved/closed properly
    with PdfPages(buf) as pdf:
        
        # --- Page 1: Title & Abstract ---
        fig_text = plt.figure(figsize=(8.5, 11))
        plt.axis('off') # Hide axes for the text page
        
        # Title Section
        fig_text.text(0.1, 0.9, "The Metabolic Switch:\nAutophagic Flux Gates Synaptic Stability", 
                      fontsize=16, weight='bold', color='darkblue')
        fig_text.text(0.1, 0.85, "Author: Polykleitos Rengos", fontsize=12, style='italic')
        
        # Abstract Section
        fig_text.text(0.1, 0.80, "Abstract:", fontsize=12, weight='bold')
        
        raw_abstract = ("This computational model links metabolic stress, autophagic flux, and "
                        "neural plasticity. Simulations demonstrate that while acute stress "
                        "may be tolerated, chronic suppression of autophagy leads to "
                        "damage accumulation (D) and the failure of LTP maintenance (S).")
        
        # Wrap text so it fits on the page
        wrapped_abstract = textwrap.fill(raw_abstract, width=80)
        fig_text.text(0.1, 0.75, wrapped_abstract, fontsize=11, ha='left', va='top')
        
        # Optional: Print Simulation Parameters on Page 1
        fig_text.text(0.1, 0.60, "Simulation Parameters:", fontsize=12, weight='bold')
        param_str = "\n".join([f"{k}: {v}" for k, v in params.items()])
        fig_text.text(0.1, 0.58, param_str, fontsize=10, fontfamily='monospace', va='top')
        
        pdf.savefig(fig_text)
        plt.close(fig_text)

        # --- Page 2: Data Visualization ---
        fig_plots, axs = plt.subplots(4, 1, figsize=(8.5, 11), sharex=True)
        
        # Synaptic Strength
        axs[0].plot(t, S, color='blue', linewidth=2)
        axs[0].set_ylabel('Strength (S)')
        axs[0].set_title('Synaptic Strength Dynamics', weight='bold')
        axs[0].grid(True, alpha=0.3)
        
        # Damage
        axs[1].plot(t, D, color='red', linewidth=2)
        axs[1].set_ylabel('Damage (D)')
        axs[1].set_title('Cellular Damage Accumulation', weight='bold')
        axs[1].grid(True, alpha=0.3)
        
        # Autophagy
        axs[2].plot(t, A, color='green', linewidth=2)
        axs[2].set_ylabel('Flux (A)')
        axs[2].set_title('Autophagic Flux', weight='bold')
        axs[2].grid(True, alpha=0.3)
        
        # Energy
        axs[3].plot(t, E, color='orange', linewidth=2)
        axs[3].set_ylabel('Energy (E)')
        axs[3].set_title('Metabolic Energy Availability', weight='bold')
        axs[3].set_xlabel('Time (t)')
        axs[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig_plots)
        plt.close(fig_plots)
        
    buf.seek(0)
    return buf

# --- Main App Logic ---

# Generate PDF (assuming t, S, D, A, E, params exist from your simulation)
pdf_buffer = create_pdf_bytes(t, S, D, A, E, params)

st.markdown("### 📄 Results & Report")

# 1. Native Streamlit Download Button (Much faster/cleaner than base64 link)
st.download_button(
    label="📥 Download Full PDF Report",
    data=pdf_buffer,
    file_name="autophagy_plasticity_report.pdf",
    mime="application/pdf",
    use_container_width=True
)

# 2. Optimized "Quick Analysis" Dashboard
st.divider()
st.subheader("📊 Quick Analysis")

# Use columns for metrics instead of a raw dictionary
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Final Strength (S)", f"{S[-1]:.3f}", delta_color="normal")
with col2:
    st.metric("Final Damage (D)", f"{D[-1]:.3f}", delta_color="inverse") # Red if high
with col3:
    st.metric("Final Autophagy (A)", f"{A[-1]:.3f}")
with col4:
    st.metric("Final Energy (E)", f"{E[-1]:.3f}")

# Logical interpretation text
if S[-1] > 1.2:
    status = "✅ **LTP Maintained:** Autophagy successfully cleared damage, allowing consolidation."
    status_color = "success"
else:
    status = "⚠️ **LTP Failed:** Accumulated damage or energy deficit prevented consolidation."
    status_color = "error"

if status_color == "success":
    st.success(status)
else:
    st.error(status)
