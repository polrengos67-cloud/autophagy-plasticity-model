import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import textwrap

# --- 1. Model Definition ---
def model(t, y, k_decay, k_damage, beta, alpha, A0, sigma, k_maintain, k_atp, k_energy, P_val):
    S, D, A, E = y
    
    # Autophagic Flux (A)
    target_A = A0 / (1 + sigma)
    dAdt = 0.05 * (target_A - A)
    
    # Energy (E): Production - Consumption (Non-negative constraint)
    # We use a "soft" clamp in the derivative logic to prevent runaway negatives
    production = k_atp * (1 - min(D, 0.9)) 
    consumption = k_energy * S
    dEdt = production - consumption
    # If Energy is 0 or less and trying to drop, stop it
    if E <= 0 and dEdt < 0: dEdt = 0 
    
    # Synaptic Strength (S)
    # Maintenance requires Energy > 0
    maintenance = k_maintain * S * E if E > 0 else 0
    dSdt = -k_decay * S - k_damage * D * S + P_val + maintenance
    
    # Synaptic Damage (D) (Non-negative)
    clearance = alpha * A * D
    dDdt = beta - clearance
    if D <= 0 and dDdt < 0: dDdt = 0 
    
    return [dSdt, dDdt, dAdt, dEdt]

# --- 2. Plotting Helper (The Supergraph) ---
def create_supergraph(t, S, D, A, E, title):
    fig, axs = plt.subplots(4, 1, figsize=(8.5, 11), sharex=True)
    plt.subplots_adjust(hspace=0.15)

    def style_ax(ax, data, color, label, ylabel):
        ax.plot(t, data, color=color, linewidth=2.5, label=label)
        ax.set_ylabel(ylabel, fontsize=10, weight='bold')
        ax.grid(True, which='major', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', frameon=False, fontsize=9)

    style_ax(axs[0], S, '#003366', 'Synaptic Strength (S)', 'Strength (a.u.)')
    style_ax(axs[1], D, '#CC0000', 'Cellular Damage (D)', 'Damage (a.u.)')
    style_ax(axs[2], A, '#006600', 'Autophagic Flux (A)', 'Flux (a.u.)')
    style_ax(axs[3], E, '#FF9900', 'Metabolic Energy (E)', 'Energy (a.u.)')

    axs[0].set_title(title, weight='bold', fontsize=14, pad=15)
    axs[3].set_xlabel('Time (arbitrary units)', fontsize=11, weight='bold')
    
    return fig

# --- 3. PDF Report Generator ---
def create_pdf_bytes(fig_plots, params):
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        
        # --- Page 1: Title & Abstract ---
        fig_text = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        
        fig_text.text(0.1, 0.9, "The Metabolic Switch:\nAutophagic Flux Gates Synaptic Stability", 
                      fontsize=16, weight='bold', color='#003366')
        fig_text.text(0.1, 0.85, "Author: Polykleitos Rengos", fontsize=12, style='italic')
        
        fig_text.text(0.1, 0.80, "Abstract & Hypothesis:", fontsize=12, weight='bold')
        
        # Corrected Abstract text regarding Lateral Habenula vs PFC
        raw_abstract = (
            "This computational model explores the interplay between metabolic stress, "
            "autophagic flux, and synaptic plasticity. While stress-induced suppression "
            "of autophagy has been definitively characterized in the lateral habenula "
            "(Yang et al., 2025), this model tests the hypothesis that similar "
            "mTOR-mediated mechanisms operate in the Prefrontal Cortex (PFC). "
            "Simulations demonstrate that chronic suppression of clearance mechanisms "
            "leads to damage accumulation and the metabolic failure of LTP maintenance."
        )
        
        wrapped_abstract = textwrap.fill(raw_abstract, width=80)
        fig_text.text(0.1, 0.75, wrapped_abstract, fontsize=11, ha='left', va='top', linespacing=1.5)
        
        fig_text.text(0.1, 0.60, "Simulation Parameters (a.u.):", fontsize=12, weight='bold')
        param_str = "\n".join([f"{k}: {v}" for k, v in params.items()])
        fig_text.text(0.1, 0.58, param_str, fontsize=10, fontfamily='monospace', va='top')
        
        pdf.savefig(fig_text)
        plt.close(fig_text)

        # --- Page 2: The Supergraph ---
        pdf.savefig(fig_plots)
        
    buf.seek(0)
    return buf

# --- 4. Streamlit App Interface ---
st.set_page_config(page_title="Autophagy Model", layout="wide")
st.title("Interactive Model: Stress, Autophagy, and Synaptic Plasticity")

with st.sidebar:
    st.header("Simulation Control")
    
    st.subheader("1. Biological Constants")
    k_decay = st.slider("k_decay (Natural Forgetting)", 0.001, 0.02, 0.005, format="%.3f")
    k_damage = st.slider("k_damage (Damage Sensitivity)", 0.1, 2.0, 0.8)
    k_maintain = st.slider("k_maintain (Energy Dependency)", 0.0, 0.05, 0.02, format="%.3f")
    
    st.subheader("2. Metabolic Rates")
    beta = st.slider("β (Waste Production)", 0.001, 0.1, 0.01, format="%.3f")
    alpha = st.slider("α (Clearance Efficiency)", 0.1, 1.0, 0.2)
    k_atp = st.slider("ATP Production Rate", 0.1, 1.0, 0.3)
    k_energy = st.slider("Synaptic Energy Cost", 0.05, 0.5, 0.15)

    st.subheader("3. Conditions")
    sigma = st.slider("σ (Metabolic Stress)", 0.0, 10.0, 0.0)
    p_strength = st.slider("LTP Stimulus Strength", 0.0, 5.0, 1.5)
    p_duration = st.slider("Stimulus Duration", 1, 50, 30)

# --- 5. Simulation Execution ---
# Initial Conditions [S, D, A, E]
# Start with healthy energy (E=2.0) so we don't crash immediately
y0 = [1.0, 0.0, 1.0/(1+sigma), 2.0] 

# Simulation Parameters
t_stim_start = 30
t_stim_end = t_stim_start + p_duration
t_final = 150
params = (k_decay, k_damage, beta, alpha, 1.0, sigma, k_maintain, k_atp, k_energy)

# Run Simulation in 3 Phases (Before, During, After Stimulus)
# Phase 1
t1 = np.linspace(0, t_stim_start, 100)
sol1 = solve_ivp(model, [0, t_stim_start], y0, args=(*params, 0.0), t_eval=t1)

# Phase 2 (Stimulus ON)
y1 = sol1.y[:, -1]
t2 = np.linspace(t_stim_start, t_stim_end, 100)
sol2 = solve_ivp(model, [t_stim_start, t_stim_end], y1, args=(*params, p_strength), t_eval=t2)

# Phase 3 (Stimulus OFF)
y2 = sol2.y[:, -1]
t3 = np.linspace(t_stim_end, t_final, 200)
sol3 = solve_ivp(model, [t_stim_end, t_final], y2, args=(*params, 0.0), t_eval=t3)

# Combine Data
t = np.concatenate([sol1.t, sol2.t, sol3.t])
S = np.concatenate([sol1.y[0], sol2.y[0], sol3.y[0]])
D = np.concatenate([sol1.y[1], sol2.y[1], sol3.y[1]])
A = np.concatenate([sol1.y[2], sol2.y[2], sol3.y[2]])
E = np.concatenate([sol1.y[3], sol2.y[3], sol3.y[3]])

# --- 6. Output Display ---
col_main, col_info = st.columns([3, 1])

with col_main:
    # Dynamically title the graph based on stress level
    graph_title = "Optimized Condition: Healthy Control" if sigma < 1.0 else "Suboptimal Condition: Chronic Stress"
    fig_super = create_supergraph(t, S, D, A, E, graph_title)
    st.pyplot(fig_super)

with col_info:
    st.subheader("Results")
    st.metric("Final Strength (S)", f"{S[-1]:.3f}")
    st.metric("Final Damage (D)", f"{D[-1]:.3f}")
    
    param_dict = {
        "k_decay": k_decay, "k_damage": k_damage, "sigma": sigma,
        "beta": beta, "alpha": alpha, "Stimulus": p_strength
    }
    
    pdf_data = create_pdf_bytes(fig_super, param_dict)
    
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_data,
        file_name="autophagy_plasticity_results.pdf",
        mime="application/pdf",
        use_container_width=True
    )
    
    st.info("Adjust sliders to see how Stress (σ) breaks the Energy-Plasticity coupling.")