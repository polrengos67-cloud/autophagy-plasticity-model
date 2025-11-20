import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import textwrap

# --- 1. Model Definition (The "10/10" Logic) ---
def model(t, y, k_decay, k_damage, beta, alpha, A0, sigma, k_maintain, k_atp, k_energy, P_val):
    S, D, A, E = y
    
    # Autophagic Flux (A) adapts to stress
    target_A = A0 / (1 + sigma)
    dAdt = 0.05 * (target_A - A)  # Slow adaptation
    
    # Synaptic Strength (S) with Energy-dependent maintenance
    # Note: P_val is the external stimulus (0 or value) passed in
    dSdt = -k_decay * S - k_damage * D * S + P_val + k_maintain * S * E
    
    # Synaptic Damage (D)
    dDdt = beta - alpha * A * D
    
    # Energy (E): Production impaired by damage, consumption by S
    dEdt = k_atp * (1 - D/2) - k_energy * S
    
    return [dSdt, dDdt, dAdt, dEdt]

# --- 2. PDF Generation (Fixed Yang/Habenula Claim + Pro Graphs) ---
def create_pdf_bytes(t, S, D, A, E, params):
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        
        # --- Page 1: Title & Corrected Abstract ---
        fig_text = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        
        fig_text.text(0.1, 0.9, "The Metabolic Switch:\nAutophagic Flux Gates Synaptic Stability", 
                      fontsize=16, weight='bold', color='#003366')
        fig_text.text(0.1, 0.85, "Author: Polykleitos Rengos", fontsize=12, style='italic')
        
        fig_text.text(0.1, 0.80, "Abstract & Hypothesis:", fontsize=12, weight='bold')
        
        # *** CRITICAL FIX: Addressing the Regional Specificity ***
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
        
        # Parameters List
        fig_text.text(0.1, 0.60, "Simulation Parameters (a.u.):", fontsize=12, weight='bold')
        param_str = "\n".join([f"{k}: {v}" for k, v in params.items()])
        fig_text.text(0.1, 0.58, param_str, fontsize=10, fontfamily='monospace', va='top')
        
        pdf.savefig(fig_text)
        plt.close(fig_text)

        # --- Page 2: The Professional "Supergraph" ---
        # 4 Subplots, Shared X-axis, Publication Style
        fig, axs = plt.subplots(4, 1, figsize=(8.5, 11), sharex=True)
        plt.subplots_adjust(hspace=0.1) # Tight vertical spacing

        # Plot Styling Helper
        def style_ax(ax, data, color, label, ylabel):
            ax.plot(t, data, color=color, linewidth=2, label=label)
            ax.set_ylabel(ylabel, fontsize=10, weight='bold')
            ax.grid(True, which='major', linestyle='--', alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(loc='upper right', frameon=False, fontsize=9)

        style_ax(axs[0], S, '#003366', 'Synaptic Strength (S)', 'Strength (a.u.)')
        style_ax(axs[1], D, '#CC0000', 'Cellular Damage (D)', 'Damage (a.u.)')
        style_ax(axs[2], A, '#006600', 'Autophagic Flux (A)', 'Flux (a.u.)')
        style_ax(axs[3], E, '#FF9900', 'Metabolic Energy (E)', 'Energy (a.u.)')

        axs[0].set_title('Dynamics of Stress-Induced Plasticity Failure', weight='bold', fontsize=14, pad=15)
        axs[3].set_xlabel('Time (arbitrary units)', fontsize=11, weight='bold')

        pdf.savefig(fig)
        plt.close(fig)
        
    buf.seek(0)
    return buf

# --- 3. Streamlit App Layout ---
st.set_page_config(page_title="Autophagy Model", layout="wide")
st.title("Interactive Model: Stress, Autophagy, and Synaptic Plasticity")

with st.sidebar:
    st.header("Simulation Control")
    
    st.subheader("1. Biological Constants")
    # The "10/10" Low Decay Rate
    k_decay = st.slider("k_decay (Natural Forgetting)", 0.001, 0.02, 0.003, format="%.3f")
    k_damage = st.slider("k_damage (Damage Sensitivity)", 0.1, 2.0, 0.8)
    k_maintain = st.slider("k_maintain (Energy Dependency)", 0.0, 0.05, 0.015, format="%.3f")
    
    st.subheader("2. Metabolic Rates")
    beta = st.slider("β (Waste Production)", 0.001, 0.1, 0.005, format="%.3f")
    alpha = st.slider("α (Clearance Efficiency)", 0.1, 1.0, 0.2)
    k_atp = st.slider("ATP Production Rate", 0.1, 1.0, 0.2)
    k_energy = st.slider("Synaptic Energy Cost", 0.05, 0.5, 0.1)

    st.subheader("3. Conditions")
    sigma = st.slider("σ (Metabolic Stress)", 0.0, 10.0, 0.0)
    p_strength = st.slider("LTP Stimulus Strength", 0.0, 5.0, 1.0)
    p_duration = st.slider("Stimulus Duration", 1, 50, 30)

# --- 4. Simulation Execution (Piecewise for Sustained Stimulus) ---
y0 = [1.0, 0.0, 1.0/(1+sigma), 0.8] # Initial S, D, A, E
t_stim_start = 30
t_stim_end = t_stim_start + p_duration
t_final = 150

params = (k_decay, k_damage, beta, alpha, 1.0, sigma, k_maintain, k_atp, k_energy)

# Phase 1: Pre-stimulus
t1 = np.linspace(0, t_stim_start, 100)
sol1 = solve_ivp(model, [0, t_stim_start], y0, args=(*params, 0.0), t_eval=t1)

# Phase 2: During Stimulus (Sustained P)
y1 = sol1.y[:, -1]
t2 = np.linspace(t_stim_start, t_stim_end, 100)
sol2 = solve_ivp(model, [t_stim_start, t_stim_end], y1, args=(*params, p_strength), t_eval=t2)

# Phase 3: Post-stimulus
y2 = sol2.y[:, -1]
t3 = np.linspace(t_stim_end, t_final, 200)
sol3 = solve_ivp(model, [t_stim_end, t_final], y2, args=(*params, 0.0), t_eval=t3)

# Combine
t = np.concatenate([sol1.t, sol2.t, sol3.t])
S = np.concatenate([sol1.y[0], sol2.y[0], sol3.y[0]])
D = np.concatenate([sol1.y[1], sol2.y[1], sol3.y[1]])
A = np.concatenate([sol1.y[2], sol2.y[2], sol3.y[2]])
E = np.concatenate([sol1.y[3], sol2.y[3], sol3.y[3]])

# --- 5. Display & Download ---
col_main, col_info = st.columns([3, 1])

with col_main:
    # Preview Plot (Simplified for screen)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, S, label='Synaptic Strength (S)', color='#003366', linewidth=2)
    ax.plot(t, D, label='Damage (D)', color='#CC0000', linestyle='--')
    ax.set_xlabel("Time (a.u.)")
    ax.set_ylabel("Normalized Value (a.u.)")
    ax.set_title("Real-time Simulation Preview")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

with col_info:
    st.success(f"**Final Strength:** {S[-1]:.3f}")
    st.info(f"**Final Damage:** {D[-1]:.3f}")
    
    # Parameter dict for the PDF
    param_dict = {
        "k_decay": k_decay, "k_damage": k_damage, "sigma": sigma,
        "beta": beta, "alpha": alpha, "Stimulus": p_strength
    }
    
    pdf_data = create_pdf_bytes(t, S, D, A, E, param_dict)
    
    st.download_button(
        label="📄 Download Publication Report",
        data=pdf_data,
        file_name="autophagy_plasticity_results.pdf",
        mime="application/pdf",
        use_container_width=True
    )
    
    st.caption("Note: PDF includes high-res plots and corrected regional hypothesis text.")
