import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
import textwrap

# --- Optimized Model Definition ---
def model(t, y, k_decay, k_damage, beta, alpha, A0, sigma, k_maintain, k_atp, k_energy, P_val):
    S, D, A, E = y
    
    # Apply bounds to state variables (cleaner than bounding derivatives)
    S = max(0, S)
    D = max(0, min(1.0, D))
    A = max(0, A)
    E = max(0.1, min(5.0, E))
    
    # Autophagic Flux: adapts to stress level
    target_A = A0 / (1 + sigma)
    dAdt = 0.2 * (target_A - A)
    
    # Energy: homeostatic regulation
    production = k_atp * (1 - 0.5 * D)
    consumption = k_energy * S
    dEdt = production - consumption
    
    # Synaptic Strength: energy-dependent maintenance
    maintenance = k_maintain * S * E
    dSdt = -k_decay * S - k_damage * D * S + P_val + maintenance
    
    # Synaptic Damage: production vs clearance
    clearance = alpha * A * D
    dDdt = beta - clearance
    
    return [dSdt, dDdt, dAdt, dEdt]

# [REST OF YOUR DOCUMENT 13 CODE STAYS EXACTLY THE SAME]
# ... (all the plotting, PDF, and Streamlit code from Document 13)
