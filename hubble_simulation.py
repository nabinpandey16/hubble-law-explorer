"""
hubble_simulation.py
--------------------
Standalone Python simulation for Hubble's Law Explorer.
Covers three levels of analysis matching the web modules.

Requirements:
    pip install numpy matplotlib scipy

Usage:
    python hubble_simulation.py

Author: Nabin Pandey
Institution: Tri-Chandra Multiple Campus, Tribhuvan University
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONSTANTS AND STYLING
# ============================================================

plt.rcParams.update({
    'figure.facecolor': '#04060f',
    'axes.facecolor': '#080c1a',
    'axes.edgecolor': '#1e2d55',
    'axes.labelcolor': '#d8e4f0',
    'xtick.color': '#6a7e99',
    'ytick.color': '#6a7e99',
    'text.color': '#d8e4f0',
    'grid.color': '#1e2d55',
    'grid.alpha': 0.5,
    'font.family': 'monospace',
})

COLORS = {
    'accent': '#4f8cff',
    'amber': '#f5a623',
    'teal': '#00c9a7',
    'red': '#ff5f6d',
    'dim': '#6a7e99',
}

# Physical constants
H0_FIDUCIAL = 70.0          # km/s/Mpc
MPC_TO_KM = 3.086e19        # 1 Mpc in km
GYR_TO_S = 3.156e16         # 1 Gyr in seconds
H0_SI = H0_FIDUCIAL * 1e3 / MPC_TO_KM  # H0 in s^-1


# ============================================================
# MODULE 1: BEGINNER — VISUALIZING HUBBLE EXPANSION
# ============================================================

def plot_beginner():
    """
    Animated raisin-bread analogy showing uniform expansion.
    All velocities proportional to distance from any observer.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("HUBBLE'S LAW — BEGINNER MODULE\nThe Expanding Universe",
                 fontsize=11, color=COLORS['accent'], y=0.98)

    # Left panel: snapshot of expansion at two times
    ax1 = axes[0]
    ax1.set_title("Galaxy Positions: Before and After Expansion", fontsize=9, color=COLORS['dim'])

    np.random.seed(42)
    N = 15
    angles = np.random.uniform(0, 2*np.pi, N)
    radii = np.random.uniform(0.2, 1.0, N)
    x0 = radii * np.cos(angles)
    y0 = radii * np.sin(angles)

    scale_factor = 2.0
    x1 = x0 * scale_factor
    y1 = y0 * scale_factor

    for i in range(N):
        ax1.annotate('', xy=(x1[i], y1[i]), xytext=(x0[i], y0[i]),
                     arrowprops=dict(arrowstyle='->', color=COLORS['accent'], alpha=0.5, lw=1.2))

    ax1.scatter(x0, y0, s=40, color=COLORS['dim'], alpha=0.6, zorder=5, label='Initial position')
    ax1.scatter(x1, y1, s=60, color=COLORS['teal'], zorder=6, label='After expansion (a=2)')
    ax1.scatter(0, 0, s=120, color=COLORS['amber'], zorder=7, marker='*', label='Observer (you)')

    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(True, alpha=0.2)
    ax1.set_xlabel('x (arbitrary units)')
    ax1.set_ylabel('y (arbitrary units)')

    # Right panel: velocity vs distance (Hubble diagram, basic)
    ax2 = axes[1]
    ax2.set_title("Recession Speed vs Distance", fontsize=9, color=COLORS['dim'])

    distances = radii * 100  # scale to Mpc-like units
    velocities = H0_FIDUCIAL * distances
    velocities_scatter = velocities + np.random.normal(0, 300, N)

    d_range = np.linspace(0, 110, 200)
    ax2.plot(d_range, H0_FIDUCIAL * d_range, color=COLORS['accent'], lw=2, label=f'v = H0 × d  (H0={H0_FIDUCIAL})')
    ax2.scatter(distances, velocities_scatter, s=50, color=COLORS['amber'],
                alpha=0.85, zorder=5, label='Observed galaxies')

    ax2.set_xlabel('Distance (Mpc)')
    ax2.set_ylabel('Recession Velocity (km/s)')
    ax2.legend(fontsize=8)
    ax2.grid(True)

    # Annotate a sample galaxy
    idx = 7
    ax2.annotate(f'd = {distances[idx]:.0f} Mpc\nv = {velocities_scatter[idx]:.0f} km/s',
                 xy=(distances[idx], velocities_scatter[idx]),
                 xytext=(distances[idx]+10, velocities_scatter[idx]+500),
                 fontsize=7, color=COLORS['teal'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['teal'], lw=0.8))

    plt.tight_layout()
    fig.text(0.5, 0.01,
             "The farther a galaxy, the faster it recedes — Hubble's Law: v = H0 × d",
             ha='center', fontsize=8, color=COLORS['dim'], style='italic')
    plt.savefig('beginner_hubble.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("Saved: beginner_hubble.png")
    plt.show()


# ============================================================
# MODULE 2: INTERMEDIATE — MEASUREMENT AND FITTING
# ============================================================

def hubble_model(d, H0):
    """Linear model for curve fitting: v = H0 * d"""
    return H0 * d


def plot_intermediate():
    """
    Interactive Hubble diagram with distance ladder uncertainty,
    redshift measurement, and least-squares fitting.
    """
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("HUBBLE'S LAW — INTERMEDIATE MODULE\nMeasurement, Redshift, and the Distance Ladder",
                 fontsize=11, color=COLORS['accent'])

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax_hub = fig.add_subplot(gs[0, :2])  # Main Hubble diagram
    ax_red = fig.add_subplot(gs[0, 2])   # Redshift spectrum schematic
    ax_ld  = fig.add_subplot(gs[1, :])   # Distance ladder

    # ----- HUBBLE DIAGRAM -----
    np.random.seed(12)
    N = 50
    true_H0 = 70.0
    distances = np.random.uniform(10, 500, N)
    true_v = true_H0 * distances
    # Peculiar velocity scatter + measurement uncertainty
    peculiar = np.random.normal(0, 400, N)
    meas_err = np.random.normal(0, 0.08 * true_v)
    observed_v = true_v + peculiar + meas_err

    # Fit
    popt, pcov = curve_fit(hubble_model, distances, observed_v)
    fitted_H0 = popt[0]
    sigma_H0 = np.sqrt(pcov[0, 0])

    d_line = np.linspace(0, 520, 300)
    ax_hub.plot(d_line, true_H0 * d_line, '--', color=COLORS['teal'],
                alpha=0.5, lw=1.5, label=f'True H0 = {true_H0} km/s/Mpc')
    ax_hub.plot(d_line, fitted_H0 * d_line, color=COLORS['accent'],
                lw=2, label=f'Fitted H0 = {fitted_H0:.1f} ± {sigma_H0:.1f} km/s/Mpc')

    # Color points by distance quintile (proxy for measurement method)
    quintile_colors = [COLORS['teal'], COLORS['accent'], COLORS['amber'], '#a78bfa', COLORS['red']]
    bins = np.percentile(distances, [0, 20, 40, 60, 80, 100])
    methods = ['Parallax/Cepheid', 'Cepheid', 'TF relation', 'SNe Ia near', 'SNe Ia far']
    for q in range(5):
        mask = (distances >= bins[q]) & (distances < bins[q+1])
        ax_hub.scatter(distances[mask], observed_v[mask], s=35, color=quintile_colors[q],
                       alpha=0.8, zorder=5, label=methods[q])

    ax_hub.set_xlabel('Distance (Mpc)')
    ax_hub.set_ylabel('Recession Velocity (km/s)')
    ax_hub.set_title('Hubble Diagram with Measurement Scatter', fontsize=9)
    ax_hub.legend(fontsize=6.5, loc='upper left')
    ax_hub.grid(True)
    ax_hub.set_xlim(0, 520)
    ax_hub.set_ylim(min(observed_v)*0.9, max(observed_v)*1.05)

    # ----- REDSHIFT SCHEMATIC -----
    wavelengths = np.linspace(380, 700, 500)
    em_line = 656.3  # H-alpha rest
    z = 0.08

    def spectrum(wl, center, shift=0):
        gauss = np.exp(-0.5*((wl - center + shift) / 3)**2) * 1.5
        cont = 0.3 + np.random.normal(0, 0.03, len(wl))
        return np.maximum(0, cont - gauss)

    cmap = plt.cm.Spectral_r
    for i, wl in enumerate(wavelengths):
        norm = (wl - 380) / (700 - 380)
        ax_red.axvline(wl, color=cmap(norm), alpha=0.6, lw=0.8)

    ax_red.axvline(em_line, color='white', lw=2, alpha=0.9, label=f'H-alpha rest: {em_line} nm')
    obs_line = em_line * (1 + z)
    ax_red.axvline(obs_line, color='red', lw=2, alpha=0.9, label=f'Observed: {obs_line:.0f} nm (z={z})')

    ax_red.annotate('', xy=(obs_line, 0.7), xytext=(em_line, 0.7),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=1.5))
    ax_red.text((em_line + obs_line)/2, 0.76, 'redshift', ha='center',
                fontsize=7, color='yellow')

    ax_red.set_xlim(380, 700)
    ax_red.set_ylim(0, 1)
    ax_red.set_xlabel('Wavelength (nm)', fontsize=8)
    ax_red.set_title('Spectroscopic Redshift', fontsize=9)
    ax_red.legend(fontsize=6)
    ax_red.set_yticks([])
    ax_red.text(0.5, 0.15, f'z = Δλ/λ = v/c\nv = {z*3e5:.0f} km/s',
                transform=ax_red.transAxes, ha='center', fontsize=8, color=COLORS['teal'])

    # ----- DISTANCE LADDER -----
    ax_ld.set_xlim(0, 10)
    ax_ld.set_ylim(0, 2)
    ax_ld.axis('off')
    ax_ld.set_title('The Cosmic Distance Ladder', fontsize=9)

    ladder_items = [
        (0.3, 'PARALLAX\n< 10 kpc\nGeometric baseline', COLORS['teal']),
        (2.5, 'CEPHEID VARIABLES\nup to 30 Mpc\nPeriod-luminosity', COLORS['accent']),
        (5.0, 'TYPE Ia SUPERNOVAE\nup to 1000 Mpc\nStandard candles', COLORS['amber']),
        (7.5, 'HUBBLE\'S LAW\nCosmological scales\nRequires H0', COLORS['red']),
    ]

    for i, (x, label, color) in enumerate(ladder_items):
        y_base = 0.3
        height = 0.9 - i * 0.15
        rect = plt.Rectangle((x - 0.9, y_base), 1.8, height,
                              facecolor=color, alpha=0.2, edgecolor=color, lw=1.5)
        ax_ld.add_patch(rect)
        ax_ld.text(x, y_base + height/2, label, ha='center', va='center',
                   fontsize=6.5, color=color, fontweight='bold')
        if i < 3:
            ax_ld.annotate('', xy=(ladder_items[i+1][0]-0.9, 0.7),
                           xytext=(x+0.9, 0.7),
                           arrowprops=dict(arrowstyle='->', color=COLORS['dim'], lw=1.5))
            ax_ld.text((x + ladder_items[i+1][0])/2, 0.85, 'calibrates',
                       ha='center', fontsize=6, color=COLORS['dim'], style='italic')

    plt.savefig('intermediate_hubble.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("Saved: intermediate_hubble.png")
    plt.show()


# ============================================================
# MODULE 3: ADVANCED — FRIEDMANN EQUATION INTEGRATOR
# ============================================================

def H_dimensionless(a, Om, Or, Ol):
    """
    Dimensionless Hubble parameter E(a) = H(a)/H0
    Includes matter, radiation, cosmological constant, curvature.
    """
    Ok = 1 - Om - Or - Ol
    val = Om * a**-3 + Or * a**-4 + Ol + Ok * a**-2
    return np.sqrt(np.maximum(val, 0))


def integrate_scale_factor(Om, Or, Ol, H0_kms_Mpc, a_start=1e-4, a_end=4.0, n_steps=4000):
    """
    Integrate dt/da = 1/(a * H(a)) to get t(a).
    Returns arrays of (a, t_Gyr).
    """
    H0 = H0_kms_Mpc * 1e3 / MPC_TO_KM  # s^-1
    a_arr = np.linspace(a_start, a_end, n_steps)
    da = a_arr[1] - a_arr[0]

    t = 0.0
    t_arr = np.zeros(n_steps)

    for i, a in enumerate(a_arr):
        E = H_dimensionless(a, Om, Or, Ol)
        if E == 0:
            break
        dtda = 1.0 / (a * H0 * E)
        t += dtda * da
        t_arr[i] = t / GYR_TO_S

    return a_arr, t_arr


def compute_H_of_z(z_arr, Om, Or, Ol, H0):
    """H(z) in km/s/Mpc"""
    a_arr = 1.0 / (1.0 + z_arr)
    E = H_dimensionless(a_arr, Om, Or, Ol)
    return H0 * E


def plot_advanced():
    """
    Four-panel advanced plot:
    1. Scale factor evolution for different cosmologies
    2. H(z) evolution
    3. Hubble tension visualization
    4. Growth factor (qualitative)
    """
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("HUBBLE'S LAW — ADVANCED MODULE\nFriedmann Framework and Observational Cosmology",
                 fontsize=11, color=COLORS['accent'])

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax_at = fig.add_subplot(gs[0, 0])
    ax_hz = fig.add_subplot(gs[0, 1])
    ax_ht = fig.add_subplot(gs[1, 0])
    ax_gf = fig.add_subplot(gs[1, 1])

    # ----- PANEL 1: a(t) for different cosmologies -----
    cosmologies = [
        {'Om': 0.31, 'Or': 9e-5, 'Ol': 0.69, 'H0': 70, 'label': 'LCDM (standard)', 'color': COLORS['accent']},
        {'Om': 1.0,  'Or': 9e-5, 'Ol': 0.0,  'H0': 70, 'label': 'Matter-only (Einstein-de Sitter)', 'color': COLORS['amber']},
        {'Om': 0.31, 'Or': 9e-5, 'Ol': 1.0,  'H0': 70, 'label': 'High dark energy (Om+OL > 1)', 'color': COLORS['red']},
        {'Om': 0.0,  'Or': 0.0,  'Ol': 1.0,  'H0': 70, 'label': 'de Sitter (vacuum dominated)', 'color': COLORS['teal']},
    ]

    for cosmo in cosmologies:
        a, t = integrate_scale_factor(cosmo['Om'], cosmo['Or'], cosmo['Ol'], cosmo['H0'])
        mask = (t > 0) & (a <= 4)
        ax_at.plot(t[mask], a[mask], color=cosmo['color'], lw=2, label=cosmo['label'])

    ax_at.axhline(1.0, color='white', lw=0.8, alpha=0.3, ls='--')
    ax_at.axvline(13.8, color=COLORS['amber'], lw=0.8, alpha=0.4, ls=':', label='Today (13.8 Gyr)')
    ax_at.set_xlabel('Cosmic Time (Gyr)')
    ax_at.set_ylabel('Scale Factor a(t)')
    ax_at.set_title('Scale Factor Evolution a(t)', fontsize=9)
    ax_at.legend(fontsize=6)
    ax_at.grid(True)
    ax_at.set_xlim(0, 30)
    ax_at.set_ylim(0, 4)
    ax_at.text(0.5, 0.92, 'a=1 today', transform=ax_at.transAxes,
               ha='center', fontsize=7, color='white', alpha=0.5)

    # ----- PANEL 2: H(z) evolution -----
    z = np.linspace(0, 5, 500)

    cosmo_hz = [
        {'Om': 0.31, 'Or': 9e-5, 'Ol': 0.69, 'H0': 70, 'label': 'LCDM', 'color': COLORS['accent']},
        {'Om': 0.31, 'Or': 9e-5, 'Ol': 0.0,  'H0': 70, 'label': 'No dark energy', 'color': COLORS['amber']},
        {'Om': 0.5,  'Or': 9e-5, 'Ol': 0.5,  'H0': 70, 'label': 'Om=0.5, OL=0.5', 'color': COLORS['teal']},
    ]

    for cosmo in cosmo_hz:
        Hz = compute_H_of_z(z, cosmo['Om'], cosmo['Or'], cosmo['Ol'], cosmo['H0'])
        ax_hz.plot(z, Hz, color=cosmo['color'], lw=2, label=cosmo['label'])

    ax_hz.axvline(0, color='white', lw=0.5, alpha=0.3)
    ax_hz.set_xlabel('Redshift z')
    ax_hz.set_ylabel('H(z)  [km/s/Mpc]')
    ax_hz.set_title('Hubble Parameter Evolution H(z)', fontsize=9)
    ax_hz.legend(fontsize=7)
    ax_hz.grid(True)
    ax_hz.set_yscale('log')
    ax_hz.text(0.02, 0.92, 'H(z) = H0 · E(z)\nE(z) = [Om(1+z)³ + Or(1+z)⁴ + OL]^½',
               transform=ax_hz.transAxes, fontsize=6.5, color=COLORS['dim'],
               verticalalignment='top')

    # ----- PANEL 3: Hubble Tension -----
    measurements = [
        ('CMB Planck 2018\n(Early universe)',   67.4, 0.5,  COLORS['teal']),
        ('BAO + BBN\n(Early universe)',          67.6, 1.1,  COLORS['teal']),
        ('SH0ES Cepheids+SNe\n(Late universe)',  73.0, 1.0,  COLORS['red']),
        ('CCHP TRGB\n(Late universe)',           69.8, 1.7,  COLORS['amber']),
        ('Gravitational Waves\n(Independent)',   70.0, 12.0, COLORS['accent']),
    ]

    y_pos = np.arange(len(measurements))

    for i, (label, h0, err, color) in enumerate(measurements):
        ax_ht.errorbar(h0, i, xerr=err, fmt='o', color=color,
                       capsize=4, capthick=1.5, elinewidth=1.5, markersize=7, zorder=5)
        ax_ht.text(64.5, i, label, va='center', fontsize=6.5, color=color)

    ax_ht.axvline(67.4, color=COLORS['teal'], lw=1, alpha=0.3, ls='--', label='Planck value')
    ax_ht.axvline(73.0, color=COLORS['red'], lw=1, alpha=0.3, ls='--', label='SH0ES value')
    ax_ht.axvspan(67.4, 73.0, alpha=0.05, color='white')
    ax_ht.set_xlabel('H0  (km/s/Mpc)')
    ax_ht.set_xlim(55, 90)
    ax_ht.set_yticks([])
    ax_ht.set_title('The Hubble Tension (4-5 sigma discrepancy)', fontsize=9)
    ax_ht.grid(True, axis='x')
    ax_ht.legend(fontsize=7, loc='upper right')
    ax_ht.text(0.5, 0.03, 'Tension: ~5 km/s/Mpc, probability < 0.001% if statistical',
               transform=ax_ht.transAxes, ha='center', fontsize=7, color=COLORS['red'],
               style='italic')

    # ----- PANEL 4: Growth Factor D+(a) -----
    a_gf = np.linspace(0.01, 1.5, 300)

    # Approximate growth factor: D+(a) computed via integral
    def growth_integrand(a, Om, Ol):
        Or = 9e-5
        Ok = 1 - Om - Or - Ol
        E = H_dimensionless(a, Om, Or, Ol)
        if E < 1e-10:
            return 0
        return 1.0 / (a * E)**3

    from scipy.integrate import quad

    def D_plus(a_val, Om, Ol):
        E = H_dimensionless(a_val, Om, 9e-5, Ol)
        integral, _ = quad(growth_integrand, 1e-5, a_val, args=(Om, Ol), limit=100)
        return E * integral

    gf_cosmologies = [
        {'Om': 0.31, 'Ol': 0.69, 'label': 'LCDM (GR)', 'color': COLORS['accent']},
        {'Om': 1.0,  'Ol': 0.0,  'label': 'EdS (D ~ a)', 'color': COLORS['amber']},
        {'Om': 0.1,  'Ol': 0.9,  'label': 'Dark energy dominated', 'color': COLORS['red']},
    ]

    for cosmo in gf_cosmologies:
        D = np.array([D_plus(a, cosmo['Om'], cosmo['Ol']) for a in a_gf])
        D_norm = D / D_plus(1.0, cosmo['Om'], cosmo['Ol'])
        ax_gf.plot(a_gf, D_norm, color=cosmo['color'], lw=2, label=cosmo['label'])

    # EdS reference
    ax_gf.plot(a_gf, a_gf, '--', color='white', lw=1, alpha=0.3, label='D ~ a (pure matter)')

    ax_gf.set_xlabel('Scale Factor a')
    ax_gf.set_ylabel('Growth Factor D+(a)  [normalized]')
    ax_gf.set_title('Linear Growth Factor D+(a)', fontsize=9)
    ax_gf.legend(fontsize=7)
    ax_gf.grid(True)
    ax_gf.set_xlim(0, 1.5)
    ax_gf.text(0.02, 0.92, 'f = d ln D+ / d ln a ≈ Ωm(a)^0.55\n(GR prediction)',
               transform=ax_gf.transAxes, fontsize=6.5, color=COLORS['dim'],
               verticalalignment='top')

    plt.savefig('advanced_hubble.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("Saved: advanced_hubble.png")
    plt.show()


# ============================================================
# INTERACTIVE SIMULATION: H0 Fitting Tool
# ============================================================

def interactive_hubble_fitter():
    """
    Interactive matplotlib widget for generating synthetic galaxy
    catalogs and fitting H0 via least squares.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(left=0.1, bottom=0.38, right=0.95, top=0.92)
    fig.suptitle("INTERACTIVE H0 FITTER", fontsize=10, color=COLORS['accent'])

    # Slider axes
    ax_h0    = plt.axes([0.15, 0.26, 0.70, 0.03], facecolor='#0d1228')
    ax_noise = plt.axes([0.15, 0.20, 0.70, 0.03], facecolor='#0d1228')
    ax_npts  = plt.axes([0.15, 0.14, 0.70, 0.03], facecolor='#0d1228')

    s_h0    = Slider(ax_h0,    'True H0',  50, 100, valinit=70,  color=COLORS['accent'])
    s_noise = Slider(ax_noise, 'Scatter %', 0,  50, valinit=15,  color=COLORS['teal'])
    s_npts  = Slider(ax_npts,  'N galaxies', 5, 100, valinit=30, color=COLORS['amber'], valstep=1)

    # Button axes
    ax_gen  = plt.axes([0.15, 0.05, 0.18, 0.06], facecolor='#0d1228')
    ax_fit  = plt.axes([0.38, 0.05, 0.18, 0.06], facecolor='#0d1228')
    ax_rst  = plt.axes([0.61, 0.05, 0.18, 0.06], facecolor='#0d1228')

    b_gen = Button(ax_gen, 'Generate', color='#111830', hovercolor='#1e2d55')
    b_fit = Button(ax_fit, 'Fit H0',   color='#111830', hovercolor='#1e2d55')
    b_rst = Button(ax_rst, 'Reset',    color='#111830', hovercolor='#1e2d55')

    state = {'distances': np.array([]), 'velocities': np.array([]), 'fitted': None}

    def draw():
        ax.clear()
        ax.set_facecolor('#080c1a')
        ax.set_xlabel('Distance (Mpc)')
        ax.set_ylabel('Recession Velocity (km/s)')
        ax.set_title('Hubble Diagram', fontsize=9)
        ax.grid(True, alpha=0.3)

        if len(state['distances']) > 0:
            ax.scatter(state['distances'], state['velocities'],
                       s=30, color=COLORS['amber'], alpha=0.8, zorder=5, label='Simulated galaxies')

            d_range = np.linspace(0, max(state['distances'])*1.05, 200)
            true_h0 = s_h0.val
            ax.plot(d_range, true_h0 * d_range, '--', color=COLORS['teal'],
                    lw=1.5, alpha=0.6, label=f'True H0 = {true_h0:.0f} km/s/Mpc')

            if state['fitted'] is not None:
                fh0 = state['fitted']
                ax.plot(d_range, fh0 * d_range, color=COLORS['accent'],
                        lw=2, label=f'Fitted H0 = {fh0:.1f} km/s/Mpc')
                err = abs(fh0 - true_h0) / true_h0 * 100
                ax.text(0.02, 0.95,
                        f'Fitted H0 = {fh0:.1f} km/s/Mpc\nError: {err:.1f}%',
                        transform=ax.transAxes, fontsize=9, color=COLORS['accent'],
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='#0d1228', alpha=0.8))

        ax.legend(fontsize=8, loc='lower right')
        fig.canvas.draw_idle()

    def generate(event):
        h0 = s_h0.val
        noise = s_noise.val / 100
        n = int(s_npts.val)
        np.random.seed(None)
        distances = np.random.uniform(10, 500, n)
        true_v = h0 * distances
        scatter = np.random.normal(0, noise * true_v)
        peculiar = np.random.normal(0, 400, n)
        velocities = true_v + scatter + peculiar
        state['distances'] = distances
        state['velocities'] = velocities
        state['fitted'] = None
        draw()

    def fit(event):
        if len(state['distances']) < 2:
            return
        popt, _ = curve_fit(hubble_model, state['distances'], state['velocities'])
        state['fitted'] = popt[0]
        draw()

    def reset(event):
        state['distances'] = np.array([])
        state['velocities'] = np.array([])
        state['fitted'] = None
        draw()

    b_gen.on_clicked(generate)
    b_fit.on_clicked(fit)
    b_rst.on_clicked(reset)
    generate(None)
    plt.show()


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("HUBBLE'S LAW EXPLORER — Python Simulation")
    print("Author: Nabin Pandey, Tri-Chandra Multiple Campus")
    print("=" * 60)
    print("\nChoose a module:")
    print("  1. Beginner     — Expanding universe visualization")
    print("  2. Intermediate — Hubble diagram and distance ladder")
    print("  3. Advanced     — Friedmann framework (GR derivation)")
    print("  4. Interactive  — H0 fitting tool (interactive sliders)")
    print("  5. Run all      — Generate all static plots")
    print()

    choice = input("Enter choice [1-5]: ").strip()

    if choice == '1':
        plot_beginner()
    elif choice == '2':
        plot_intermediate()
    elif choice == '3':
        plot_advanced()
    elif choice == '4':
        interactive_hubble_fitter()
    elif choice == '5':
        print("Generating all modules...")
        plot_beginner()
        plot_intermediate()
        plot_advanced()
    else:
        print("Invalid choice. Running interactive fitter by default.")
        interactive_hubble_fitter()
