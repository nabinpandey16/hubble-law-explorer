# Hubble's Law Explorer

An interactive, multi-level educational tool for learning Hubble's Law — from complete beginner to graduate-level cosmology. Built as a static web page (no server required) with a companion Python simulation suite.

Live site: [nabinpandey16.github.io/hubble-law-explorer](https://nabinpandey16.github.io/hubble-law-explorer)

---

## What is this?

Hubble's Law states that galaxies recede from us at velocities proportional to their distance:

```
v = H0 * d
```

This project presents that idea at three levels of depth, each with interactive simulations, visualizations, and concept checks.

---

## Modules

### Beginner — General Public
- The raisin-bread analogy for cosmic expansion
- Animated galaxy expansion simulation with adjustable H0
- Explanation of redshift as a cosmic speedometer
- Quick-check quiz

### Intermediate — Undergraduate (B.Sc. Physics)
- Formal statement with units (km/s/Mpc)
- Redshift formula: z = delta-lambda / lambda = v/c
- Interactive Hubble diagram builder with least-squares H0 fitting
- The cosmic distance ladder (parallax, Cepheids, Type Ia supernovae)
- Peculiar velocities and the Hubble flow limit (~10 Mpc)
- Hubble time estimate: t ~ 1/H0
- Multi-question concept quiz

### Advanced — Master's Level
Four sub-sections accessible via tabs:

1. **GR Derivation** — FLRW metric, Friedmann equations, density parameters (Omega_m, Omega_r, Omega_Lambda), equation of state
2. **Hubble Tension** — Measurement comparison table (Planck, SH0ES, CCHP, BAO+BBN, gravitational waves), proposed resolutions
3. **Scale Factor Integrator** — Numerical Runge-Kutta integration of da/dt = a * H(a); configurable Om, OL, H0; plots a(t) with cosmic age
4. **Perturbation Theory** — Linear growth factor D+(a), growth rate f ~ Omega_m^0.55, redshift-space distortions, Hubble horizon

---

## Python Simulation (hubble_simulation.py)

A standalone scientific Python script with four modes:

| Mode | Description |
|------|-------------|
| Beginner | Static plots: galaxy expansion arrows, basic Hubble diagram |
| Intermediate | Multi-panel: fitted Hubble diagram, redshift spectrum, distance ladder visualization |
| Advanced | Friedmann integrator for a(t), H(z) evolution, Hubble tension data, linear growth factor |
| Interactive | matplotlib widget with sliders for H0, scatter, N-galaxies and live curve fitting |

### Requirements

```bash
pip install numpy matplotlib scipy
```

### Usage

```bash
python hubble_simulation.py
```

Select a module (1-5) from the terminal prompt. All static plots are saved as PNG files.

---

## File Structure

```
hubble-law-explorer/
├── index.html              Main web application (self-contained)
├── hubble_simulation.py    Python simulation suite
└── README.md               This file
```

---

## Hosting on GitHub Pages

1. Push this repository to GitHub
2. Go to Settings > Pages
3. Set Source to "Deploy from a branch" and select `main` / `root`
4. Your site will be live at `https://yourusername.github.io/hubble-law-explorer`

No build step required. `index.html` is fully self-contained — all CSS, JavaScript, and fonts are inlined or loaded from Google Fonts CDN.

---

## Physics Notes

### Hubble Constant Value

The currently accepted range is 67-73 km/s/Mpc, depending on measurement method. The discrepancy between early-universe (CMB) and late-universe (distance ladder) measurements is known as the Hubble Tension and remains an open problem.

### Scale Factor Integration

The Python advanced module numerically integrates the Friedmann equation using a simple Euler scheme:

```
dt = da / (a * H0 * E(a))
E(a) = sqrt(Om * a^-3 + Or * a^-4 + OL + Ok * a^-2)
```

The age of the universe is obtained by evaluating t at a = 1.

### Linear Growth Factor

The growth factor D+(a) is computed via the integral:

```
D+(a) ~ H(a) * integral from 0 to a of [a' * H(a')]^-3 da'
```

The logarithmic growth rate f = d ln D+ / d ln a approximately equals Omega_m(a)^0.55 in general relativity (Linder 2005), a prediction that can be tested against galaxy survey data.

---

## Data Sources

- Planck Collaboration 2018 (arXiv:1807.06209)
- Riess et al. 2022 / SH0ES Team (arXiv:2112.04510)
- Freedman et al. 2021 / CCHP (arXiv:2106.15656)
- Addison et al. 2021 / BAO+BBN
- Abbott et al. / LIGO-Virgo (gravitational wave H0)

---

## Author

**Nabin Pandey**  
B.Sc. Physics, Third Year  
Tri-Chandra Multiple Campus, Tribhuvan University, Kathmandu  
APS Student Ambassador 2024-2025  
Member, QNepal | ANPA | American Physical Society

GitHub: [github.com/nabinpandey16](https://github.com/nabinpandey16)  
LinkedIn: [np.linkedin.com/in/nabinpandey1661](https://np.linkedin.com/in/nabinpandey1661)

---

## License

MIT License. Free to use, modify, and distribute with attribution.
