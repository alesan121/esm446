#!/usr/bin/env python3
"""
range_estimator.py — Log-Distance path loss → círculos de probabilidad CoT

Uso: python3 range_estimator.py <canal> <freq_hz> <power_linear> <rx_lat> <rx_lon>
Salida: uno o varios CoT XML por stdout, separados por \n---\n
"""

import sys
import math
from datetime import datetime, timezone, timedelta

# ── Constantes de propagación ──────────────────────────────────────────────
SPEED_OF_LIGHT = 3e8
D0_METERS      = 1.0

# Modelo Log-Distance para PMR446 en entorno urbano/forestal mixto
# n y sigma calibrados para 446 MHz banda UHF
PATH_LOSS_EXPONENT = 3.5   # urbano-NLOS / forestal (Rappaport Table 3.2)
SIGMA_DB           = 10.0  # desviación shadowing típica en este entorno

# EIRP catalogado PMR446 EU — límite legal ETSI EN 300 296: 500mW = 27 dBm
# Con antena dipolo integrada ~2 dBi → EIRP ≈ 29 dBm
EIRP_DBM   = 29.0
GAIN_RX    = 0.0    # antena HackRF omnidireccional, sin ganancia extra

# Anillos de probabilidad (factor k sobre sigma)
RINGS = {"68pct": 1.0, "90pct": 1.645, "95pct": 2.0}

# ── Conversión de unidades ──────────────────────────────────────────────────
def linear_to_dbm(power_linear: float) -> float:
    """
    Convierte potencia IQ normalizada del channelizer a dBm estimados.

    El channelizer produce np.mean(|IQ|²) con muestras CF32 en [-1,1].
    Esta escala es relativa al fondo de escala del ADC del HackRF,
    NO en mW. El offset es un punto de calibración empírico.

    CALIBRACIÓN OBLIGATORIA:
        1. Coloca un PMR446 a distancia conocida D_cal (ej: 100m)
        2. Anota power_linear que reporta el channelizer
        3. Calcula pr_real con Friis: pr = EIRP - PL(D_cal)
        4. offset_cal = pr_real - 10*log10(power_linear)

    Valor provisional: -49.3 (señal media a ~50-100m con VGA=20)
    Este valor produce distancias razonables pero NO calibradas.
    """
    if power_linear <= 0:
        return -120.0
    # CALIBRAR ESTE OFFSET con señal de potencia conocida
    OFFSET_CAL = -49.3
    return 10 * math.log10(power_linear) + OFFSET_CAL
def fspl_at_d0(freq_hz: float) -> float:
    """FSPL a d0=1m — la 'resistencia base del sustrato vacío'."""
    return 20 * math.log10((4 * math.pi * D0_METERS * freq_hz) / SPEED_OF_LIGHT)

# ── Estimación de distancia ─────────────────────────────────────────────────
def estimate_range(freq_hz: float, power_linear: float) -> tuple[float, float]:
    """
    Retorna (distancia_media_m, sigma_m).
    Analogía: medir longitud de cable por su resistencia, con tolerancia ±σ.
    """
    pr_dbm       = linear_to_dbm(power_linear)
    pl_observed  = EIRP_DBM + GAIN_RX - pr_dbm          # pérdida real observada
    pl_d0        = fspl_at_d0(freq_hz)                   # pérdida a 1m
    delta        = pl_observed - pl_d0                   # lo que corresponde a la distancia

    d_est    = D0_METERS * (10 ** (delta / (10 * PATH_LOSS_EXPONENT)))
    # Propagación de error: ∂d/∂PL · σ_PL
    sigma_m  = d_est * (SIGMA_DB / (10 * PATH_LOSS_EXPONENT)) * math.log(10)

    return d_est, sigma_m

# ── Generador CoT XML ────────────────────────────────────────────────────────
def _iso(offset_s: int = 0) -> str:
    t = datetime.now(timezone.utc) + timedelta(seconds=offset_s)
    return t.strftime("%Y-%m-%dT%H:%M:%S.00Z")

def build_cot_circles(
    ch_num: str,
    lat: float,
    lon: float,
    d_est: float,
    sigma_m: float,
    status: str,
) -> list[str]:
    """
    Genera un CoT 'u-d-r' (drawing circle) por cada anillo de probabilidad.
    ITAK renderiza cada mensaje como un círculo independiente sobre el mapa.
    El campo 'ce' en <point> es el radio del círculo en metros.
    """
    messages = []
    color_map = {"68pct": "red", "90pct": "orange", "95pct": "yellow"}

    for label, k in RINGS.items():
        radius_m = d_est + k * sigma_m
        uid      = f"EW-PMR{ch_num}-{label}"
        remarks  = (
            f"[EW] PMR{ch_num} | {status} | "
            f"d̂={d_est:.0f}m ±{sigma_m:.0f}m | "
            f"P({label[:-3]}%)={radius_m:.0f}m"
        )
        xml = f"""<event version="2.0"
    uid="{uid}"
    type="u-d-r"
    time="{_iso()}"
    start="{_iso()}"
    stale="{_iso(300)}"
    how="m-g">
  <point lat="{lat:.7f}"
         lon="{lon:.7f}"
         hae="0"
         ce="{radius_m:.1f}"
         le="9999999.0"/>
  <detail>
    <remarks>{remarks}</remarks>
    <color argb="-1"/>
    <shape>
      <ellipse semiMajor="{radius_m:.1f}"
               semiMinor="{radius_m:.1f}"
               angle="0"/>
    </shape>
  </detail>
</event>"""
        messages.append(xml)

    return messages

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 7:
        print("uso: range_estimator.py <ch> <freq_hz> <power> <rx_lat> <rx_lon> <status>")
        sys.exit(1)

    ch_num        = sys.argv[1]
    freq_hz       = float(sys.argv[2])
    power_linear  = float(sys.argv[3])
    rx_lat        = float(sys.argv[4])
    rx_lon        = float(sys.argv[5])
    status        = sys.argv[6]

    d_est, sigma = estimate_range(freq_hz, power_linear)

    cot_list = build_cot_circles(ch_num, rx_lat, rx_lon, d_est, sigma, status)

    for cot in cot_list:
        print(cot)
        print("---")

    # Stderr para logging (el bash lo puede redirigir)
    print(
        f"[RANGE] d̂={d_est:.0f}m σ={sigma:.0f}m | "
        f"95%={d_est + 2*sigma:.0f}m",
        file=sys.stderr,
        flush=True,
    )

if __name__ == "__main__":
    main()
