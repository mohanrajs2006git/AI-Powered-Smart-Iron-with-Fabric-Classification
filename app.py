"""
Smart Iron — Fabric Classifier Web Server
==========================================
Flask backend with pre-loaded ML model + Arduino serial reader.

HOW TO RUN:
    python app.py --port COM3          (Windows)
    python app.py --port /dev/ttyUSB0  (Linux/Mac)
    python app.py                      (no Arduino, manual mode only)

Visit: http://localhost:5000
"""

import os
import json
import time
import threading
import argparse

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════
# CLI ARGUMENTS
# ══════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="Smart Iron Web Server")
parser.add_argument("--port", default=None,
                    help="Arduino serial port e.g. COM3 or /dev/ttyUSB0")
parser.add_argument("--baud", type=int, default=115200,
                    help="Baud rate (default 115200)")
args, _ = parser.parse_known_args()

# ══════════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════════
app = Flask(__name__)

# ══════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smart_iron_model.pkl")

print(f"[STARTUP] Loading model from {MODEL_PATH} ...")
try:
    bundle     = joblib.load(MODEL_PATH)
    MODEL      = bundle["model"]
    LE         = bundle["label_encoder"]
    MODEL_NAME = bundle["model_name"]
    print(f"[STARTUP] ✔ Model loaded: {MODEL_NAME}")
    print(f"[STARTUP] ✔ Classes    : {list(LE.classes_)}")
except Exception as e:
    print(f"[STARTUP] ✗ Failed to load model: {e}")
    raise SystemExit(1)

# ══════════════════════════════════════════════════════════════════
# SENSOR STATE  (shared between serial thread and Flask routes)
# ══════════════════════════════════════════════════════════════════
sensor_lock  = threading.Lock()
sensor_state = {
    "temperature_c":    None,
    "motion_variation": None,
    "static_time_s":    None,
    "timestamp":        None,
    "connected":        False,
    "port":             args.port if args.port else "not configured",
}

# ══════════════════════════════════════════════════════════════════
# SERIAL READER THREAD
# ══════════════════════════════════════════════════════════════════
def serial_reader(port, baud):
    """
    Runs in a background daemon thread.
    Reads lines from Arduino over serial.
    Expected format:  SENSOR,<temp_c>,<motion_var>,<static_s>
    Example:          SENSOR,185.40,0.0312,5
    """
    try:
        import serial
    except ImportError:
        print("[SERIAL] ✗ pyserial not installed — run: pip install pyserial")
        return

    while True:   # outer loop: auto-reconnect on disconnect
        try:
            print(f"[SERIAL] Connecting to {port} @ {baud} baud …")
            with serial.Serial(port, baud, timeout=2) as ser:
                with sensor_lock:
                    sensor_state["connected"] = True
                print(f"[SERIAL] ✔ Connected to {port}")

                while True:   # inner loop: read lines
                    raw = ser.readline().decode("utf-8", errors="ignore").strip()

                    if not raw.startswith("SENSOR,"):
                        continue          # skip debug / junk lines

                    parts = raw.split(",")
                    if len(parts) != 4:
                        continue          # malformed line

                    try:
                        T = float(parts[1])
                        M = float(parts[2])
                        S = int(parts[3])
                        with sensor_lock:
                            sensor_state.update({
                                "temperature_c":    T,
                                "motion_variation": M,
                                "static_time_s":    S,
                                "timestamp":        time.time(),
                            })
                    except ValueError:
                        pass              # bad number, skip

        except Exception as e:
            with sensor_lock:
                sensor_state["connected"] = False
            print(f"[SERIAL] ✗ Error: {e}")
            print("[SERIAL]   Retrying in 5 s …")
            time.sleep(5)


# Start serial thread ONLY if a port was given
if args.port:
    serial_thread = threading.Thread(
        target=serial_reader,
        args=(args.port, args.baud),
        daemon=True          # dies automatically when main process exits
    )
    serial_thread.start()
    print(f"[SERIAL] Thread started for port {args.port}")
else:
    print("[SERIAL] No --port given. Running in manual-only mode.")

# ══════════════════════════════════════════════════════════════════
# FABRIC METADATA
# ══════════════════════════════════════════════════════════════════
FABRIC_INFO = {
    "Cotton": {
        "emoji":      "🧺",
        "temp_range": "175 – 204 °C",
        "tip":        "Use high heat with steady strokes. Cotton is durable and handles steam well.",
        "color":      "#FF6B35",
        "glow":       "rgba(255,107,53,0.4)",
    },
    "Silk": {
        "emoji":      "🌸",
        "temp_range": "110 – 148 °C",
        "tip":        "Use low heat, gentle gliding motion. Iron inside-out and avoid direct steam.",
        "color":      "#C39BD3",
        "glow":       "rgba(195,155,211,0.4)",
    },
    "Wool": {
        "emoji":      "🐑",
        "temp_range": "120 – 155 °C",
        "tip":        "Use a damp pressing cloth. Medium heat with lifting motion, never dragging.",
        "color":      "#2ECC71",
        "glow":       "rgba(46,204,113,0.4)",
    },
    "Polyester": {
        "emoji":      "🧬",
        "temp_range": "110 – 150 °C",
        "tip":        "Low heat only. Iron on reverse side to prevent shiny marks. No steam.",
        "color":      "#3498DB",
        "glow":       "rgba(52,152,219,0.4)",
    },
    "Anomaly": {
        "emoji":      "⚠️",
        "temp_range": "< 100°C  or  > 210°C",
        "tip":        "DANGER: Iron settings are outside safe range! Adjust temperature immediately.",
        "color":      "#E74C3C",
        "glow":       "rgba(231,76,60,0.5)",
    },
}

# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (must match training-time features exactly)
# ══════════════════════════════════════════════════════════════════
def engineer_features(T: float, M: float, S: int) -> pd.DataFrame:
    eps = 1e-9
    row = {
        "Temperature_C":       T,
        "Motion_Variation":    M,
        "Static_Time_s":       S,
        "Temp_Motion_Ratio":   T / (M + eps),
        "Temp_Static_Product": T * S,
        "Motion_Static_Ratio": M / (S + eps),
        "Temp_Motion_Product": T * M,
        "Motion_log":          np.log1p(M),
        "Static_log":          np.log1p(S),
        "Temp_squared":        T ** 2,
        "Motion_squared":      M ** 2,
        "Heat_Exposure_Index": T * S / (M + eps),
    }
    return pd.DataFrame([row])


def run_prediction(T: float, M: float, S: int) -> dict:
    X        = engineer_features(T, M, S)
    pred_idx = MODEL.predict(X)[0]
    proba    = MODEL.predict_proba(X)[0]
    label    = LE.inverse_transform([pred_idx])[0]
    conf     = float(proba.max()) * 100
    all_prob = {cls: round(float(p) * 100, 2)
                for cls, p in zip(LE.classes_, proba)}
    info = FABRIC_INFO.get(label, {})
    return {
        "fabric":            label,
        "confidence":        round(conf, 2),
        "all_probabilities": all_prob,
        "emoji":             info.get("emoji",      "❓"),
        "temp_range":        info.get("temp_range", "—"),
        "tip":               info.get("tip",        ""),
        "color":             info.get("color",      "#fff"),
        "glow":              info.get("glow",       "rgba(0,212,255,0.4)"),
    }

# ══════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ══════════════════════════════════════════════════════════════════
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Smart Iron · Fabric Intelligence</title>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800;900&family=Exo+2:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
  :root{--bg:#040810;--panel:#080f1e;--border:#0d2040;--accent:#00d4ff;--accent2:#ff6b35;--text:#c8d8f0;--dim:#4a6080;--success:#00ff9d;--danger:#ff3d5a;--warn:#ffb020;}
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--bg);color:var(--text);font-family:'Exo 2',sans-serif;min-height:100vh;overflow-x:hidden;}
  #bg-canvas{position:fixed;inset:0;z-index:0;pointer-events:none;}
  body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,212,255,.03)1px,transparent 1px),linear-gradient(90deg,rgba(0,212,255,.03)1px,transparent 1px);background-size:60px 60px;pointer-events:none;z-index:0;}
  .wrapper{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:0 24px 80px;}

  /* Header */
  header{text-align:center;padding:60px 0 40px;}
  .logo-ring{width:110px;height:110px;margin:0 auto 24px;position:relative;display:flex;align-items:center;justify-content:center;}
  .logo-ring svg{position:absolute;inset:0;animation:spin 8s linear infinite;}
  .logo-ring svg.slow{animation-duration:14s;animation-direction:reverse;}
  .iron-icon{font-size:44px;z-index:1;filter:drop-shadow(0 0 18px var(--accent));animation:pulse-glow 2.5s ease-in-out infinite;}
  @keyframes spin{to{transform:rotate(360deg);}}
  @keyframes pulse-glow{0%,100%{filter:drop-shadow(0 0 18px var(--accent));}50%{filter:drop-shadow(0 0 36px var(--accent)) drop-shadow(0 0 60px var(--accent));}}
  h1{font-family:'Orbitron',sans-serif;font-size:clamp(1.8rem,4vw,3rem);font-weight:900;letter-spacing:.12em;background:linear-gradient(135deg,var(--accent) 0%,#fff 50%,var(--accent2) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;text-transform:uppercase;}
  .subtitle{margin-top:10px;font-size:1rem;color:var(--dim);letter-spacing:.25em;text-transform:uppercase;font-weight:300;}
  .model-badge{display:inline-block;margin-top:16px;padding:4px 14px;border:1px solid var(--accent);border-radius:20px;font-size:.75rem;color:var(--accent);letter-spacing:.15em;font-family:'Orbitron',sans-serif;background:rgba(0,212,255,.07);}
  .status-row{display:flex;align-items:center;justify-content:center;gap:24px;margin-top:12px;flex-wrap:wrap;}
  .status-dot{display:flex;align-items:center;gap:6px;font-size:.72rem;color:var(--dim);}
  .dot{width:7px;height:7px;border-radius:50%;background:var(--success);box-shadow:0 0 8px var(--success);animation:blink 1.8s ease infinite;}
  .dot.warn{background:var(--warn);box-shadow:0 0 8px var(--warn);}
  .dot.red{background:var(--danger);box-shadow:0 0 8px var(--danger);}
  .dot.blue{background:var(--accent);box-shadow:0 0 8px var(--accent);}
  @keyframes blink{0%,100%{opacity:1;}50%{opacity:.3;}}

  /* Cards */
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin-top:28px;}
  @media(max-width:700px){.grid{grid-template-columns:1fr;}}
  .card{background:var(--panel);border:1px solid var(--border);border-radius:20px;padding:32px;position:relative;overflow:hidden;transition:border-color .3s,box-shadow .3s;}
  .card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--accent),transparent);opacity:0;transition:opacity .3s;}
  .card:hover::before{opacity:1;}
  .card:hover{border-color:rgba(0,212,255,.3);box-shadow:0 0 40px rgba(0,212,255,.08);}
  .card-title{font-family:'Orbitron',sans-serif;font-size:.7rem;letter-spacing:.3em;text-transform:uppercase;color:var(--accent);margin-bottom:24px;display:flex;align-items:center;gap:10px;}
  .card-title::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--accent),transparent);opacity:.3;}

  /* Live Sensor */
  .sensor-section{margin-top:40px;background:var(--panel);border:1px solid rgba(0,255,157,.2);border-radius:20px;padding:28px 32px;position:relative;overflow:hidden;}
  .sensor-section::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--success),transparent);opacity:.7;}
  .sensor-header{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:14px;margin-bottom:24px;}
  .sensor-title{font-family:'Orbitron',sans-serif;font-size:.7rem;letter-spacing:.3em;text-transform:uppercase;color:var(--success);display:flex;align-items:center;gap:10px;}
  .sensor-title::after{content:'';width:60px;height:1px;background:linear-gradient(90deg,var(--success),transparent);opacity:.4;}
  .conn-badge{display:flex;align-items:center;gap:8px;padding:5px 16px;border-radius:20px;border:1px solid rgba(0,255,157,.2);background:rgba(0,255,157,.04);font-family:'Orbitron',sans-serif;font-size:.68rem;letter-spacing:.12em;color:var(--dim);}
  .sensor-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;}
  @media(max-width:600px){.sensor-metrics{grid-template-columns:1fr;}}
  .metric-box{background:rgba(0,255,157,.03);border:1px solid rgba(0,255,157,.12);border-radius:14px;padding:22px 16px;text-align:center;position:relative;overflow:hidden;transition:border-color .4s,box-shadow .4s;}
  .metric-box.flash{animation:mflash .6s ease;}
  @keyframes mflash{0%{border-color:var(--success);box-shadow:0 0 22px rgba(0,255,157,.35);}100%{border-color:rgba(0,255,157,.12);box-shadow:none;}}
  .metric-icon{font-size:20px;display:block;margin-bottom:8px;}
  .metric-label{font-size:.68rem;letter-spacing:.18em;text-transform:uppercase;color:var(--dim);margin-bottom:10px;}
  .metric-value{font-family:'Orbitron',sans-serif;font-size:2rem;font-weight:800;color:var(--success);line-height:1;transition:color .3s;}
  .metric-value.no-data{color:var(--dim);font-size:1.4rem;}
  .metric-unit{font-size:.7rem;color:var(--dim);margin-top:6px;letter-spacing:.1em;}
  .metric-bar{height:3px;border-radius:2px;background:rgba(255,255,255,.05);margin-top:14px;overflow:hidden;}
  .metric-bar-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--success),#00d4ff);width:0;transition:width .9s cubic-bezier(.16,1,.3,1);}
  .sensor-bottom{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:14px;margin-top:22px;padding-top:18px;border-top:1px solid var(--border);}
  .sensor-age-label{font-size:.75rem;color:var(--dim);}
  .sensor-age-label span{color:var(--success);font-weight:600;}
  .auto-row{display:flex;align-items:center;gap:10px;font-size:.78rem;color:var(--dim);}
  .toggle{position:relative;width:38px;height:20px;flex-shrink:0;}
  .toggle input{opacity:0;width:0;height:0;}
  .toggle-track{position:absolute;inset:0;border-radius:10px;background:var(--border);cursor:pointer;transition:background .2s;}
  .toggle-track::after{content:'';position:absolute;width:14px;height:14px;border-radius:50%;top:3px;left:3px;background:var(--dim);transition:transform .2s,background .2s;}
  .toggle input:checked~.toggle-track{background:rgba(0,255,157,.2);}
  .toggle input:checked~.toggle-track::after{transform:translateX(18px);background:var(--success);}
  .btn-sensor{padding:11px 22px;border:1px solid rgba(0,255,157,.4);border-radius:12px;background:rgba(0,255,157,.07);color:var(--success);font-family:'Orbitron',sans-serif;font-size:.7rem;letter-spacing:.2em;text-transform:uppercase;cursor:pointer;font-weight:700;transition:all .2s;}
  .btn-sensor:hover:not(:disabled){background:rgba(0,255,157,.15);box-shadow:0 6px 20px rgba(0,255,157,.2);transform:translateY(-1px);}
  .btn-sensor:disabled{opacity:.3;cursor:not-allowed;transform:none;}

  /* Manual Input */
  .input-group{margin-bottom:24px;}
  .input-group label{display:block;font-size:.78rem;letter-spacing:.15em;text-transform:uppercase;color:var(--dim);margin-bottom:8px;font-weight:500;}
  .input-row{display:flex;align-items:center;gap:12px;}
  input[type=range]{flex:1;-webkit-appearance:none;height:4px;border-radius:2px;background:var(--border);outline:none;cursor:pointer;}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:18px;height:18px;border-radius:50%;background:var(--accent);box-shadow:0 0 12px var(--accent);cursor:pointer;transition:transform .15s;}
  input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.3);}
  input[type=range]::-webkit-slider-runnable-track{background:linear-gradient(90deg,var(--accent) var(--pct,0%),var(--border) var(--pct,0%));height:4px;border-radius:2px;}
  .val-display{font-family:'Orbitron',sans-serif;font-size:1.05rem;color:#fff;min-width:90px;text-align:right;font-weight:600;}
  .range-chips{display:flex;gap:8px;margin-top:8px;flex-wrap:wrap;}
  .chip{font-size:.7rem;padding:3px 10px;border-radius:12px;border:1px solid var(--border);color:var(--dim);cursor:pointer;transition:all .2s;letter-spacing:.08em;}
  .chip:hover{border-color:var(--accent);color:var(--accent);background:rgba(0,212,255,.08);}
  .btn-predict{width:100%;padding:18px;margin-top:8px;border:none;border-radius:14px;background:linear-gradient(135deg,#0060ff,#00d4ff);color:#fff;font-family:'Orbitron',sans-serif;font-size:.9rem;letter-spacing:.25em;text-transform:uppercase;cursor:pointer;position:relative;overflow:hidden;transition:transform .2s,box-shadow .2s;font-weight:700;}
  .btn-predict:hover{transform:translateY(-2px);box-shadow:0 12px 40px rgba(0,212,255,.35);}
  .btn-predict.loading{pointer-events:none;}
  .btn-predict .btn-text{transition:opacity .2s;}
  .btn-predict .spinner{display:none;width:20px;height:20px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);}
  .btn-predict.loading .btn-text{opacity:0;}
  .btn-predict.loading .spinner{display:block;}

  /* Result */
  #result-panel{grid-column:1/-1;display:none;animation:fadeUp .5s ease both;}
  #result-panel.show{display:block;}
  @keyframes fadeUp{from{opacity:0;transform:translateY(24px);}to{opacity:1;transform:translateY(0);}}
  .result-inner{display:grid;grid-template-columns:auto 1fr;gap:32px;align-items:center;}
  @media(max-width:600px){.result-inner{grid-template-columns:1fr;text-align:center;}}
  .result-icon-wrap{width:120px;height:120px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:52px;position:relative;flex-shrink:0;transition:box-shadow .4s;}
  .result-icon-wrap::after{content:'';position:absolute;inset:-4px;border-radius:50%;border:2px solid currentColor;opacity:.4;animation:orbit 3s linear infinite;}
  @keyframes orbit{to{transform:rotate(360deg);}}
  .fabric-name{font-family:'Orbitron',sans-serif;font-size:clamp(1.8rem,4vw,2.8rem);font-weight:900;letter-spacing:.08em;line-height:1;}
  .fabric-range{font-size:.85rem;color:var(--dim);margin-top:6px;letter-spacing:.1em;}
  .fabric-tip{margin-top:14px;font-size:.92rem;line-height:1.6;max-width:500px;color:var(--text);}
  .src-badge{display:inline-block;margin-top:10px;padding:4px 12px;border-radius:12px;font-size:.7rem;letter-spacing:.1em;font-weight:600;}
  .conf-row{margin-top:20px;}
  .conf-label{font-size:.72rem;letter-spacing:.2em;color:var(--dim);text-transform:uppercase;margin-bottom:6px;}
  .conf-track{height:8px;border-radius:4px;background:var(--border);overflow:hidden;}
  .conf-fill{height:100%;border-radius:4px;width:0;transition:width 1.2s cubic-bezier(.16,1,.3,1);}
  .conf-pct{font-family:'Orbitron',sans-serif;font-size:1.6rem;font-weight:800;margin-top:6px;}
  .prob-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-top:28px;}
  @media(max-width:600px){.prob-grid{grid-template-columns:repeat(3,1fr);}}
  .prob-item{background:rgba(255,255,255,.03);border:1px solid var(--border);border-radius:12px;padding:12px 8px;text-align:center;}
  .prob-item.top{border-color:rgba(0,212,255,.5);background:rgba(0,212,255,.06);}
  .prob-item .p-label{font-size:.7rem;color:var(--dim);letter-spacing:.1em;}
  .prob-item .p-val{font-family:'Orbitron',sans-serif;font-size:1rem;font-weight:700;margin-top:4px;}
  .prob-item .p-bar{height:3px;border-radius:2px;background:var(--border);margin-top:8px;overflow:hidden;}
  .prob-item .p-bar-fill{height:100%;border-radius:2px;width:0;transition:width 1s .3s ease;}

  /* Fabric Quick Presets */
  .ref-section{margin-top:48px;}
  .ref-title{font-family:'Orbitron',sans-serif;font-size:.65rem;letter-spacing:.4em;color:var(--dim);text-align:center;text-transform:uppercase;margin-bottom:20px;}
  .ref-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;}
  @media(max-width:700px){.ref-grid{grid-template-columns:repeat(3,1fr);}}
  @media(max-width:420px){.ref-grid{grid-template-columns:repeat(2,1fr);}}
  .ref-card{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:16px 12px;text-align:center;cursor:pointer;transition:transform .2s,border-color .2s,box-shadow .2s;}
  .ref-card:hover{transform:translateY(-4px);}
  .ref-card .r-emoji{font-size:26px;display:block;margin-bottom:8px;}
  .ref-card .r-name{font-family:'Orbitron',sans-serif;font-size:.62rem;letter-spacing:.12em;color:var(--text);}
  .ref-card .r-temp{font-size:.62rem;color:var(--dim);margin-top:4px;}

  .scan-line{position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,var(--accent),transparent);top:-2px;animation:scan 2.5s linear infinite;opacity:.5;}
  @keyframes scan{to{top:100%;}}
  .reveal{opacity:0;transform:translateY(30px);transition:opacity .7s ease,transform .7s ease;}
  .reveal.visible{opacity:1;transform:translateY(0);}
</style>
</head>
<body>
<canvas id="bg-canvas"></canvas>
<div class="wrapper">

  <!-- HEADER -->
  <header>
    <div class="logo-ring">
      <svg viewBox="0 0 110 110" fill="none">
        <circle cx="55" cy="55" r="52" stroke="rgba(0,212,255,0.15)" stroke-width="1"/>
        <circle cx="55" cy="55" r="52" stroke="url(#g1)" stroke-width="1.5" stroke-dasharray="40 280" stroke-linecap="round"/>
        <defs><linearGradient id="g1" x1="0" y1="0" x2="110" y2="110" gradientUnits="userSpaceOnUse"><stop offset="0%" stop-color="#00d4ff"/><stop offset="100%" stop-color="transparent"/></linearGradient></defs>
      </svg>
      <svg class="slow" viewBox="0 0 110 110" fill="none">
        <circle cx="55" cy="55" r="44" stroke="rgba(255,107,53,0.15)" stroke-width="1"/>
        <circle cx="55" cy="55" r="44" stroke="url(#g2)" stroke-width="1" stroke-dasharray="20 240" stroke-linecap="round"/>
        <defs><linearGradient id="g2" x1="0" y1="0" x2="110" y2="110" gradientUnits="userSpaceOnUse"><stop offset="0%" stop-color="#ff6b35"/><stop offset="100%" stop-color="transparent"/></linearGradient></defs>
      </svg>
      <span class="iron-icon">🔬</span>
    </div>
    <h1>Smart Iron</h1>
    <p class="subtitle">Fabric Intelligence System</p>
    <div class="model-badge">⚡ {{ model_name }} · Active</div>
    <div class="status-row" style="margin-top:18px">
      <div class="status-dot"><div class="dot"></div> Model Loaded</div>
      <div class="status-dot">
        <div class="dot warn" id="hdr-sensor-dot"></div>
        <span id="hdr-sensor-label">Sensor: Checking…</span>
      </div>
      <div class="status-dot"><div class="dot blue"></div> API Ready</div>
    </div>
  </header>

  <!-- LIVE SENSOR SECTION -->
  <div class="sensor-section reveal">
    <div class="sensor-header">
      <div class="sensor-title">📡 Live Sensor Data</div>
      <div class="conn-badge">
        <div class="dot warn" id="conn-dot"></div>
        <span id="conn-label">{{ sensor_port }}</span>
      </div>
    </div>

    <div class="sensor-metrics">
      <div class="metric-box" id="mb-temp">
        <span class="metric-icon">🌡️</span>
        <div class="metric-label">Temperature</div>
        <div class="metric-value no-data" id="mv-temp">—</div>
        <div class="metric-unit">°C</div>
        <div class="metric-bar"><div class="metric-bar-fill" id="mbar-temp"></div></div>
      </div>
      <div class="metric-box" id="mb-motion">
        <span class="metric-icon">📳</span>
        <div class="metric-label">Motion Variation</div>
        <div class="metric-value no-data" id="mv-motion">—</div>
        <div class="metric-unit">IMU variance</div>
        <div class="metric-bar"><div class="metric-bar-fill" id="mbar-motion"></div></div>
      </div>
      <div class="metric-box" id="mb-static">
        <span class="metric-icon">⏱️</span>
        <div class="metric-label">Static Time</div>
        <div class="metric-value no-data" id="mv-static">—</div>
        <div class="metric-unit">seconds</div>
        <div class="metric-bar"><div class="metric-bar-fill" id="mbar-static"></div></div>
      </div>
    </div>

    <div class="sensor-bottom">
      <div class="sensor-age-label">Last update: <span id="sensor-age-text">waiting for signal…</span></div>
      <div class="auto-row">
        <label class="toggle">
          <input type="checkbox" id="auto-chk"/>
          <span class="toggle-track"></span>
        </label>
        Auto-predict every 5 s
      </div>
      <button class="btn-sensor" id="btn-sensor" onclick="predictFromSensor()" disabled>
        ⚡ Predict from Sensor
      </button>
    </div>
  </div>

  <!-- MAIN GRID -->
  <div class="grid">

    <!-- Manual Input -->
    <div class="card reveal" style="margin-top:0">
      <div class="scan-line"></div>
      <div class="card-title">⚙ Manual Input Parameters</div>

      <div class="input-group">
        <label>🌡 Temperature (°C)</label>
        <div class="input-row">
          <input type="range" id="temp" min="60" max="250" value="180" step="0.5"
                 oninput="updateSlider(this,'temp-val','°C')"/>
          <div class="val-display" id="temp-val">180.0 °C</div>
        </div>
        <div class="range-chips">
          <span class="chip" onclick="setVal('temp',128,'temp-val','°C')">Silk 128°</span>
          <span class="chip" onclick="setVal('temp',136,'temp-val','°C')">Wool 136°</span>
          <span class="chip" onclick="setVal('temp',130,'temp-val','°C')">Poly 130°</span>
          <span class="chip" onclick="setVal('temp',192,'temp-val','°C')">Cotton 192°</span>
          <span class="chip" onclick="setVal('temp',235,'temp-val','°C')">⚠ High 235°</span>
        </div>
      </div>

      <div class="input-group">
        <label>📡 Motion Variation (IMU)</label>
        <div class="input-row">
          <input type="range" id="motion" min="0.001" max="0.05" value="0.030" step="0.001"
                 oninput="updateSlider(this,'motion-val','')"/>
          <div class="val-display" id="motion-val">0.0300</div>
        </div>
        <div class="range-chips">
          <span class="chip" onclick="setVal('motion',0.009,'motion-val','',4)">Silk ~0.009</span>
          <span class="chip" onclick="setVal('motion',0.019,'motion-val','',4)">Poly ~0.019</span>
          <span class="chip" onclick="setVal('motion',0.031,'motion-val','',4)">Wool ~0.031</span>
          <span class="chip" onclick="setVal('motion',0.042,'motion-val','',4)">Cotton ~0.042</span>
        </div>
      </div>

      <div class="input-group">
        <label>⏱ Static Time (seconds)</label>
        <div class="input-row">
          <input type="range" id="static" min="1" max="45" value="5" step="1"
                 oninput="updateSlider(this,'static-val','s')"/>
          <div class="val-display" id="static-val">5 s</div>
        </div>
        <div class="range-chips">
          <span class="chip" onclick="setVal('static',3,'static-val','s')">Silk 3s</span>
          <span class="chip" onclick="setVal('static',5,'static-val','s')">Cotton 5s</span>
          <span class="chip" onclick="setVal('static',6,'static-val','s')">Wool 6s</span>
          <span class="chip" onclick="setVal('static',7,'static-val','s')">Poly 7s</span>
          <span class="chip" onclick="setVal('static',30,'static-val','s')">⚠ Long 30s</span>
        </div>
      </div>

      <button class="btn-predict" id="predict-btn" onclick="predictManual()">
        <span class="btn-text">⚡ Analyse Fabric</span>
        <div class="spinner"></div>
      </button>
    </div>

    <!-- Fabric Reference -->
    <div class="card reveal" style="transition-delay:.12s">
      <div class="card-title">📊 Fabric Reference Guide</div>
      <div style="display:grid;gap:14px">
        {% for name, info in fabric_info.items() %}
        {% if name != 'Anomaly' %}
        <div style="display:flex;align-items:center;gap:14px;padding:12px;border-radius:10px;border:1px solid var(--border);background:rgba(255,255,255,.02)">
          <span style="font-size:28px;flex-shrink:0">{{ info.emoji }}</span>
          <div>
            <div style="font-family:'Orbitron',sans-serif;font-size:.8rem;color:{{ info.color }};font-weight:700">{{ name }}</div>
            <div style="font-size:.75rem;color:var(--dim);margin-top:3px">{{ info.temp_range }}</div>
            <div style="font-size:.72rem;color:var(--text);margin-top:4px;line-height:1.4">{{ info.tip[:80] }}…</div>
          </div>
        </div>
        {% endif %}
        {% endfor %}
        <div style="display:flex;align-items:center;gap:14px;padding:12px;border-radius:10px;border:1px solid rgba(231,76,60,.3);background:rgba(231,76,60,.05)">
          <span style="font-size:28px;flex-shrink:0">⚠️</span>
          <div>
            <div style="font-family:'Orbitron',sans-serif;font-size:.8rem;color:#E74C3C;font-weight:700">Anomaly</div>
            <div style="font-size:.75rem;color:var(--dim);margin-top:3px">&lt;100°C or &gt;210°C</div>
            <div style="font-size:.72rem;color:var(--text);margin-top:4px;line-height:1.4">Dangerous iron settings detected — adjust immediately.</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Result Panel -->
    <div class="card reveal" id="result-panel" style="grid-column:1/-1">
      <div class="card-title">🎯 Prediction Result</div>
      <div class="result-inner">
        <div class="result-icon-wrap" id="result-icon-wrap">
          <span id="result-emoji" style="font-size:52px">🧺</span>
        </div>
        <div>
          <div class="fabric-name" id="result-name">—</div>
          <div class="fabric-range" id="result-range">—</div>
          <div class="fabric-tip" id="result-tip">—</div>
          <div class="src-badge" id="src-badge"></div>
          <div class="conf-row">
            <div class="conf-label">Model Confidence</div>
            <div class="conf-track"><div class="conf-fill" id="conf-fill"></div></div>
            <div class="conf-pct" id="conf-pct">0%</div>
          </div>
        </div>
      </div>
      <div style="margin-top:28px">
        <div class="conf-label" style="margin-bottom:12px">All Class Probabilities</div>
        <div class="prob-grid" id="prob-grid"></div>
      </div>
    </div>

  </div><!-- /grid -->

  <!-- Quick Presets -->
  <div class="ref-section reveal" style="transition-delay:.2s">
    <div class="ref-title">Quick Presets — Click to Load</div>
    <div class="ref-grid">
      {% for name, info in fabric_info.items() %}
      <div class="ref-card" onclick="loadPreset('{{ name }}')"
           style="border-color:{{ info.color }}20"
           onmouseover="this.style.borderColor='{{ info.color }}80';this.style.boxShadow='0 8px 30px {{ info.glow }}'"
           onmouseout="this.style.borderColor='{{ info.color }}20';this.style.boxShadow='none'">
        <span class="r-emoji">{{ info.emoji }}</span>
        <div class="r-name">{{ name }}</div>
        <div class="r-temp">{{ info.temp_range }}</div>
      </div>
      {% endfor %}
    </div>
  </div>

</div><!-- /wrapper -->

<script>
// ── Background Particles ─────────────────────────────────────────
const canvas=document.getElementById('bg-canvas'),ctx=canvas.getContext('2d');
let W,H,particles=[];
function resize(){W=canvas.width=window.innerWidth;H=canvas.height=window.innerHeight;}
resize();window.addEventListener('resize',resize);
class Particle{constructor(){this.reset();}reset(){this.x=Math.random()*W;this.y=Math.random()*H;this.vx=(Math.random()-.5)*.3;this.vy=(Math.random()-.5)*.3;this.r=Math.random()*1.5+.3;this.a=Math.random()*.4+.1;this.hue=Math.random()>.7?30:200;}draw(){ctx.beginPath();ctx.arc(this.x,this.y,this.r,0,Math.PI*2);ctx.fillStyle=`hsla(${this.hue},100%,70%,${this.a})`;ctx.fill();}update(){this.x+=this.vx;this.y+=this.vy;if(this.x<0||this.x>W||this.y<0||this.y>H)this.reset();}}
for(let i=0;i<120;i++)particles.push(new Particle());
function animBg(){ctx.clearRect(0,0,W,H);particles.forEach(p=>{p.update();p.draw();});for(let i=0;i<particles.length;i++)for(let j=i+1;j<particles.length;j++){const dx=particles[i].x-particles[j].x,dy=particles[i].y-particles[j].y,d=Math.sqrt(dx*dx+dy*dy);if(d<100){ctx.beginPath();ctx.moveTo(particles[i].x,particles[i].y);ctx.lineTo(particles[j].x,particles[j].y);ctx.strokeStyle=`rgba(0,212,255,${.06*(1-d/100)})`;ctx.lineWidth=.5;ctx.stroke();}}requestAnimationFrame(animBg);}
animBg();

// ── Sliders ──────────────────────────────────────────────────────
function updateSlider(el,dispId,unit,dec=null){
  const v=parseFloat(el.value),mn=parseFloat(el.min),mx=parseFloat(el.max);
  el.style.setProperty('--pct',((v-mn)/(mx-mn)*100).toFixed(1)+'%');
  let d;
  if(dec!==null) d=v.toFixed(dec)+unit;
  else if(unit==='°C') d=v.toFixed(1)+' °C';
  else if(unit==='s')  d=v+' s';
  else d=v.toFixed(4);
  document.getElementById(dispId).textContent=d;
}
function setVal(id,val,dispId,unit,dec=null){
  const el=document.getElementById(id);el.value=val;
  updateSlider(el,dispId,unit,dec);
}
['temp','motion','static'].forEach(id=>{
  const el=document.getElementById(id);
  el.style.setProperty('--pct',((parseFloat(el.value)-parseFloat(el.min))/(parseFloat(el.max)-parseFloat(el.min))*100).toFixed(1)+'%');
});

// ── Presets ──────────────────────────────────────────────────────
const PRESETS={
  Cotton:    {temp:192, motion:0.042, static:5},
  Silk:      {temp:128, motion:0.009, static:3},
  Wool:      {temp:136, motion:0.031, static:6},
  Polyester: {temp:130, motion:0.019, static:7},
  Anomaly:   {temp:235, motion:0.005, static:30}
};
function loadPreset(name){
  const p=PRESETS[name];if(!p)return;
  setVal('temp',  p.temp,  'temp-val',  '°C');
  setVal('motion',p.motion,'motion-val','',4);
  setVal('static',p.static,'static-val','s');
}

const FABRIC_COLORS={{ fabric_colors | safe }};
const FABRIC_GLOWS ={{ fabric_glows  | safe }};

// ── Show Result ──────────────────────────────────────────────────
function showResult(data, source){
  const panel=document.getElementById('result-panel');
  panel.classList.remove('show');void panel.offsetWidth;panel.classList.add('show');
  panel.classList.add('visible');

  const color=FABRIC_COLORS[data.fabric]||'#00d4ff';
  const glow =FABRIC_GLOWS [data.fabric]||'rgba(0,212,255,0.4)';
  const wrap=document.getElementById('result-icon-wrap');
  wrap.style.background=`radial-gradient(circle,${glow} 0%,transparent 70%)`;
  wrap.style.boxShadow=`0 0 50px ${glow}`;
  wrap.style.color=color;

  document.getElementById('result-emoji').textContent=data.emoji;
  document.getElementById('result-name').textContent=data.fabric;
  document.getElementById('result-name').style.color=color;
  document.getElementById('result-range').textContent='📍 Ideal Range: '+data.temp_range;
  document.getElementById('result-tip').textContent=data.tip;

  const sb=document.getElementById('src-badge');
  if(source==='sensor'){
    sb.textContent='📡 Predicted from Arduino Sensor';
    sb.style.cssText='background:rgba(0,255,157,.1);color:#00ff9d;border:1px solid rgba(0,255,157,.3);';
  }else{
    sb.textContent='✋ Predicted from Manual Input';
    sb.style.cssText='background:rgba(0,212,255,.1);color:#00d4ff;border:1px solid rgba(0,212,255,.3);';
  }

  const fill=document.getElementById('conf-fill');
  fill.style.background=`linear-gradient(90deg,${color},#fff)`;
  fill.style.boxShadow=`0 0 12px ${glow}`;
  setTimeout(()=>{fill.style.width=data.confidence+'%';},50);

  const pct=document.getElementById('conf-pct');
  pct.style.color=color;
  animCount(pct,0,data.confidence,1200,v=>v.toFixed(1)+'%');

  const grid=document.getElementById('prob-grid');
  grid.innerHTML='';
  for(const [cls,prob] of Object.entries(data.all_probabilities)){
    const c=FABRIC_COLORS[cls]||'#888';
    const div=document.createElement('div');
    div.className='prob-item'+(cls===data.fabric?' top':'');
    div.innerHTML=`<div class="p-label">${cls}</div>
      <div class="p-val" style="color:${c}">${prob.toFixed(1)}%</div>
      <div class="p-bar"><div class="p-bar-fill" style="background:${c}" data-w="${prob}"></div></div>`;
    grid.appendChild(div);
  }
  setTimeout(()=>{
    grid.querySelectorAll('.p-bar-fill').forEach(el=>{el.style.width=el.dataset.w+'%';});
  },100);

  panel.scrollIntoView({behavior:'smooth',block:'start'});
}

function animCount(el,from,to,dur,fmt){
  const start=performance.now();
  (function step(now){
    const t=Math.min((now-start)/dur,1);
    const v=from+(to-from)*(1-Math.pow(1-t,3));
    el.textContent=fmt(v);
    if(t<1)requestAnimationFrame(step);
  })(performance.now());
}

// ── Manual Predict ───────────────────────────────────────────────
async function predictManual(){
  const btn=document.getElementById('predict-btn');
  btn.classList.add('loading');
  try{
    const res=await fetch('/predict',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        temperature_c:    parseFloat(document.getElementById('temp').value),
        motion_variation: parseFloat(document.getElementById('motion').value),
        static_time_s:    parseInt(document.getElementById('static').value)
      })
    });
    const data=await res.json();
    if(data.error){alert('Error: '+data.error);return;}
    showResult(data,'manual');
  }catch(e){alert('Server error: '+e.message);}
  finally{btn.classList.remove('loading');}
}

// ── Sensor Predict ───────────────────────────────────────────────
async function predictFromSensor(){
  const btn=document.getElementById('btn-sensor');
  btn.disabled=true;
  try{
    const res=await fetch('/predict-sensor');
    const data=await res.json();
    if(data.error){alert('⚠ '+data.error);return;}
    showResult(data,'sensor');
  }catch(e){alert('Server error: '+e.message);}
  finally{btn.disabled=false;}
}

// ── Sensor Polling every 2 s ─────────────────────────────────────
let lastTs=null, autoTimer=null;

async function pollSensor(){
  try{
    const res=await fetch('/sensor-data');
    const d=await res.json();

    const hdrDot=document.getElementById('hdr-sensor-dot');
    const hdrLbl=document.getElementById('hdr-sensor-label');
    const cDot  =document.getElementById('conn-dot');
    const cLbl  =document.getElementById('conn-label');
    const btn   =document.getElementById('btn-sensor');

    if(d.connected && d.temperature_c!==null){
      // Green — live data
      hdrDot.className='dot';
      cDot.className  ='dot';
      hdrLbl.textContent='Sensor: '+d.port;
      cLbl.textContent  =d.port+' · Live';
      btn.disabled=false;

      const isNew=(lastTs!==d.timestamp);
      lastTs=d.timestamp;

      function updateBox(boxId,valId,barId,val,min,max,dec){
        const el=document.getElementById(valId);
        el.textContent=(dec===0&&Number.isInteger(val))?val:val.toFixed(dec);
        el.classList.remove('no-data');
        const pct=Math.max(0,Math.min(((val-min)/(max-min))*100,100));
        document.getElementById(barId).style.width=pct+'%';
        if(isNew){
          const b=document.getElementById(boxId);
          b.classList.remove('flash');void b.offsetWidth;b.classList.add('flash');
        }
      }
      updateBox('mb-temp',  'mv-temp',  'mbar-temp',  d.temperature_c,   60,    250, 1);
      updateBox('mb-motion','mv-motion','mbar-motion', d.motion_variation,0.001, 0.05,4);
      updateBox('mb-static','mv-static','mbar-static', d.static_time_s,   0,     45,  0);

      document.getElementById('sensor-age-text').textContent=
        (d.age_s!==null) ? d.age_s.toFixed(0)+'s ago' : 'just now';

      if(document.getElementById('auto-chk').checked && isNew){
        clearTimeout(autoTimer);
        autoTimer=setTimeout(predictFromSensor,600);
      }

    } else if(d.connected && d.temperature_c===null){
      // Connected but no data yet
      hdrDot.className='dot warn';
      cDot.className  ='dot warn';
      hdrLbl.textContent='Sensor: waiting for data';
      cLbl.textContent  =d.port+' · Connected';
      btn.disabled=true;
      document.getElementById('sensor-age-text').textContent='waiting…';

    } else {
      // Not connected
      hdrDot.className='dot red';
      cDot.className  ='dot red';
      hdrLbl.textContent='Sensor: not connected';
      cLbl.textContent  =d.port;
      btn.disabled=true;
      document.getElementById('sensor-age-text').textContent='no signal';
    }
  }catch(e){
    console.warn('Poll error:',e);
  }
}

setInterval(pollSensor, 2000);
pollSensor();   // run immediately on page load

// ── Scroll Reveal ────────────────────────────────────────────────
const observer=new IntersectionObserver(entries=>{
  entries.forEach(e=>{if(e.isIntersecting)e.target.classList.add('visible');});
},{threshold:0.1});
document.querySelectorAll('.reveal').forEach(el=>observer.observe(el));
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    fabric_colors = {k: v["color"] for k, v in FABRIC_INFO.items()}
    fabric_glows  = {k: v["glow"]  for k, v in FABRIC_INFO.items()}
    with sensor_lock:
        port = sensor_state["port"]
    return render_template_string(
        HTML,
        model_name    = MODEL_NAME,
        fabric_info   = FABRIC_INFO,
        fabric_colors = json.dumps(fabric_colors),
        fabric_glows  = json.dumps(fabric_glows),
        sensor_port   = port,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        T = float(data["temperature_c"])
        M = float(data["motion_variation"])
        S = int(data["static_time_s"])
        if not (60 <= T <= 250):
            return jsonify({"error": f"Temperature {T} out of range (60–250 °C)"}), 400
        if not (0 < M <= 0.05):
            return jsonify({"error": "Motion must be 0.001 – 0.05"}), 400
        if not (1 <= S <= 45):
            return jsonify({"error": "Static time must be 1 – 45 s"}), 400
        return jsonify(run_prediction(T, M, S))
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict-sensor")
def predict_sensor():
    with sensor_lock:
        T = sensor_state["temperature_c"]
        M = sensor_state["motion_variation"]
        S = sensor_state["static_time_s"]
    if T is None:
        return jsonify({"error": "No sensor data yet. Check Arduino is sending SENSOR,T,M,S lines."}), 503
    try:
        return jsonify(run_prediction(T, M, S))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/sensor-data")
def sensor_data():
    with sensor_lock:
        d = dict(sensor_state)
    ts = d.get("timestamp")
    d["age_s"] = round(time.time() - ts, 1) if ts else None
    return jsonify(d)


@app.route("/health")
def health():
    with sensor_lock:
        conn = sensor_state["connected"]
        port = sensor_state["port"]
    return jsonify({
        "status":           "ok",
        "model":            MODEL_NAME,
        "classes":          list(LE.classes_),
        "sensor_connected": conn,
        "sensor_port":      port,
    })


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    border = "═" * 56
    print(f"\n╔{border}╗")
    print(f"║{'  SMART IRON WEB SERVER'.center(56)}║")
    print(f"╠{border}╣")
    print(f"║  Model  : {MODEL_NAME:<45}║")
    print(f"║  URL    : http://localhost:5000{'':<25}║")
    print(f"║  Serial : {(args.port or 'none  —  use  --port COM3'):<45}║")
    print(f"╚{border}╝\n")

    # use_reloader=False → only ONE process, so only ONE serial thread
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)