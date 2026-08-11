#!/bin/bash
# =============================================================================
# SIGINT Node v16.4 -- PMR446 Wideband IFF Scanner
# Daemon: HackRF permanente + autorecuperacion
# =============================================================================

# NOTE: credentials were hardcoded here in the original version. Redacted before
# the first commit; the Telegram dispatch path is removed entirely in the rewrite
# (see docs/06_legal_ethics.md - the node is metadata-only).
readonly TOKEN="${EW_TELEGRAM_TOKEN:-REDACTED}"
readonly CHAT_ID="${EW_TELEGRAM_CHAT_ID:-REDACTED}"
readonly WORK_DIR="/tmp/sigint_$$"
readonly MIN_RAW_BYTES="24000"
readonly IFF_SCRIPT="$HOME/EW_suite/iff_detector.py"
readonly CH_SCRIPT="$HOME/EW_suite/channelizer.py"
readonly MAX_RESTARTS=5
readonly RESTART_DELAY=5

# ── AÑADIR al bloque readonly del inicio ───────────────────────────────────
readonly RANGE_SCRIPT="$HOME/EW_suite/range_estimator.py"
readonly ITAK_IP="192.168.1.100"        # ← IP del dispositivo ITAK/WinTAK
readonly ITAK_PORT="4242"               # ← Puerto CoT UDP (por defecto ATAK: 4242)
readonly RX_LAT="40.4168"              # ← Tu posición GPS (Madrid ejemplo)
readonly RX_LON="-3.7038"
log() { echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*"; }

process_audio() {
    local input="$1"
    local output="$2"
    ffmpeg -hide_banner -loglevel error         -f s16le -ar 12000 -ac 1 -i "$input"         -af "highpass=f=300,lowpass=f=3000,volume=20dB,alimiter=limit=0.9:attack=1:release=10"         -ar 16000 -ac 1         -c:a aac -b:a 32k         "$output" -y
}

dispatch_to_telegram() {
    local audio_file="$1" status="$2" ch_num="$3" freq_hz="$4" hora="$5"
    local freq_mhz
    freq_mhz=$(echo "scale=5; $freq_hz / 1000000" | bc)
    local caption
    caption=$(printf "%s

Canal: PMR%s
Freq:  %s MHz
Hora:  %s"               "$status" "$ch_num" "$freq_mhz" "$hora")
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}"         -X POST "https://api.telegram.org/bot${TOKEN}/sendAudio"         -F "chat_id=${CHAT_ID}"         -F "audio=@${audio_file}"         -F "caption=${caption}")
    if [[ "$http_code" != "200" ]]; then
        log "[ERROR] Telegram fallo. HTTP: $http_code"; return 1
    fi
    log "[OK] Enviado. HTTP: $http_code"
}

cleanup() {
    log "[*] Terminando..."
    pkill -9 -f "channelizer.py" 2>/dev/null
    kill -- -$$ 2>/dev/null
    rm -rf "$WORK_DIR"
    exit 0
}
trap cleanup SIGINT SIGTERM

mkdir -p "$WORK_DIR"
log "[*] Nodo SIGINT v16.4 | PID: $$ | 16 canales PMR446 EU | Daemon"

log "[*] Arrancando daemon HackRF..."
python3 "$CH_SCRIPT" "$WORK_DIR" &
CH_PID=$!
log "[*] Daemon PID: $CH_PID"

RESTARTS=0

while true; do
    if ! kill -0 $CH_PID 2>/dev/null; then
        RESTARTS=$((RESTARTS + 1))
        log "[WARN] Daemon caido -- reinicio $RESTARTS/$MAX_RESTARTS"

        if [ $RESTARTS -ge $MAX_RESTARTS ]; then
            log "[CRITICAL] $MAX_RESTARTS fallos -- pausa 60s"
            sleep 60
            RESTARTS=0
        fi

        sleep $RESTART_DELAY
        pkill -9 -f "channelizer.py" 2>/dev/null
        rm -f "$WORK_DIR"/raw_*.s16 "$WORK_DIR"/freq_*.txt
        python3 "$CH_SCRIPT" "$WORK_DIR" &
        CH_PID=$!
        log "[*] Daemon reiniciado PID: $CH_PID"
    fi

    for FREQ_FILE in "$WORK_DIR"/freq_*.txt; do
        [ -f "$FREQ_FILE" ] || continue

        TS=$(basename "$FREQ_FILE" .txt | sed "s/freq_//")
        RAW_FILE="${WORK_DIR}/raw_${TS}.s16"
        OUT_FILE="${WORK_DIR}/capture_${TS}.m4a"

        [ -f "$RAW_FILE" ] || continue

        HORA=$(date +%H:%M:%S)
        CH_NUM=$(cut -d, -f1 "$FREQ_FILE")
        FREQ_HZ=$(cut -d, -f2 "$FREQ_FILE")
        STATUS=$(python3 "$IFF_SCRIPT" < "$RAW_FILE")
        STATUS="${STATUS:-DESCONOCIDO}"

        log "[*] PMR${CH_NUM} | ${FREQ_HZ}Hz | ${STATUS}"

        process_audio "$RAW_FILE" "$OUT_FILE"
        dispatch_to_telegram "$OUT_FILE" "$STATUS" "$CH_NUM" "$FREQ_HZ" "$HORA"


	# ── SUSTITUIR el bloque de procesamiento dentro del for loop ───────────────
# (después de que STATUS ya está calculado, antes del rm final)

        # ── Geolocalización → ITAK ──────────────────────────────────────────
        POWER=$(cut -d, -f3 "$FREQ_FILE")   # el tercer campo que añadimos

        if [[ -n "$POWER" && -f "$RANGE_SCRIPT" ]]; then
            log "[GEO] Estimando rango PMR${CH_NUM} pwr=${POWER}"

            # Genera CoT XML y envía cada anillo a ITAK por UDP
            python3 "$RANGE_SCRIPT" \
                "$CH_NUM" "$FREQ_HZ" "$POWER" \
                "$RX_LAT" "$RX_LON" "$STATUS" \
            | grep -v "^---$" \
            | while IFS= read -r line; do
                # Acumula hasta el cierre </event> y envía un datagrama UDP
                COT_BUF="${COT_BUF}${line}"$'\n'
                if [[ "$line" == "</event>" ]]; then
                    echo "$COT_BUF" | nc -u -w1 "$ITAK_IP" "$ITAK_PORT"
                    log "[CoT] Enviado anillo a ${ITAK_IP}:${ITAK_PORT}"
                    COT_BUF=""
                fi
              done
        fi
        # ── Fin Geolocalización ─────────────────────────────────────────
        rm -f "$RAW_FILE" "$FREQ_FILE" "$OUT_FILE"
    done

    sleep 0.3
done
