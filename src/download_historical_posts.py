import os
import re
import time
import json
import datetime as dt
from pathlib import Path

import langid
import ujson as uj
import zstandard as zstd
from dotenv import load_dotenv
import tqdm

# ---- CONFIG ----
load_dotenv()

SUBREDDITS = os.getenv("SUBREDDITS", "").split(",")
LANGS_KEEP = {"en", "es"}  # inglés + español
DATE_START = int(dt.datetime(2020, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc).timestamp())
DATE_END = int(dt.datetime(2024, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc).timestamp())

# Directorios
BASE_DIR = Path("data/bronze")
RAW_DIR = BASE_DIR / "raw_zst"
OUT_PARQUET = BASE_DIR

for p in [RAW_DIR, OUT_PARQUET]:
    p.mkdir(parents=True, exist_ok=True)

# Torrents (AcademicTorrents hashes -> magnet)
MAGNET_TOP40K = "magnet:?xt=urn:btih:1614740ac8c94505e4ecb9d88be8bed7b6afddd4"
MAGNET_2005_2023 = "magnet:?xt=urn:btih:9c263fc85366c1ef8f5bb9da0203f4c8c8db75f4"
MAGNET_2024_12 = "magnet:?xt=urn:btih:eb2017da9f63a49460dde21a4ebe3b7c517f3ad9"

RS_MONTHS = {(y, m) for y in range(2020, 2025) for m in range(1, 13)}  # 2020..2024

# ---- BitTorrent helper (libtorrent) ----
try:
    import libtorrent as lt
except Exception as e:
    raise SystemExit(
        "libtorrent (python-libtorrent) no está instalado o no carga correctamente.\n"
        "Intenta: pip install python-libtorrent\n"
        f"Detalle: {e}"
    )

def add_and_select_files(magnet, dst_dir, select_func):
    ses = lt.session()
    ses.listen_on(6881, 6891)

    atp = lt.add_torrent_params()
    atp.url = magnet
    atp.save_path = str(dst_dir)
    h = ses.add_torrent(atp)

    print(f"▶ Añadiendo torrent: {magnet[:80]}...")

    start_time = time.time()
    timeout = 3600  # 1 hora como límite

    while not h.has_metadata():
        if time.time() - start_time > timeout:
            raise TimeoutError("Tiempo excedido para obtener metadata del torrent.")
        time.sleep(1)
    print("  ✓ Metadata obtenida.")

    ti = h.get_torrent_info()

    # Pausar el torrent antes de aplicar prioridades
    h.pause()

    # Aplicar prioridades
    priorities = [select_func(ti.files().file_path(idx)) for idx in range(ti.files().num_files())]
    h.prioritize_files(priorities)

    # Forzar prioridad 0 para archivos no deseados
    for file_index, priority in enumerate(priorities):
        if priority == 0:
            h.file_priority(file_index, 0)  # Desactivar archivo

    # Depuración: Verificar prioridades aplicadas
    for file_index, priority in enumerate(priorities):
        actual_priority = h.file_priority(file_index)
        print(f"Archivo: {ti.files().file_path(file_index)}, Prioridad aplicada: {actual_priority}")

    # Reanudar el torrent después de aplicar prioridades
    h.resume()

    print("  ▶ Descargando archivos seleccionados...")
    progress_bar = tqdm.tqdm(total=ti.total_size(), unit="B", unit_scale=True, desc="Progreso")

    while not h.status().is_seeding:
        status = h.status()
        progress_bar.n = status.total_done
        progress_bar.refresh()

        if time.time() - start_time > timeout:
            progress_bar.close()
            raise TimeoutError("Tiempo excedido para completar la descarga del torrent.")

        time.sleep(2)

    progress_bar.close()
    print("  ✓ Descarga finalizada.")
    return h

# Ajuste para depuración: Verificar subreddits cargados y formato
print(f"Subreddits cargados: {SUBREDDITS}")

# Ajuste en select_top40k para depuración adicional
# Verificar coincidencia exacta de subreddits y fechas

def select_top40k(path: str) -> int:
    pl = path.lower()
    print(f"Evaluando archivo: {path}")  # Depuración

    # Verificar si el archivo pertenece a un subreddit relevante
    if "/submissions/" in pl and any(f"/{s.lower()}/" in pl for s in SUBREDDITS):
        print(f"Coincidencia de subreddit para archivo: {path}")  # Depuración

        # Filtrar por rango de fechas si el nombre del archivo contiene un timestamp
        match = re.search(r"(\d{4})-(\d{2})", pl)
        if match:
            year, month = int(match.group(1)), int(match.group(2))
            file_date = dt.datetime(year, month, 1, tzinfo=dt.timezone.utc).timestamp()
            if DATE_START <= file_date <= DATE_END:
                print(f"Archivo dentro del rango de fechas: {path}")  # Depuración
                return 7  # Alta prioridad para archivos relevantes
            else:
                print(f"Archivo fuera del rango de fechas: {path}")  # Depuración
        else:
            print(f"No se encontró fecha en el archivo: {path}")  # Depuración
    else:
        print(f"Archivo omitido por coincidencia de subreddit: {path}")  # Depuración

    return 0  # Omitir archivos irrelevantes

# Ajuste en select_monthlies_rs para depuración adicional
# Verificar coincidencia exacta de fechas

def select_monthlies_rs(path: str) -> int:
    name = os.path.basename(path)
    print(f"Evaluando archivo mensual: {name}")  # Depuración

    m = re.match(r"(?i)RS_(\d{4})-(\d{2})\.zst$", name)
    if not m:
        print(f"Archivo omitido por formato incorrecto: {name}")  # Depuración
        return 0

    y, mth = int(m.group(1)), int(m.group(2))
    file_date = dt.datetime(y, mth, 1, tzinfo=dt.timezone.utc).timestamp()
    if (y, mth) in RS_MONTHS and DATE_START <= file_date <= DATE_END:
        print(f"Archivo mensual dentro del rango: {name}")  # Depuración
        return 6  # Prioridad media para archivos relevantes
    else:
        print(f"Archivo mensual fuera del rango: {name}")  # Depuración

    return 0

# ---- Descarga ----
print("=== Paso 1: Descarga de torrents seleccionados ===")
try:
    add_and_select_files(MAGNET_TOP40K, RAW_DIR, select_top40k)
except Exception as e:
    print(f"⚠ No se pudo descargar top40k: {e}")

for mg in (MAGNET_2005_2023, MAGNET_2024_12):
    try:
        add_and_select_files(mg, RAW_DIR, select_monthlies_rs)
    except Exception as e:
        print(f"⚠ No se pudo descargar {mg[:48]}: {e}")

print("\n=== Paso 2: Filtrado por fecha e idioma ===")

def iter_zst_lines(zst_path: Path):
    dctx = zstd.ZstdDecompressor()
    with open(zst_path, "rb") as fh, dctx.stream_reader(fh) as reader:
        buf = b""
        while chunk := reader.read(1 << 20):
            buf += chunk
            while (nl := buf.find(b"\n")) != -1:
                yield buf[:nl]
                buf = buf[nl + 1:]
        if buf.strip():
            yield buf

def month_key(ts_utc: int) -> str:
    d = dt.datetime.utcfromtimestamp(ts_utc)
    return f"{d.year:04d}-{d.month:02d}"

def detect_lang(title: str, selftext: str) -> str:
    text = f"{title} {selftext}".strip()
    return langid.classify(text)[0] if text else "und"

zst_files = list(RAW_DIR.rglob("*.zst"))
print(f"  Encontrados {len(zst_files)} archivos .zst para procesar.")

kept_total = 0
for zf in zst_files:
    try:
        for raw in iter_zst_lines(zf):
            try:
                obj = uj.loads(raw)
            except Exception:
                continue

            sub = (obj.get("subreddit") or "").lower()
            if sub not in SUBREDDITS:
                continue

            ts = obj.get("created_utc")
            if not isinstance(ts, (int, float)) or not (DATE_START <= int(ts) <= DATE_END):
                continue

            lang = detect_lang(obj.get("title", ""), obj.get("selftext", ""))
            if lang not in LANGS_KEEP:
                continue

            mk = month_key(int(ts))
            out_dir = OUT_PARQUET / sub / mk
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{sub}_{mk}.jsonl"

            with open(out_path, "ab") as out_f:
                out_f.write(raw + b"\n")
                kept_total += 1
        print(f"  ✓ Procesado: {zf}")
    except Exception as e:
        print(f"⚠ Error procesando {zf}: {e}")

print(f"\n✓ Filtrado completado. Líneas retenidas: {kept_total:,}")
