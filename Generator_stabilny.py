# -*- coding: utf-8 -*-
import adsk.core, adsk.fusion, traceback
import os, csv, math, random, statistics, json, time, re

# =========================================================
# USTAWIENIA GLOBALNE
# =========================================================
EXPORT_DIR = r"C:\Users\szymi\OneDrive\Pulpit\Studia\II_Stopien\Praca\roi\Noweklasy\MODEL"
TOP_SURFACE_DIR = r"C:\Users\szymi\OneDrive\Pulpit\Studia\II_Stopien\Praca\roi\Noweklasy\TOP_SURFACE"
CSV_DIR = r"C:\Users\szymi\OneDrive\Pulpit\Studia\II_Stopien\Praca\roi\Noweklasy\CSV"
THR_PATH = os.path.join(CSV_DIR, "_dwca_thresholds.json")
MODELS_CSV_PATH = os.path.join(CSV_DIR, "models_data.csv")
START_MODEL_INDEX = 0
NUMBER_OF_MODELS = 62                        # ile brył wygenerować
CUTS_PER_MODEL_MIN, CUTS_PER_MODEL_MAX = 18, 25
SEED = 42

# Balans łączny klas 4-bit (D,W,C,A) – 16 klas po równo
MODEL_TARGET = "round_robin"  # "round_robin" albo "deficit"
MODEL_TARGET_OFFSET = 0      # przesunięcie startu w permutacji klas modeli
BIAS_TOWARD_TARGET = True
STOP_WHEN_TARGET_REACHED = True
TARGET_EPS = 1e-9 # margines przy sprawdzaniu udziałów
BALANCE_MODE = "gmm"    # "quartile" albo "gmm"
RECALC_EVERY = 60            # co ile zarejestrowanych nacięć przeliczać progi
MIN_WARMUP   = 50            # zanim pierwsze progi będą z mediany/GMM
WARMUP_MODELS = 20          
FREEZE_AFTER_WARMUP = True   # czy zamrozić progi po warm-up
USE_DEFICIT_CLASS = True     # czy wybierać klasę nacięcia z deficytu

# Rozmiary BRYŁY (mm) – zbliżone do rzeczywistych
LENGTH_MM_MIN, LENGTH_MM_MAX = 30.0, 90.0
WIDTH_MM_MIN,  WIDTH_MM_MAX  = 18.0, 28.0
HEIGHT_MM_MIN, HEIGHT_MM_MAX = 3.0,  8.0

# Zakresy losowania parametrów NACIECIA (mm/deg)
X_MARGIN_MM   = 2.0
Y_MARGIN_MM   = 1.0
DEPTH_MIN_MM  = 0.06
DEPTH_MAX_MM  = 0.2
WIDTH_MIN_MM  = 0.16
WIDTH_MAX_MM  = 0.7
ANGLE_MAX_DEG = 20.0
CAP_UP_MM     = 0.25
TOP_THICKNESS_MM = 0.3
NUM_PTS_MIN, NUM_PTS_MAX = 5, 11  # gęstość splajnu

# Debug – pozwala zostawić szkice/widoczność dla weryfikacji
DEBUG_KEEP_SKETCHES = False
DEBUG_LOG = True
DISABLE_PREVIEW = True
SHARE_THRESH = 0.48
MIN_ANGLE_DEG = 1.5
RESUME_FROM_CSV = True
AUTO_LOAD_THRESHOLDS = True
AUTO_SAVE_THRESHOLDS = True
SKIP_WARMUP_IF_LOADED = True
THR_VERSION = 1

# ===== Strict mode: wszystkie nacięcia w modelu mają klasę celu (DWCA) =====
STRICT_CUTS_TO_TARGET = True        # włącz/wyłącz tryb twardy
FILL_MODEL_WITH_TARGET = True       # True -> generuj aż do max_cuts, False -> tylko min_cuts
MAX_ATTEMPTS_PER_CUT = 20           # ile prób na jedno nacięcie tej klasy, zanim uznamy porażkę
FAIL_HARD_ON_MISS = False           # True -> przerwij model jeśli nie da się trafić klasy; False -> zakończ z tym co jest

# Top-surface
TOP_THICKNESS_MM = 0.2
TOP_FACE_Z_TOL_MM = 0.02  # tolerancja przy wyborze najwyższych ścian
FORBIDDEN_CLASSES = set()
SKIP_FORBIDDEN_IN_TARGET = True  # czy pomijać FORBIDDEN_CLASSES przy round-robin
RADIUS_THR_OVERRIDE: float | None = None  # jeśli podane, to nadpisuje próg promienia
RADIUS_THR_SCALE: float = 0.6
RADIUS_THR_OFFSET: float = 0.0

# =========================================================
# POMOCNICZE – jednostki, losowość
# =========================================================
def mm_to_internal(design, val_mm: float) -> float:
    return design.unitsManager.convert(val_mm, 'mm', 'cm')

def internal_to_mm(design, val_internal: float) -> float:
    return design.unitsManager.convert(val_internal, 'cm', 'mm')

def randf(a,b): return a + (b-a)*random.random()

def log(msg):
    if not DEBUG_LOG:
        return
    try:
        with open(os.path.join(CSV_DIR, "_debug_log.txt"), "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")
    except Exception:
        pass

def save_dwca_thresholds(path: str, thr: dict, *, n_samples: int | None = None, method: str | None = None):
    # thr może być z kluczami 'depth','width','radius','angle' (bal.thresh)
    _map = {
        "D": float(thr["depth"])  if "depth"  in thr else float(thr["D"]),
        "W": float(thr["width"])  if "width"  in thr else float(thr["W"]),
        "C": float(thr["radius"]) if "radius" in thr else float(thr["C"]),
        "A": float(thr["angle"])  if "angle"  in thr else float(thr["A"]),
    }
    payload = {
        "version": THR_VERSION,
        "thresholds": _map,
        "meta": {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method": method or "warmup",
            "n_samples": int(n_samples or 0),
            "units": {"D":"mm","W":"mm","C":"mm","A":"deg"}
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_dwca_thresholds(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        t = data.get("thresholds") or {}
        if not all(k in t for k in ("D","W","C","A")):
            return None
        # Zwracamy w formacie bal.thresh:
        return {"depth": float(t["D"]), "width": float(t["W"]),
                "radius": float(t["C"]), "angle": float(t["A"])}
    except Exception:
        return None
    

def apply_radius_threshold_bias(bal) -> None:
    """
    Jednorazowo modyfikuje bal.thresh['radius'] po warm-upie.
    Jeśli ustawiono OVERRIDE -> przyjmuje wartość stałą.
    W przeciwnym razie skaluje auto-próg (r * SCALE - OFFSET).
    """
    if getattr(bal, "_radius_bias_applied", False):
        return
    r_auto = bal.thresh.get("radius", None)
    if r_auto is None:
        return
    if RADIUS_THR_OVERRIDE is not None:
        r_final = float(RADIUS_THR_OVERRIDE)
    else:
        r_final = max(1e-6, float(r_auto) * float(RADIUS_THR_SCALE) - float(RADIUS_THR_OFFSET))
    bal.thresh["radius"] = r_final
    bal._radius_bias_applied = True

def _norm(s: str) -> str:
    return (s or "").strip().lower()

def _try_bits(s: str) -> str | None:
    if not s:
        return None
    s = s.strip()
    if s.startswith("'"):  # Excelowy prefiks
        s = s[1:]
    return s if len(s) == 4 and set(s) <= {"0", "1"} else None

def compute_rr_offset_from_models_csv(csv_path: str,
                                      skip_forbidden: set[str] | None = None,
                                      fallback: int = 0,
                                      prefer_col: str = "ModelClassID_0_15") -> int:
    """
    Zwraca offset RR = (ostatnia_klasa + 1) % 16 na podstawie CSV z kolumną:
      - ModelClassID_0_15  (preferowana)
    Fallbacki (gdyby preferowana nie istnieła): ModelClass4DWA, TargetID, TargetBits.
    """
    if not os.path.exists(csv_path):
        return fallback

    last_id = None
    last_bits = None

    with open(csv_path, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        header = next(r, None) or []
        idx = { _norm(h): i for i, h in enumerate(header) }

        idx_id_pref    = idx.get(_norm(prefer_col))
        idx_bits_dwca  = idx.get("modelclass4dwa")
        idx_target_id  = idx.get("targetid")
        idx_target_bits= idx.get("targetbits")

        # Przejdź do ostatniego niepustego wiersza
        rows = [row for row in r if row and any((c or "").strip() for c in row)]
        if not rows:
            return fallback
        row = rows[-1]

        # 1) Preferowana: ModelClassID_0_15
        if idx_id_pref is not None and idx_id_pref < len(row):
            try:
                last_id = int(row[idx_id_pref])
                last_bits = bits4(last_id)
            except Exception:
                last_id = None

        # 2) Fallback: ModelClass4DWA (bity)
        if last_bits is None and idx_bits_dwca is not None and idx_bits_dwca < len(row):
            b = _try_bits(row[idx_bits_dwca])
            if b:
                last_bits = b
                last_id = int(b, 2)

        # 3) Fallback: TargetID
        if last_bits is None and idx_target_id is not None and idx_target_id < len(row):
            try:
                last_id = int(row[idx_target_id])
                last_bits = bits4(last_id)
            except Exception:
                pass

        # 4) Fallback: TargetBits
        if last_bits is None and idx_target_bits is not None and idx_target_bits < len(row):
            b = _try_bits(row[idx_target_bits])
            if b:
                last_bits = b
                last_id = int(b, 2)

    if last_bits is None and last_id is None:
        return fallback
    if last_id is None:
        last_id = int(last_bits, 2)

    offset = (last_id + 1)

    # Pomiń klasy zabronione (jeśli masz taką listę)
    if skip_forbidden:
        hop = 0
        while bits4(offset) in skip_forbidden and hop < 16:
            offset = (offset + 1)
            hop += 1
        if hop >= 16:
            return fallback

    return offset

def auto_start_from_csv(path: str) -> int:
    if not os.path.exists(path):
        return START_MODEL_INDEX
    try:
        with open(path, newline='', encoding='utf-8') as f:
            r = csv.reader(f)
            header = next(r, None) or []
            if not header:
                return START_MODEL_INDEX

            # 1) mapuj nazwy -> indeksy
            idx_map = { _norm(h): i for i, h in enumerate(header) }

            # 2) preferowane, dokładne nazwy kolumn z indeksem modelu
            preferred_keys = ("model_id", "modelid", "model index", "model_index")
            idx_col = next((idx_map[k] for k in preferred_keys if k in idx_map), None)

            # 3) jeśli brak – heurystyka (ale wyklucz "classid")
            if idx_col is None:
                for i, h in enumerate(header):
                    hh = _norm(h)
                    if "classid" in hh:
                        continue  # nie myl z ModelClassID_0_15
                    if "model" in hh and ("id" in hh or "index" in hh):
                        idx_col = i
                        break

            # 4) awaryjnie pierwsza kolumna
            if idx_col is None:
                idx_col = 0

            last_id = None
            for row in r:
                if not row:
                    continue
                if idx_col >= len(row):
                    continue
                cell = (row[idx_col] or "").strip()
                # usuń ewentualny excelowy apostrof
                if cell.startswith("'"):
                    cell = cell[1:]
                # wyciągnij liczby (gdyby były jakieś znaki)
                m = re.search(r"-?\d+", cell)
                if not m:
                    continue
                try:
                    last_id = int(m.group(0))
                except:
                    pass

            return (last_id + 1) if last_id is not None else START_MODEL_INDEX
    except:
        return START_MODEL_INDEX

# ---------------------------- Model target picking / steering ----------------------------
def bits4(i: int) -> str:
    return f"{i:04b}"

def id_from_bits(bits: str) -> int:
    return int(bits, 2)

def make_force_bits_from_bits(bits: str) -> dict:
    """Zamień 'DWCA' -> {'D':0/1,'W':0/1,'C':0/1,'A':0/1}"""
    d,w,c,a = [int(x) for x in bits]
    return {'D': d, 'W': w, 'C': c, 'A': a}

def is_forbidden_class(bits: str) -> bool:
    """Prosty filtr niemożliwych klas, żeby uniknąć pętli bez końca."""
    return bool(bits) and bits in FORBIDDEN_CLASSES

def build_round_robin_schedule(num_models: int, warmup_models: int = 0,
                               offset: int = 0, skip_forbidden: set | None = None) -> list[int]:
    """
    Zwraca listę docelowych klas (ID 0..15) dla modeli po warm-upie:
    0,1,2,...,15,0,1,2... z ew. pomijaniem FORBIDDEN_CLASSES.
    """
    after = max(0, num_models - warmup_models)
    seq = []
    cur = offset % 16
    while len(seq) < after:
        if not skip_forbidden or bits4(cur) not in skip_forbidden:
            seq.append(cur)
        cur = (cur + 1) % 16
    return seq

def next_round_robin_target(model_idx: int, warmup_models: int,
                            offset: int = 0, forbidden: set | None = None) -> tuple[int | None, str]:
    """
    Zwraca (target_id, target_bits) dla danego modelu liczonym od 1.
    W warm-upie -> (None,"").
    """
    k = model_idx - warmup_models
    if k <= 0:
        return None, ""
    cur = (k - 1 + offset) % 16
    if forbidden:
        seen = 0
        while bits4(cur) in forbidden and seen < 16:
            cur = (cur + 1) % 16
            seen += 1
    return cur, bits4(cur)

def _share_ok(ones: int, total: int, want_one: int, thr: float, eps: float = TARGET_EPS) -> bool:
    if total <= 0:
        return (want_one == 0)
    share = ones / float(total)
    return (share + eps >= thr) if want_one == 1 else (share < thr - eps)

def compute_force_bits_for_target(model_bit_ones: dict, total: int,
                                  target_bits: str, share_thresh: float) -> dict:
    """
    Zwraca słownik {'D':0/1, 'W':..., 'C':..., 'A':...} tylko dla tych bitów,
    które NIE spełniają jeszcze udziału względem `share_thresh`. Reszta pominięta.
    """
    if not target_bits or len(target_bits) != 4:
        return {}
    force = {}
    for letter, want_one in zip("DWCA", [int(b) for b in target_bits]):
        if not _share_ok(model_bit_ones.get(letter, 0), total, want_one, share_thresh):
            force[letter] = want_one
    return force

def target_reached(model_bit_ones: dict, total: int,
                   target_bits: str, share_thresh: float) -> bool:
    if not target_bits or len(target_bits) != 4:
        return False
    return all(_share_ok(model_bit_ones.get(letter, 0), total, want, share_thresh)
               for letter, want in zip("DWCA", [int(b) for b in target_bits]))

def update_counts_from_cut(metrics: dict, counts: dict) -> None:
    """
    Zwiększa liczniki bitów na podstawie klasy cięcia zapisanej w metrics['klass'] (format 'DWCA').
    """
    kb = f"{metrics['klass']:04b}" if isinstance(metrics['klass'], int) else str(metrics['klass'])
    for letter, b in zip("DWCA", kb):
        counts[letter] = counts.get(letter, 0) + (1 if b == "1" else 0)

def drive_cuts_for_model(design, root, body, L_mm, W_mm, H_mm, balancer,
                         target_bits: str, min_cuts: int, max_cuts: int,
                         share_thresh: float) -> tuple[list[dict], dict]:
    """
    Generuje cięcia dla jednego modelu. Jeśli `target_bits` ≠ "", biasuje kolejne
    cięcia w stronę celu DWCA. Zwraca (cut_metrics_list, bit_counts_dict).
    """
    cut_metrics: list[dict] = []
    counts = {'D': 0, 'W': 0, 'C': 0, 'A': 0}
    total = 0
    fails = 0

    # Warm-up lub brak celu
    if not target_bits:
        while len(cut_metrics) < min_cuts:
            ok, m = add_single_cut(design, root, body, L_mm, W_mm, H_mm, balancer, force_bits=None)
            if not ok:
                fails += 1
                if fails > 200:
                    break
                continue
            cut_metrics.append(m)
            update_counts_from_cut(m, counts)
        return cut_metrics, counts

    # Biasowany tryb z celem DWCA
    while total < max_cuts:
        force_bits = None
        if BIAS_TOWARD_TARGET:
            fb = compute_force_bits_for_target(counts, total, target_bits, share_thresh)
            force_bits = fb or None

        ok, m = add_single_cut(design, root, body, L_mm, W_mm, H_mm, balancer, force_bits=force_bits)
        if not ok:
            fails += 1
            if fails > 200:
                break
            continue

        cut_metrics.append(m)
        total += 1
        update_counts_from_cut(m, counts)

        if STOP_WHEN_TARGET_REACHED and total >= min_cuts and target_reached(counts, total, target_bits, share_thresh):
            break

    # generowanie do min_cuts już bez biasu
    while total < min_cuts:
        ok, m = add_single_cut(design, root, body, L_mm, W_mm, H_mm, balancer, force_bits=None)
        if not ok:
            fails += 1
            if fails > 200:
                break
            continue
        cut_metrics.append(m)
        total += 1
        update_counts_from_cut(m, counts)

    return cut_metrics, counts

def drive_cuts_strict_to_target(
    design, root, body, L_mm, W_mm, H_mm, balancer,
    target_bits: str, min_cuts: int, max_cuts: int,
    per_cut_attempts: int = None,
    fill_all: bool = None,
    fail_hard: bool = None
) -> tuple[list[dict], dict, bool]:
    """
    Generuje WYŁĄCZNIE nacięcia klasy 'target_bits' (DWCA).
    Zwraca (cut_metrics_list, bit_counts_dict, ok_bool).

    - Jeśli klasa jest na liście FORBIDDEN_CLASSES -> natychmiast ok=False.
    - Każde nacięcie próbuje do `per_cut_attempts` razy (domyślnie MAX_ATTEMPTS_PER_CUT).
    - Jeśli `fill_all=True` -> celuj do max_cuts, inaczej tylko do min_cuts.
    - Gdy nie uda się uzyskać wymaganego nacięcia:
        * fail_hard=True  -> zwróć ok=False (przerwij model),
        * fail_hard=False -> zakończ z tym, co udało się wygenerować (ok=False).
    """
    per_cut_attempts = MAX_ATTEMPTS_PER_CUT if per_cut_attempts is None else int(per_cut_attempts)
    fill_all = FILL_MODEL_WITH_TARGET if fill_all is None else bool(fill_all)
    fail_hard = FAIL_HARD_ON_MISS if fail_hard is None else bool(fail_hard)

    cut_metrics: list[dict] = []
    counts = {'D': 0, 'W': 0, 'C': 0, 'A': 0}

    # Brak celu albo klasa zakazana -> od razu STOP
    if not target_bits or len(target_bits) != 4:
        return cut_metrics, counts, False
    if is_forbidden_class(target_bits):
        return cut_metrics, counts, False

    target_id = id_from_bits(target_bits)
    force_bits = make_force_bits_from_bits(target_bits)

    required = random.randint(min_cuts, max_cuts) if fill_all else min_cuts
    while len(cut_metrics) < required:
        attempts = 0
        made = False
        while attempts < per_cut_attempts:
            ok, m = add_single_cut(
                design, root, body, L_mm, W_mm, H_mm, balancer,
                force_bits=force_bits
            )
            attempts += 1
            if not ok:
                continue
            # Sprawdź, czy cięcie ma dokładnie pożądaną klasę
            k = m.get('klass')
            kb = f"{k:04b}" if isinstance(k, int) else str(k)
            if kb == target_bits:
                cut_metrics.append(m)
                # aktualizuj liczniki bitów (wiemy że to target_bits)
                for letter, bit in zip("DWCA", target_bits):
                    counts[letter] += (1 if bit == '1' else 0)
                made = True
                break
            # jeśli nie trafiliśmy, próbuj dalej; żadnych "nudge" – tryb twardy
        if not made:
            # nie udało się wygenerować jednego nacięcia tej klasy
            if fail_hard or len(cut_metrics) < min_cuts:
                # przerwij model w trybie "hard" lub gdy nawet minimum nie jest osiągnięte
                return cut_metrics, counts, False
            # miękko kończymy wcześniej (zostaw co udało się zebrać)
            break

    return cut_metrics, counts, True

def build_model_schedule(num_models: int, warmup_models: int = 0, seed: int = None):
    """
    Zwraca listę docelowych klas modeli (ID 0..15) długości `num_models`.
    Po `warmup_models` pierwszych modelach (które są „rozgrzewką” bez celu)
    idziemy wg permutacji 0..15 powtarzanej i tasowanej.
    """
    rng = random.Random(SEED if seed is None else seed)
    after_warmup = max(0, num_models - warmup_models)
    schedule = []
    base = list(range(16))
    # Ile pełnych bloków 16 potrzebujemy
    blocks = (after_warmup + 15) // 16
    for _ in range(blocks):
        block = base[:]
        rng.shuffle(block)
        schedule.extend(block)
    # Przytnij dokładnie do długości „po warm-upie”
    schedule = schedule[:after_warmup]
    return schedule


def compute_model_class(cut_metrics_list, balancer, share_thresh=SHARE_THRESH):
    """
    Klasa modelu z udziałów (>= share_thresh) względem progów balancera.
    Zwraca: klass 'DWCA', mediany (D,W,C,A) i udziały (shareD/W/C/A).
    """
    if not cut_metrics_list:
        return "0000", 0,0,0,0, 0,0,0,0

    depths = [m['depth_mm'] for m in cut_metrics_list]
    widths = [m['width_mm'] for m in cut_metrics_list]
    radii  = [m['radius_mm'] for m in cut_metrics_list]
    angles = [abs(m['angle_deg']) for m in cut_metrics_list]

    td, tw, tr, ta = (balancer.thresh[k] for k in ('depth','width','radius','angle'))

    shareD = sum(x >= td for x in depths)/len(cut_metrics_list)
    shareW = sum(x >= tw for x in widths)/len(cut_metrics_list)
    shareC = sum(x <= tr for x in radii )/len(cut_metrics_list)  # ostrzej = mniejszy promień
    shareA = sum(x >= ta for x in angles)/len(cut_metrics_list)

    D = 1 if shareD >= share_thresh else 0
    W = 1 if shareW >= share_thresh else 0
    C = 1 if shareC >= share_thresh else 0
    A = 1 if shareA >= share_thresh else 0
    klass = f"{D}{W}{C}{A}"

    return (klass,
            statistics.median(depths), statistics.median(widths),
            statistics.median(radii),  statistics.median(angles),
            shareD, shareW, shareC, shareA)

    

# =========================================================
# BALANSER KLAS – quartile / gmm
# =========================================================
class GlobalBalancer:
    """
    Utrzymuje docelowo równy rozkład 16 klas 4-bit (D,W,C,A).
    Progi wyznacza z dotychczasowych nacięć:
      - 'quartile': mediany cech,
      - 'gmm': GMM(2) w 1D (jeśli brak sklearn → fallback na medianę).
    """
    def __init__(self):
        self.depth = []
        self.width = []
        self.radius = []
        self.angle = []
        self.counts = {f"{d}{w}{c}{a}": 0
                       for d in (0,1) for w in (0,1) for c in (0,1) for a in (0,1)}
        self.targets = {k: 1.0/16 for k in self.counts}
        # progi startowe – umiarkowane (zostaną zastąpione po warm-up)
        self.thresh = dict(depth=0.1, width=0.4, radius=0.6, angle=8.0)
        self.locked = False
        self._last_recalc = 0

    def random_class(self):
        return f"{random.randint(0,1)}{random.randint(0,1)}{random.randint(0,1)}{random.randint(0,1)}"
    
    def _median(self, arr, default):
        return statistics.median(arr) if len(arr) > 0 else default

    def _gmm_split(self, data, default):
        # próba GMM(2) w 1D; w razie braku sklearn – median
        if len(data) < 8:
            return self._median(data, default)
        try:
            from math import sqrt, pi, exp
            # mini EM 1D bez sklearn (stabilny, kilkanaście iteracji)
            xs = data[:]
            m1, m2 = min(xs), max(xs)
            s1 = s2 = max(1e-6, statistics.pstdev(xs) or 1.0)
            w1 = w2 = 0.5
            for _ in range(20):
                # E: odpowiedzialności
                r1s = []
                r2s = []
                for x in xs:
                    n1 = (1/(s1*sqrt(2*pi))) * exp(-0.5*((x-m1)/s1)**2) * w1
                    n2 = (1/(s2*sqrt(2*pi))) * exp(-0.5*((x-m2)/s2)**2) * w2
                    s = n1 + n2 + 1e-12
                    r1s.append(n1/s); r2s.append(n2/s)
                # M: parametry
                w1 = sum(r1s)/len(xs); w2 = 1-w1
                m1 = sum(r*x for r,x in zip(r1s,xs))/(sum(r1s)+1e-12)
                m2 = sum(r*x for r,x in zip(r2s,xs))/(sum(r2s)+1e-12)
                s1 = sqrt(sum(r*(x-m1)**2 for r,x in zip(r1s,xs))/(sum(r1s)+1e-12) + 1e-9)
                s2 = sqrt(sum(r*(x-m2)**2 for r,x in zip(r2s,xs))/(sum(r2s)+1e-12) + 1e-9)
            # punkt przecięcia gęstości
            # w1*N(m1,s1) == w2*N(m2,s2) -> rozwiązanie kwadratowe
            A = 1/(2*s2*s2) - 1/(2*s1*s1)
            B = m1/(s1*s1) - m2/(s2*s2)
            C = (m2*m2)/(2*s2*s2) - (m1*m1)/(2*s1*s1) + math.log((s2*w1)/(s1*w2)+1e-12)
            if abs(A) < 1e-12:
                x0 = -C/(B+1e-12)
                return x0
            disc = B*B - 4*A*C
            if disc < 0: 
                return self._median(xs, default)
            r1 = (-B + math.sqrt(disc))/(2*A)
            r2 = (-B - math.sqrt(disc))/(2*A)
            # wybierz punkt pomiędzy środkami
            m_lo, m_hi = (m1,m2) if m1 < m2 else (m2,m1)
            cand = [r for r in (r1,r2) if m_lo <= r <= m_hi]
            return cand[0] if cand else self._median(xs, default)
        except Exception:
            return self._median(data, default)

    def _recalc_impl(self, mode="quartile"):
        if mode == "gmm":
            self.thresh['depth']  = self._gmm_split(self.depth,  0.1)
            self.thresh['width']  = self._gmm_split(self.width,  0.4)
            try:
                import math as _math
                if len(self.radius) >= 8:
                    _logr = [_math.log1p(max(1e-9, r)) for r in self.radius]
                    _x0   = self._gmm_split(_logr, default=_math.log1p(0.6))
                    self.thresh['radius'] = max(1e-6, _math.expm1(_x0))
                else:
                    self.thresh['radius'] = self._gmm_split(self.radius, 0.6)
            except Exception:
                self.thresh['radius'] = self._gmm_split(self.radius, 0.6)
            self.thresh['angle']  = self._gmm_split(self.angle,  8.0)
        else:
            self.thresh['depth']  = self._median(self.depth,  0.1)
            self.thresh['width']  = self._median(self.width,  0.4)
            self.thresh['radius'] = self._median(self.radius, 0.6)
            self.thresh['angle']  = self._median(self.angle,  8.0)

    def maybe_recalc(self, mode="quartile", force: bool = False):
        # NIE aktualizujemy progów, gdy są zamrożone (chyba że force=True)
        if self.locked and not force:
            return
        total = len(self.depth)
        if not force:
            if total < MIN_WARMUP: return
            if total - self._last_recalc < RECALC_EVERY: return
        self._recalc_impl(mode)
        self._last_recalc = total

    # wygodny alias do wymuszenia przeliczenia „tu i teraz”
    def recalc_now(self, mode="quartile"):
        self.maybe_recalc(mode, force=True)

    def classify(self, depth_mm, width_mm, radius_mm, angle_deg):
        d = 1 if depth_mm  >= self.thresh['depth']  else 0
        w = 1 if width_mm  >= self.thresh['width']  else 0
        c = 1 if radius_mm <= self.thresh['radius'] else 0
        a = 1 if abs(angle_deg) >= self.thresh['angle'] else 0
        return f"{d}{w}{c}{a}"

    def deficit_class(self):
        # który klucz jest najbardziej „pod kreską” względem celu?
        totals = sum(self.counts.values()) + 1e-9
        ratios = {k: self.counts[k]/totals for k in self.counts}
        return min(self.counts.keys(), key=lambda k: ratios[k] - self.targets[k])

    def push_sample(self, depth_mm, width_mm, radius_mm, angle_deg, klass):
        self.depth.append(depth_mm)
        self.width.append(width_mm)
        self.radius.append(radius_mm)
        self.angle.append(abs(angle_deg))
        if klass in self.counts:
            self.counts[klass] += 1
        else:
            self.counts[klass] = 1

    def set_thresholds(self, thr: dict):
        self.thresh = dict(thr)  # {'D':..,'W':..,'C':..,'A':..}

class ModelBalancer:
    """Balans 16 klas modeli (0..15)."""
    def __init__(self):
        self.counts = {i: 0 for i in range(16)}

    def deficit(self) -> int:
        total = sum(self.counts.values()) + 1e-9
        target = 1.0 / 16.0
        # najmniejsza nadwyżka względem celu
        return min(self.counts.keys(),
                   key=lambda k: (self.counts[k]/total) - target)

    def push(self, class_id: int):
        self.counts[class_id] = self.counts.get(class_id, 0) + 1


# =========================================================
# GEOMETRIA: bryła bazowa i pojedyncze nacięcie
# =========================================================
def build_base_block(design, root):
    L = randf(LENGTH_MM_MIN, LENGTH_MM_MAX)
    W = randf(WIDTH_MM_MIN,  WIDTH_MM_MAX)
    H = randf(HEIGHT_MM_MIN, HEIGHT_MM_MAX)

    sk = root.sketches.add(root.xYConstructionPlane)
    p1 = adsk.core.Point3D.create(0,0,0)
    p2 = adsk.core.Point3D.create(mm_to_internal(design,L), mm_to_internal(design,W), 0)
    sk.sketchCurves.sketchLines.addTwoPointRectangle(p1,p2)
    prof = sk.profiles.item(0)

    exts = root.features.extrudeFeatures
    exi = exts.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    exi.setDistanceExtent(False, adsk.core.ValueInput.createByReal(mm_to_internal(design,H)))
    ext = exts.add(exi)
    body = ext.bodies.item(0)
    if not DEBUG_KEEP_SKETCHES:
        try: sk.deleteMe()
        except: pass
    return body, L, W, H

def triangle_area(a, b, c):
    ux, uy, uz = b.x - a.x, b.y - a.y, b.z - a.z
    vx, vy, vz = c.x - a.x, c.y - a.y, c.z - a.z
    cx = uy*vz - uz*vy
    cy = uz*vx - ux*vz
    cz = ux*vy - uy*vx
    return 0.5*math.sqrt(cx*cx + cy*cy + cz*cz)

def circle_radius_mm(design, a, b, c):
    ab = a.distanceTo(b); bc = b.distanceTo(c); ca = c.distanceTo(a)
    p = ab*bc*ca
    area = triangle_area(a,b,c)
    if area < 1e-12: return 1e9
    r_internal = p/(4.0*area)
    return internal_to_mm(design, r_internal)

# === pomocnicze jak w starym generatorze ===
def _find_circle_center(p1, p2, p3):
    # Liczymy środek okręgu opisanego na trzech punktach w płaszczyźnie YZ (X wspólny)
    y1, z1 = p1.y, p1.z
    y2, z2 = p2.y, p2.z
    y3, z3 = p3.y, p3.z
    A = y1 * (z2 - z3) - y2 * (z1 - z3) + y3 * (z1 - z2)
    if abs(A) < 1e-9:
        return None
    B = (y1**2 + z1**2) * (z3 - z2) + (y2**2 + z2**2) * (z1 - z3) + (y3**2 + z3**2) * (z2 - z1)
    C = (y1**2 + z1**2) * (y2 - y3) + (y2**2 + z2**2) * (y3 - y1) + (y3**2 + z3**2) * (y1 - y2)
    y_center = -B / (2 * A)
    z_center = -C / (2 * A)
    x_center = p1.x
    return adsk.core.Point3D.create(x_center, y_center, z_center)

def _pick_around(th, bit, lo_margin, hi_margin, hard_min=None, hard_max=None):
    import random
    if th is None or th <= 0:
       th = 1.0
    if bit == 0:
        a = hard_min
        b = th * (1.0 - 1e-3)
    else:
        a = th * (1.0 + 1e-3)
        b = hard_max
    if hard_min is not None:
        a = max(a, hard_min)
    if hard_max is not None:
        b = min(b, hard_max)
    if a > b:
        a, b = b*0.9, b
    return random.uniform(a, b)

def _rotate_around_z(pt, center, angle_rad):
    x = pt.x - center.x
    y = pt.y - center.y
    x_new = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    y_new = x * math.sin(angle_rad) + y * math.cos(angle_rad)
    return adsk.core.Point3D.create(x_new + center.x, y_new + center.y, pt.z)

def add_single_cut(design, root, main_body, L_mm, W_mm, H_mm, balancer, force_bits=None):
    try:
        import math, random
        fb = force_bits or {} 
        # progi
        td, tw, tr, ta = (balancer.thresh[k] for k in ('depth', 'width', 'radius', 'angle'))
        MIN_ANGLE_DEG_LOCAL = 1.5  # lokalny bezpiecznik (uniezależniony od globali)

        # 1) klasa "kierunkowa" (używana tylko gdy brak wymuszeń)
        want = balancer.deficit_class() if USE_DEFICIT_CLASS else balancer.random_class()
        want_d, want_w, want_c, want_a = [int(ch) for ch in want]
        def _merge_force_bits(force_bits, d, w, c, a):
            if not force_bits:
                return d, w, c, a
            # krotka/lista (d,w,c,a)
            if isinstance(force_bits, (list, tuple)) and len(force_bits) == 4:
                fd, fw, fc, fa = [int(1 if x else 0) for x in force_bits]
                return fd, fw, fc, fa
            # słownik {'D':0/1, 'W':..., 'C':..., 'A':...}
            if isinstance(force_bits, dict):
                fd, fw, fc, fa = d, w, c, a
                if force_bits.get('D') is not None: fd = int(force_bits['D'])
                if force_bits.get('W') is not None: fw = int(force_bits['W'])
                if force_bits.get('C') is not None: fc = int(force_bits['C'])
                if force_bits.get('A') is not None: fa = int(force_bits['A'])
                return fd, fw, fc, fa
            return d, w, c, a
        # zastosuj ewentualne wymuszenia
        want_d, want_w, want_c, want_a = _merge_force_bits(force_bits, want_d, want_w, want_c, want_a)

        # --- helper do wstępnego doboru parametrów tak, by spełnić wymuszone bity ---
        def _rad3(p1, p2, p3):
            (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
            a = math.hypot(x2 - x1, y2 - y1)
            b = math.hypot(x3 - x2, y3 - y2)
            c = math.hypot(x1 - x3, y1 - y3)
            s = (a + b + c) / 2.0
            area2 = max(s * (s - a) * (s - b) * (s - c), 1e-12)
            return (a * b * c) / (4.0 * math.sqrt(area2))

        def sample_params():
            """Zwraca (x_start, width_x, depth, rot_deg, sharp_k, ok_c)."""
            import math, random

            # --- W (szerokość) wokół progu ---
            width_x = _pick_around(
                tw, want_w,
                lo_margin=(0.6, 0.6), hi_margin=(0.6, 0.6),
                hard_min=WIDTH_MIN_MM, hard_max=WIDTH_MAX_MM
            )
            # dopasuj do marginesów X
            if L_mm - 2 * X_MARGIN_MM < width_x:
                width_x = max(0.4, L_mm - 2 * X_MARGIN_MM - 0.4)
            x_start = randf(X_MARGIN_MM, max(X_MARGIN_MM, L_mm - X_MARGIN_MM - width_x))

            # --- D (głębokość) wokół progu z limitem od szerokości ---
            d_cap = min(H_mm * 0.75, 0.6 * width_x, DEPTH_MAX_MM)
            depth = _pick_around(
                td, want_d,
                lo_margin=(0.6, 0.6), hi_margin=(0.6, 0.6),
                hard_min=DEPTH_MIN_MM, hard_max=DEPTH_MAX_MM
            )

            # --- A (kąt) wokół progu + losowy znak ---
            base_angle = _pick_around(
                ta, want_a,
                lo_margin=(4.0, 0.6), hi_margin=(6.0, 0.6),
                hard_min=MIN_ANGLE_DEG, hard_max=20.0
            )
            rot_deg = base_angle if random.random() < 0.5 else -base_angle
            if abs(rot_deg) < MIN_ANGLE_DEG_LOCAL:
                rot_deg = math.copysign(MIN_ANGLE_DEG_LOCAL, rot_deg if rot_deg != 0 else 1.0)

            # --- Celowanie w trudny przypadek: C=0 przy W=0 ---
            if want_c == 0 and want_w == 0:
                # trzymaj się tuż poniżej progu W, by zmaksymalizować promień przy W=0
                width_x = min(width_x, tw * random.uniform(0.990, 0.998))
                # gdy równocześnie D=0 – jeszcze płycej (większy promień)
                if want_d == 0:
                    depth = min(depth, max(DEPTH_MIN_MM, td * random.uniform(0.60, 0.90)))

            # --- C (ostrość) – bazowe profile: mniejsze k => tępiej (większy r), większe k => ostrzej (mniejszy r)
            sharp_k = random.uniform(1.6, 3.0) if want_c == 1 else random.uniform(0.02, 0.06)

            # ===== STRONG C ENFORCEMENT (gdy bias modelu wymusza C) =====
            fb_local = force_bits or {}
            if 'C' in fb_local and fb_local['C'] in (0, 1):
                want_c_fb = int(fb_local['C'])
                if want_c_fb == 0:
                    # C=0 (tępo) — dążymy do dużego promienia:
                    # 1) jeśli W=0 też jest wymuszone – przytnij szerokość poniżej progu
                    if fb_local.get('W', None) == 0:
                        width_x = min(width_x, tw * random.uniform(0.92, 0.98))
                    # 2) ogranicz głębokość przez strzałkę s_max dla r_goal = tr + margines
                    margin = max(0.05, 0.15 * tr)
                    r_goal = tr + margin
                    s_max = (width_x * width_x) / (8.0 * r_goal + 1e-9)
                    if fb_local.get('D', None) == 0:
                        depth = min(depth, max(DEPTH_MIN_MM, min(td*0.98, s_max * 0.95)))
                    else:
                        desired = max(td*1.02, DEPTH_MIN_MM)
                        if s_max < desired:
                            if fb_local.get('W', None) == 1:
                                c_req = math.sqrt(max(0.0,8.0*r_goal*desired))
                                w_geo_max = min(WIDTH_MAX_MM, max(0.4, W_mm*0.9), max(0.4, L_mm - 2*X_MARGIN_MM-0.4))
                                width_x = min(w_geo_max, max(width_x, max(tw*1.02, c_req*1.02)))
                                s_max = (width_x * width_x) / (8.0 * r_goal + 1e-9)
                        depth = min(max(depth, desired), max(td * 1.02, s_max * 0.98, DEPTH_MIN_MM))
                    # 3) bardzo „tępy” profil
                    sharp_k = random.uniform(0.02, 0.06)
                else:
                    # C=1 (ostro) — mały promień:
                    # podnieś nieco głębokość względem szerokości, by uniknąć płaskich łuków
                    wand_d_eff = int((force_bits or {}).get('D', want_d))
                    want_w_eff = (force_bits or {}).get('W', None)
                    r_goal = max(1e-6, tr*0.90)
                    s_min = (width_x * width_x) / (8.0 * r_goal)
                    if wand_d_eff == 1:
                        depth = max(depth, max(td*1.02, s_min*1.05, DEPTH_MIN_MM))
                    else:
                        depth = min(depth, td*0.98)
                        if want_w_eff in (None, 1):
                            width_x = min(width_x*1.02, max(tw*1.02, width_x))
                    sharp_k = random.uniform(1.6, 3.0)
            # ===== END STRONG C ENFORCEMENT =====

            # --- Szybkie oszacowanie promienia dla weryfikacji ok_c ---
            xs = [x_start + t * width_x for t in (0.00, 0.25, 0.50, 0.75, 1.00)]
            zs = []
            for i, t in enumerate((0.00, 0.25, 0.50, 0.75, 1.00)):
                if i in (0, 4):
                    z = H_mm
                elif i == 2:
                    z = H_mm - depth
                else:
                    base = (1.0 - (abs(t - 0.5) / 0.5) ** (1.0 / (sharp_k + 1e-3)))
                    z = H_mm - base * depth
                zs.append(z)
            radius_est = _rad3((xs[1], zs[1]), (xs[2], zs[2]), (xs[3], zs[3]))

            # Łagodniejszy margines dla (W=0 & C=0), standardowy w pozostałych przypadkach
            margin_c0 = 0.15 if (want_c == 0 and want_w == 0) else 0.30
            ok_c = (radius_est <= tr * 0.98) if (want_c == 1) else (radius_est >= max(tr * 1.02, tr + margin_c0))

            return x_start, width_x, depth, rot_deg, sharp_k, ok_c


        # spróbuj kilka razy dobrać parametry, które spełnią wymuszenia
        tries = 14 if (force_bits and force_bits.get('C') in (0,1)) else 8
        for _ in range(tries):
            x_start, width_x, depth, rot_deg, sharp_k, ok_c = sample_params()
            if (force_bits and force_bits.get('C') in (0,1)) and not ok_c:
                continue
            # delikatny „nudge” pod docelowe bity (korzystamy z *want_* już po scaleniu)
            d, w, c, a = want_d, want_w, want_c, want_a
            if (depth >= td) != bool(d):
                depth = 0.95*td if d == 0 else 1.05*td
            if (width_x >= tw) != bool(w):
                width_x = 0.95*tw if w == 0 else 1.05*tw
            if (abs(rot_deg) >= ta) != bool(a):
                rot_deg = (0.9*ta if a == 0 else 1.1*ta) * (1 if rot_deg >= 0 else -1)

            # jeśli wymuszamy C, upewnij się, że „nudge” nie popsuł ok_c — szybki re-check promienia
            if (force_bits and force_bits.get('C') in (0, 1)):
                xs = [x_start + t * width_x for t in (0.00, 0.25, 0.50, 0.75, 1.00)]
                zs = []
                for i, t in enumerate((0.00, 0.25, 0.50, 0.75, 1.00)):
                    if i in (0, 4):
                        z = H_mm
                    elif i == 2:
                        z = H_mm - depth
                    else:
                        base = (1.0 - (abs(t - 0.5) / 0.5) ** (1.0 / (sharp_k + 1e-3)))
                        z = H_mm - base * depth
                    zs.append(z)
                radius_est = _rad3((xs[1], zs[1]), (xs[2], zs[2]), (xs[3], zs[3]))
                # łagodniejszy margines dla (W=0 & C=0), standardowy gdzie indziej
                margin_c0 = 0.15 if (want_c == 0 and want_w == 0) else 0.30
                ok_c2 = (radius_est <= tr * 0.98) if want_c == 1 else (radius_est >= max(tr * 1.02, tr + margin_c0))
                if not ok_c2:
                    continue
            break

        x_end = min(L_mm - X_MARGIN_MM, x_start + width_x)
        x_mid = 0.5 * (x_start + x_end)
        y = W_mm / 2.0  # jak w poprzednim generatorze – dokładnie środek po Y
        deepest_z = H_mm - depth
        rot_rad = math.radians(rot_deg)

        # 2) Szkic SPLINE na płaszczyźnie XY w y = width/2, z końcami przy z=H, środkiem przy z=H-depth
        spline_sketch = root.sketches.add(root.xYConstructionPlane)
        spline_sketch.isComputeDeferred = True
        npts = random.randint(NUM_PTS_MIN, NUM_PTS_MAX)
        pts = []
        for i in range(npts):
            t = i/(npts-1)
            x = x_start + t*(x_end - x_start)
            if i == 0 or i == npts-1:
                z = H_mm
            elif i == npts//2:
                z = deepest_z
            else:
                # delikatne wygładzenie
                base = (1.0 - (abs(t-0.5)/0.5)**(1.0/(sharp_k+1e-3)))
                z = H_mm - base*depth
                z -= randf(0.0, 0.05*depth)
            pts.append(adsk.core.Point3D.create(mm_to_internal(design,x),
                                                mm_to_internal(design,y),
                                                mm_to_internal(design,z)))
        # Środek okręgu przez 3 punkty (YZ przy x = x_mid)
        margin = randf(0.1, 0.4)
        p_center = adsk.core.Point3D.create(mm_to_internal(design, x_mid),
                                            mm_to_internal(design, y),
                                            mm_to_internal(design, deepest_z))
        p_startY = adsk.core.Point3D.create(mm_to_internal(design, x_mid),
                                            mm_to_internal(design, 0 + margin),
                                            mm_to_internal(design, H_mm))
        p_endY   = adsk.core.Point3D.create(mm_to_internal(design, x_mid),
                                            mm_to_internal(design, W_mm - margin),
                                            mm_to_internal(design, H_mm))
        circle_center = _find_circle_center(p_startY, p_center, p_endY)
        if circle_center is None:
            if not DEBUG_KEEP_SKETCHES:
                try: spline_sketch.deleteMe()
                except: pass
            return False, None

        # Rotacja punktów profilu wokół Z o rot_deg
        rot_pts = [_rotate_around_z(pt, circle_center, rot_rad) for pt in pts]
        coll = adsk.core.ObjectCollection.create()
        for rp in rot_pts: coll.add(rp)
        spline = spline_sketch.sketchCurves.sketchFittedSplines.add(coll)

        # Domknięcie profilu „daszkiem” na wysokości z = circle_center.z
        s0 = spline.fitPoints.item(0).geometry
        s1 = spline.fitPoints.item(spline.fitPoints.count-1).geometry
        p_cap0 = adsk.core.Point3D.create(s0.x, s0.y, circle_center.z)
        p_cap1 = adsk.core.Point3D.create(s1.x, s1.y, circle_center.z)
        lines = spline_sketch.sketchCurves.sketchLines
        lines.addByTwoPoints(s0, p_cap0)
        lines.addByTwoPoints(s1, p_cap1)
        lines.addByTwoPoints(p_cap0, p_cap1)
        spline_sketch.isComputeDeferred = False

        # Wybór największego profilu
        prof = None; max_area = -1.0
        for p in spline_sketch.profiles:
            try: area = p.areaProperties().area
            except: area = -1.0
            if area > max_area:
                max_area = area; prof = p
        if prof is None or max_area <= 0:
            if not DEBUG_KEEP_SKETCHES:
                try: spline_sketch.deleteMe()
                except: pass
            return False, None

        # 3) Oś obrotu (linia równoległa do X, przechodząca przez circle_center na z=circle_center.z)
        axis_sketch = root.sketches.add(root.xYConstructionPlane)
        r_mm = internal_to_mm(design, circle_center.z) - deepest_z  # promień wierzchołka w mm
        if r_mm < 0.05: r_mm = 0.05
        ax_p0_pre = adsk.core.Point3D.create(circle_center.x - mm_to_internal(design, r_mm),
                                             circle_center.y, circle_center.z)
        ax_p1_pre = adsk.core.Point3D.create(circle_center.x + mm_to_internal(design, r_mm),
                                             circle_center.y, circle_center.z)
        ax_p0 = _rotate_around_z(ax_p0_pre, circle_center, rot_rad)
        ax_p1 = _rotate_around_z(ax_p1_pre, circle_center, rot_rad)
        axis_line = axis_sketch.sketchCurves.sketchLines.addByTwoPoints(ax_p0, ax_p1)

        # ===== 4) METRYKI=====
        # promień „ostrości” z geometrii splajnu (mediana z lokalnych okręgów 3-pkt)
        fit_pts = [spline.fitPoints.item(i).geometry for i in range(spline.fitPoints.count)]
        radii = []
        for i in range(1, len(fit_pts)-1):
            radii.append(circle_radius_mm(design, fit_pts[i-1], fit_pts[i], fit_pts[i+1]))
        radii.sort()
        radius_mm = radii[len(radii)//2] if radii else 999.0
        radius_mm = min(radius_mm, 20.0)  
        width_mm = x_end - x_start
        angle_deg = rot_deg
        klass = balancer.classify(depth, width_mm, radius_mm, angle_deg)
        metrics = dict(
            depth_mm=round(depth,3), width_mm=round(width_x,3),
            radius_mm=round(radius_mm,3), angle_deg=round(angle_deg,2),
            klass=klass, x_start_mm=round(x_start,3), x_end_mm=round(x_end,3),
            y_mid_mm=round(y,3)
        )

        expected_bits = None
        if isinstance(force_bits, dict) and all(k in force_bits for k in ("D","W","C","A")):
            expected_bits = f"{int(force_bits['D'])}{int(force_bits['W'])}{int(force_bits['C'])}{int(force_bits['A'])}"
        if expected_bits is not None and klass != expected_bits:
            # sprzątamy szkice i WYCHODZIMY bez cięcia
            if not DEBUG_KEEP_SKETCHES:
                try: spline_sketch.deleteMe()
                except: pass
                try: axis_sketch.deleteMe()
                except: pass
            return False, None
        


        #4) REOLVE 360° → NewBody → COMBINE(Cut)
        rev_in = root.features.revolveFeatures.createInput(
            prof, axis_line, adsk.fusion.FeatureOperations.NewBodyFeatureOperation
        )
        rev_in.setAngleExtent(False, adsk.core.ValueInput.createByString('360 deg'))
        rev = root.features.revolveFeatures.add(rev_in)
        tool_body = rev.bodies.item(0)

        # Odejmowanie
        tools = adsk.core.ObjectCollection.create(); tools.add(tool_body)
        comb = root.features.combineFeatures
        cin = comb.createInput(main_body, tools)
        cin.operation = adsk.fusion.FeatureOperations.CutFeatureOperation
        cin.isKeepToolBodies = False
        comb.add(cin)

        # Sprzątanie
        if not DEBUG_KEEP_SKETCHES:
            try: spline_sketch.deleteMe()
            except: pass
            try: axis_sketch.deleteMe()
            except: pass

        balancer.push_sample(depth, width_mm, radius_mm, angle_deg, klass)
        balancer.maybe_recalc(BALANCE_MODE)
        return True, metrics

    except Exception as e:
        return False, None



# ================================
# Eksport top-surface (zbiera WSZYSTKIE najwyższe ściany)
# ================================
def export_top_surface_thin(design, root, main_body, out_path):
    """
    Stabilny eksport górnego 'plastra' (TOP_THICKNESS_MM) – bez kopiowania bryły.
    1) Robimy slab (cienki prostopadłościan) tuż pod z_max.
    2) Combine: Intersect ze slabem jako TARGET i main_body jako TOOL (keep tools).
    3) Exportujemy zmodyfikowany slab jako _top.stl.
    """
    import adsk.core, adsk.fusion
    try:
        thickness_cm = mm_to_internal(design, TOP_THICKNESS_MM)
        eps_cm       = mm_to_internal(design, 0.02)

        bb = main_body.boundingBox
        z_max = bb.maxPoint.z
        z_mid = z_max - thickness_cm*0.5

        # Płaszczyzna szkicu na z_mid
        planes = root.constructionPlanes
        pi = planes.createInput()
        pi.setByOffset(root.xYConstructionPlane, adsk.core.ValueInput.createByReal(z_mid))
        pl = planes.add(pi)

        # Prostokąt większy od BB (margines 2 mm)
        margin_cm = mm_to_internal(design, 2.0)
        minx, miny = bb.minPoint.x - margin_cm, bb.minPoint.y - margin_cm
        maxx, maxy = bb.maxPoint.x + margin_cm, bb.maxPoint.y + margin_cm

        sk = root.sketches.add(pl)
        sk.isComputeDeferred = True
        p1 = adsk.core.Point3D.create(minx, miny, 0)
        p2 = adsk.core.Point3D.create(maxx, maxy, 0)
        sk.sketchCurves.sketchLines.addTwoPointRectangle(p1, p2)
        sk.isComputeDeferred = False

        prof = sk.profiles.item(0)
        if not prof:
            try: sk.deleteMe(); pl.deleteMe()
            except: pass
            #log("export_top_surface_thin: no profile"); return False

        # Wyciągnij slab (symetrycznie)
        exts = root.features.extrudeFeatures
        exi = exts.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        half = thickness_cm*0.5 + eps_cm
        exi.setSymmetricExtent(adsk.core.ValueInput.createByReal(half), False)
        ext = exts.add(exi)
        slab_body = ext.bodies.item(0)
        if not slab_body:
            try: sk.deleteMe(); pl.deleteMe()
            except: pass
            #log("export_top_surface_thin: no slab"); return False

        # Intersect: TARGET = slab, TOOL = main_body (keep tools)
        tools = adsk.core.ObjectCollection.create(); tools.add(main_body)
        comb = root.features.combineFeatures
        cin = comb.createInput(slab_body, tools)
        cin.operation = adsk.fusion.FeatureOperations.IntersectFeatureOperation
        cin.isKeepToolBodies = True
        comb.add(cin)


        # Eksport
        exp = design.exportManager
        opts = exp.createSTLExportOptions(slab_body, out_path)
        opts.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementLow
        exp.execute(opts)

        # Sprzątanie
        try: slab_body.deleteMe()
        except: pass
        try: sk.deleteMe()
        except: pass
        try: pl.deleteMe()
        except: pass

        return True
    except Exception as e:
        #log(f"EXC in export_top_surface_thin: {e}")
        return False



# =========================================================
# GŁÓWNA PĘTLA
# =========================================================

def run(context):

    import time
    app = adsk.core.Application.get()
    ui  = app.userInterface
    random.seed(SEED)

    try:
        bal = GlobalBalancer()
        model_bal = ModelBalancer()
        os.makedirs(EXPORT_DIR, exist_ok=True)
        os.makedirs(TOP_SURFACE_DIR, exist_ok=True)
        os.makedirs(CSV_DIR, exist_ok=True)

        cuts_csv   = os.path.join(CSV_DIR, 'cuts_data.csv')
        models_csv = os.path.join(CSV_DIR, 'models_data.csv')

        if RESUME_FROM_CSV:
            START_MODEL_INDEX = auto_start_from_csv(models_csv)
            MODEL_TARGET_OFFSET = compute_rr_offset_from_models_csv(models_csv, skip_forbidden=FORBIDDEN_CLASSES, fallback=0, prefer_col="ModelClassID_0_15")
            loaded_thr = load_dwca_thresholds(THR_PATH) if AUTO_LOAD_THRESHOLDS else None
            if loaded_thr:
                bal.set_thresholds(loaded_thr)
                bal.locked = True
                if SKIP_WARMUP_IF_LOADED:
                    WARMUP_MODELS = 0
        
        cuts_new = (not os.path.exists(cuts_csv)) or (os.path.getsize(cuts_csv) == 0)
        models_new = (not os.path.exists(models_csv)) or (os.path.getsize(models_csv) == 0)

        cuts_f = open(cuts_csv, 'a', newline='', encoding='utf-8')
        cuts_w = csv.writer(cuts_f)
        models_f = open(models_csv, 'a', newline='', encoding='utf-8')
        models_w = csv.writer(models_f)

        if cuts_new:
            cuts_w.writerow([
                'Model_ID','Cut_Index','CutClass4DWA','ModelClass4DWA','ModelClassID_0_15',
                'Depth_mm','WidthX_mm','ApexRadius_mm','Angle_deg',
                'x_start_mm','x_end_mm','y_mid_mm','L_mm','W_mm','H_mm'
            ])
        if models_new:
            models_w.writerow([
                'Model_ID','L_mm','W_mm','H_mm','NumCuts',
                'ModelClass4DWA','ModelClassID_0_15',
                'D_med_mm','W_med_mm','C_med_mm','A_med_deg',
                'D_share','W_share','C_share','A_share', 'TargetID', 'TargetBits',
                'TopSurfaceSTL','ModelSTL'
            ])


        # round-robin po 16 klasach po warm-upie (z opcją pomijania FORBIDDEN_CLASSES)
        schedule = build_round_robin_schedule(
            NUMBER_OF_MODELS, WARMUP_MODELS,
            offset=MODEL_TARGET_OFFSET,
            skip_forbidden=FORBIDDEN_CLASSES if SKIP_FORBIDDEN_IN_TARGET else None
        )




        for model_idx in range(START_MODEL_INDEX, START_MODEL_INDEX + NUMBER_OF_MODELS):
            adsk.doEvents()
            # nowy, tymczasowy dokument (czyści timeline/mem)
            temp_doc = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
            design = adsk.fusion.Design.cast(app.activeProduct)
            root = design.rootComponent
            global USE_DEFICIT_CLASS
            USE_DEFICIT_CLASS = (model_idx > WARMUP_MODELS)

            try:
                design.designType = adsk.fusion.DesignTypes.DirectDesignType
            except:
                pass

            try:
                body, L, W, H = build_base_block(design, root)
                if DISABLE_PREVIEW:
                    for i in range(root.bRepBodies.count):
                        try: root.bRepBodies.item(i).isVisible = False
                        except: pass

                if model_idx <= WARMUP_MODELS:
                    target_id = None
                else:
                    idx_in_batch = (model_idx - START_MODEL_INDEX - WARMUP_MODELS)
                    if 0 <= idx_in_batch < len(schedule):
                        target_id = schedule[idx_in_batch]
                    else:
                        target_id = schedule[idx_in_batch % len(schedule)] if schedule else None

                target_bits = "" if target_id is None else bits4(target_id)


                if target_id is None:
                    fb_target = None
                else:
                    fb_target = {
                        'D': ((target_id >> 3) & 1),
                        'W': ((target_id >> 2) & 1),
                        'C': ((target_id >> 1) & 1),
                        'A': ( target_id       & 1)
                    }

                # -- generowanie cięć dla modelu z biasem do celu (albo bez celu w warm-upie) --
                if STRICT_CUTS_TO_TARGET and target_bits:
                    cut_metrics, bit_counts, ok_strict = drive_cuts_strict_to_target(
                        design, root, body, L, W, H, bal,
                        target_bits=target_bits,
                        min_cuts=CUTS_PER_MODEL_MIN,
                        max_cuts=CUTS_PER_MODEL_MAX
                    )
                    if not ok_strict:
                        # Jeżeli klasa jest niemożliwa albo nie dojechaliśmy do min_cuts:
                        # Przykład fallbacku:
                        if not cut_metrics:
                            cut_metrics, bit_counts = drive_cuts_for_model(
                                design, root, body, L, W, H, bal,
                                target_bits="",  # brak celu -> bez biasu
                                min_cuts=CUTS_PER_MODEL_MIN,
                                max_cuts=CUTS_PER_MODEL_MAX,
                                share_thresh=SHARE_THRESH
                            )
                else:
                    # dotychczasowy tryb "miękki"
                    cut_metrics, bit_counts = drive_cuts_for_model(
                        design, root, body, L, W, H, bal,
                        target_bits=target_bits,
                        min_cuts=CUTS_PER_MODEL_MIN,
                        max_cuts=CUTS_PER_MODEL_MAX,
                        share_thresh=SHARE_THRESH
                    )



                
                bal.maybe_recalc(BALANCE_MODE, force=False)

                model_klass, d_med, w_med, r_med, a_med, shareD, shareW, shareC, shareA = compute_model_class(cut_metrics, bal)
                
                if DISABLE_PREVIEW:
                    for i in range(root.bRepBodies.count):
                        try: root.bRepBodies.item(i).isVisible = True
                        except: pass

                 # --- eksport modelu ---
                model_path = os.path.join(EXPORT_DIR, f"model_{model_idx:04d}.stl")
                exp = design.exportManager
                stl = exp.createSTLExportOptions(body, model_path)
                stl.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementLow
                exp.execute(stl)

                # --- eksport górnego plastra (Twoja wersja z Intersect slab) ---
                top_path = os.path.join(TOP_SURFACE_DIR, f"model_{model_idx:04d}_top.stl")
                top_ok = export_top_surface_thin(design, root, body, top_path)
                if not top_ok:
                    top_path = ""
            
                # --- klasa modelu i zapis CSV ---
                

                model_klass, d_med, w_med, r_med, a_med, shareD, shareW, shareC, shareA = compute_model_class(cut_metrics, bal)
                model_class_id = int(model_klass, 2)
                model_bal.push(model_class_id)
                model_klass_excel = f"'{model_klass}"

                # cuty (z dopisaną klasą modelu)
                for cut_i, m in enumerate(cut_metrics, start=1):
                    cut_class_excel = f"'{m['klass']}"
                    cuts_w.writerow([
                        model_idx, cut_i, cut_class_excel, model_klass_excel, model_class_id,
                        m['depth_mm'], m['width_mm'], m['radius_mm'], m['angle_deg'],
                        m['x_start_mm'], m['x_end_mm'], m['y_mid_mm'],
                        round(L,3), round(W,3), round(H,3)
                    ])

                # wiersz modelu
                models_w.writerow([model_idx, round(L,3), round(W,3), round(H,3),
                                    len(cut_metrics), model_klass_excel, model_class_id, round(d_med,3),
                                     round(w_med,3), round(r_med,3), round(a_med,3),
                                      round(shareD,3), round(shareW,3), round(shareC,3), round(shareA,3), ("" if target_id is None else target_id), ("" if not target_bits else target_bits),
                                       top_path, model_path])
                cuts_f.flush()
                models_f.flush()

            finally:
                # zamknij tymczasowy dokument

                app.activeDocument.close(False)
                adsk.doEvents()
                time.sleep(0.01)
            if FREEZE_AFTER_WARMUP and model_idx == WARMUP_MODELS and not bal.locked:
                bal.recalc_now(BALANCE_MODE)
                apply_radius_threshold_bias(bal)
                bal.locked = True
                save_dwca_thresholds(THR_PATH, bal.thresh)
                log(f"Balancer locked after {WARMUP_MODELS} models: {bal.thresh}")


        # === PODSUMOWANIE: histogram klas modeli (0..15) ===
        from collections import defaultdict

        # 1) policz
        counts = defaultdict(int, getattr(model_bal, "counts", {}))
        total_models = sum(counts.values())
        expected = (total_models / 16.0) if total_models else 0.0
        max_dev = 0.0
        if total_models:
            max_dev = max(abs(counts.get(k,0) - expected) / (expected if expected else 1.0) for k in range(16))

        # 2) zapisz CSV
        hist_csv = os.path.join(CSV_DIR, 'model_class_histogram.csv')
        with open(hist_csv, 'w', newline='', encoding='utf-8') as f_hist:
            w = csv.writer(f_hist)
            w.writerow(['ClassID_0_15', 'Bits(DWCA)', 'Count', 'Share'])
            for k in range(16):
                cnt = counts.get(k, 0)
                w.writerow([k, f"{k:04b}", cnt, f"{(cnt/total_models):.4f}" if total_models else "0.0000"])

        # 3) top 4 nadmiarowe/niedomiarowe
        least = sorted(range(16), key=lambda k: counts.get(k,0))[:4]
        most  = sorted(range(16), key=lambda k: counts.get(k,0), reverse=True)[:4]
        least_str = ", ".join(f"{k:02d}({k:04b})={counts.get(k,0)}" for k in least)
        most_str  = ", ".join(f"{k:02d}({k:04b})={counts.get(k,0)}" for k in most)

        # 4) finalny komunikat (progi + histogram + ścieżki)
        ui.messageBox(
            "Zakończono.\n"
            f"Progi (mm/deg): D={bal.thresh['depth']:.3f}, W={bal.thresh['width']:.3f}, "
            f"R={bal.thresh['radius']:.3f}, A={bal.thresh['angle']:.2f}\n"
            f"Modele: {total_models} | Max odchyłka vs równy rozkład: {max_dev*100:.1f}%\n"
            f"Najmniej: {least_str}\nNajwięcej: {most_str}\n"
            f"Histogram CSV: {hist_csv}\n"
            f"Pliki STL/CSV: {EXPORT_DIR}, {TOP_SURFACE_DIR}, {CSV_DIR}"
        )
   
        cuts_f.close()
        models_f.close()
    except Exception as e:
        try:
            if cuts_f and not cuts_f.closed: cuts_f.close()
            if models_f and not models_f.closed: models_f.close()
        except: pass
        if ui:
            ui.messageBox(f"Błąd: {str(e)}\n{traceback.format_exc()}")
