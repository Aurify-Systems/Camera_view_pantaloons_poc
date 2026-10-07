import sys, logging
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
import fixture_missing as fm
logging.getLogger("fixture_missing").setLevel(logging.WARNING)

# ---------------------------------------------------------------
# USAGE
#   python test_angle_detection.py
#       -> runs the built-in synthetic self-test (lighting, person, shifts, tilt, zoom, black, blur)
#   python test_angle_detection.py original.jpg current.jpg
#       -> compares two REAL images with the same logic and settings as fixture_missing.py
#          (uses the "fixture_missing" block of your config, so you can tune thresholds)
# ---------------------------------------------------------------
if len(sys.argv) == 3:
    logging.getLogger("fixture_missing").setLevel(logging.INFO)
    a, b = cv2.imread(sys.argv[1]), cv2.imread(sys.argv[2])
    if a is None or b is None:
        sys.exit("Could not read one of the images")
    cfg = fm.load_config().get("fixture_missing", {}) if fm.CONFIG_FILE.exists() else {}
    res = fm.detect_camera_angle_change(a, b, cfg)
    print("\nRESULT:", "ALERT YES" if res["camera_angle_changed"] else "ALERT NO",
          "| movement =", res["movement_type"], "| reason =", res["reason"])
    sys.exit(0)


rng = np.random.default_rng(1)
H, W = 1080, 960
def scene(seed=1):
    r = np.random.default_rng(seed)
    img = np.full((H, W, 3), 120, np.uint8)
    for y in range(0, H, 6):  # gradient wall
        img[y:y+6] = 90 + int(60*y/H)
    for _ in range(140):
        x, y = int(r.integers(0, W-80)), int(r.integers(0, H-80))
        w, h = int(r.integers(20, 160)), int(r.integers(20, 160))
        c = tuple(int(v) for v in r.integers(20, 235, 3))
        cv2.rectangle(img, (x, y), (min(W-1,x+w), min(H-1,y+h)), c, -1 if r.random()<.6 else 2)
    for _ in range(60):
        cv2.putText(img, "SALE%d" % r.integers(10,99), (int(r.integers(0,W-150)), int(r.integers(30,H-10))),
                    cv2.FONT_HERSHEY_SIMPLEX, float(r.uniform(.6,1.6)), (255,255,255), 2)
    return cv2.GaussianBlur(img, (3,3), 0)

def noise(img, s=3):
    n = np.random.default_rng(5).normal(0, s, img.shape)
    return np.clip(img.astype(np.float32)+n, 0, 255).astype(np.uint8)

def affine(img, dx=0, dy=0, rot=0, scale=1.0):
    M = cv2.getRotationMatrix2D((W/2, H/2), rot, scale); M[0,2]+=dx; M[1,2]+=dy
    return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)

base = scene()
orig = noise(base, 2)
S = dict(fm.load_config().get("fixture_missing", {})) if fm.CONFIG_FILE.exists() else {}
S = {}  # defaults only

def person(img, frac=0.12):
    out = img.copy()
    w = int(W*0.25); h = int(H*frac*4)
    cv2.rectangle(out, (300, 400), (300+w, 400+h), (30, 40, 90), -1)
    return out
def products(img):
    out = img.copy()
    for i in range(12):
        cv2.rectangle(out, (100+i*60, 700), (140+i*60, 800), (200-i*10, 50+i*15, 90), -1)
    return out
def light(img, k, add=0): return np.clip(img.astype(np.float32)*k+add, 0, 255).astype(np.uint8)

cases = [
 ("same+noise", noise(base, 3), False),
 ("lights dim 0.6", noise(light(base, 0.6), 3), False),
 ("lights bright 1.4 +20", noise(light(base, 1.4, 20), 3), False),
 ("lights very dim 0.3", noise(light(base, 0.3), 3), False),
 ("person in frame", noise(person(base), 3), False),
 ("products changed", noise(products(base), 3), False),
 ("person+dim", noise(light(person(base),0.7), 3), False),
 ("shift +3px x", noise(affine(base, dx=3), 3), True),
 ("shift +2.6px x", noise(affine(base, dx=2.6), 3), True),
 ("shift 1px (noise level)", noise(affine(base, dx=1), 3), False),
 ("same, different noise", noise(base, 6), False),
 ("shift +6px y", noise(affine(base, dy=6), 3), True),
 ("shift -10px x", noise(affine(base, dx=-10), 3), True),
 ("rot 0.6deg", noise(affine(base, rot=0.6), 3), True),
 ("zoom 1.015", noise(affine(base, scale=1.015), 3), True),
 ("shift 5px + person", noise(person(affine(base, dx=5)), 3), True),
 ("shift 5px + dim", noise(light(affine(base, dx=5),0.6), 3), True),
 ("black", np.zeros_like(base), True),
 ("blur", cv2.GaussianBlur(noise(base,3), (0,0), 4), True),
 ("totally different view", noise(scene(9), 3), True),
]
bad = 0
for name, img, expect in cases:
    r = fm.detect_camera_angle_change(orig, img, S)
    ok = r["camera_angle_changed"] == expect
    bad += not ok
    reg = r["registration"]
    print(f"{'OK ' if ok else 'BAD'} {name:26s} -> {r['movement_type']:13s} edge={r['edge_mismatch_percentage']:5.1f} cells={r['changed_cell_count']:2d}",
          f"shift=({reg['shift_x']:.1f},{reg['shift_y']:.1f}) rot={reg['rotation_deg']:.2f} sc={reg['scale_pct']:.2f} inl={reg['inliers']}" if reg else "noreg")
print("failures:", bad)


# ---------------------------------------------------------------
# HARD SCENES: repetitive shelves, low texture, LARGE pans
# (this is what defeats plain keypoint matching on store cameras)
# ---------------------------------------------------------------
def shelf_scene(seed=3):
    r = np.random.default_rng(seed)
    img = np.full((H, W, 3), 175, np.uint8)
    for row in range(5):                       # 5 shelves
        y = 120 + row * 190
        cv2.rectangle(img, (0, y + 150), (W, y + 165), (90, 90, 95), -1)       # shelf edge
        for col in range(11):                  # repeating identical boxes
            x = 25 + col * 85
            cv2.rectangle(img, (x, y + 30), (x + 70, y + 150), (205, 205, 210), -1)
            cv2.rectangle(img, (x, y + 30), (x + 70, y + 150), (120, 120, 125), 2)
            cv2.rectangle(img, (x + 10, y + 50), (x + 60, y + 80), (60, 60, 160), -1)
        for _ in range(3):                     # a few unique products
            x = int(r.integers(10, W - 90))
            cv2.rectangle(img, (x, y + 40), (x + 60, y + 148), tuple(int(v) for v in r.integers(30, 220, 3)), -1)
    return cv2.GaussianBlur(img, (5, 5), 0)

shelf = shelf_scene()
shelf_orig = noise(shelf, 2)
hard = [
 ("[shelf] same, dimmer + noise", noise(light(shelf, 0.7), 4), False),
 ("[shelf] person in front", noise(person(shelf), 3), False),
 ("[shelf] shift 4px", noise(affine(shelf, dx=4), 3), True),
 ("[shelf] shift 40px left", noise(affine(shelf, dx=40), 3), True),
 ("[shelf] LARGE pan 150px", noise(affine(shelf, dx=150), 3), True),
 ("[shelf] LARGE pan -220px", noise(affine(shelf, dx=-220), 3), True),
 ("[shelf] tilt up 90px", noise(affine(shelf, dy=90), 3), True),
 ("[shelf] pan 150px + dim", noise(light(affine(shelf, dx=150), 0.6), 3), True),
 ("[shelf] zoom 1.03", noise(affine(shelf, scale=1.03), 3), True),
]
bad2 = 0
print("\n--- hard scenes ---")
for name, img, expect in hard:
    r = fm.detect_camera_angle_change(shelf_orig, img, {})
    ok = r["camera_angle_changed"] == expect
    bad2 += not ok
    reg, tm = r["registration"], r["template"]
    print(f"{'OK ' if ok else 'BAD'} {name:32s} -> {r['movement_type']:13s} {r['reason']:38s}",
          f"reg={reg['method']}({reg['shift_x']:.1f},{reg['shift_y']:.1f})" if reg else "reg=None",
          f"tm={tm['score']:.2f}/{tm['score_zero']:.2f}")
print("hard-scene failures:", bad2)