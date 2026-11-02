"""Entraînement : Pipeline + augmentation de données + recherche d'hyperparamètres.

Usage :  python -m digits_studio.training
"""
import time
import joblib
import numpy as np
from scipy import ndimage as ndi
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from . import MODEL_PATH
from .vision import to_8x8

RNG = np.random.default_rng(42)
SEED = 42


def augment(img8, rng):
    """Une variante 'manuscrite' d'une image 8x8 : rotation, échelle, décalage, épaisseur, bruit."""
    a = np.deg2rad(rng.uniform(-15, 15))
    s = rng.uniform(0.85, 1.15)
    shear = rng.uniform(-0.15, 0.15)
    m = np.array([[np.cos(a), -np.sin(a) + shear], [np.sin(a), np.cos(a)]]) / s
    c = np.array([3.5, 3.5])
    off = c - m @ c + rng.uniform(-1.0, 1.0, 2)
    out = ndi.affine_transform(img8, m, offset=off, order=1, mode="constant")
    r = rng.random()
    if r < 0.25:
        out = ndi.grey_dilation(out, size=(2, 2))          # trait plus épais
    elif r < 0.45:
        out = ndi.grey_erosion(out, size=(2, 2)) * 1.2     # trait plus fin
    out = ndi.gaussian_filter(out, rng.uniform(0, 0.6))
    out += rng.normal(0, 0.4, out.shape)
    out = np.clip(out, 0, 16)
    return out / max(out.max(), 1e-6) * 16


def build_augmented(X, y, copies, rng):
    Xs, ys, gs = [], [], []
    for i, (x, t) in enumerate(zip(X, y)):
        img = x.reshape(8, 8)
        Xs.append(x); ys.append(t); gs.append(i)
        for _ in range(copies):
            Xs.append(augment(img, rng).ravel()); ys.append(t); gs.append(i)
    return np.array(Xs), np.array(ys), np.array(gs)


def handwritten_like(img8, rng):
    """Simule un vrai tracé : grand dessin déformé, passé par le vrai prétraitement du canvas."""
    big = ndi.zoom(img8, rng.uniform(14, 24), order=3)
    big = np.clip(big / 16, 0, 1)
    ang = rng.uniform(-18, 18)
    big = ndi.rotate(big, ang, reshape=True, order=1)
    big = ndi.gaussian_filter(big, rng.uniform(1, 3))
    big = np.clip(big * rng.uniform(1.5, 3), 0, 1)
    if rng.random() < 0.5:
        big = ndi.grey_dilation(big, size=(rng.integers(3, 9),) * 2)
    box = (0, big.shape[1], 0, big.shape[0])
    return to_8x8(big, box, margin=rng.uniform(0.08, 0.2),
                  target_stroke=rng.choice([None, 0.10, 0.12, 0.16])).ravel()


def make_pipe(cls_kwargs=None):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", "passthrough"),
        ("mlp", MLPClassifier(max_iter=400, early_stopping=True, random_state=SEED,
                              **(cls_kwargs or {}))),
    ])


def main():
    X, y = load_digits(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    # Jeu de test "manuscrit simulé" (plus dur que le test d'origine)
    X_hw = np.array([handwritten_like(x.reshape(8, 8), RNG) for x in X_te])

    # Référence : modèle sans augmentation
    base = make_pipe().fit(X_tr, y_tr)
    print(f"[Sans augmentation] test original : {base.score(X_te, y_te):.3f} | "
          f"manuscrit simulé : {base.score(X_hw, y_te):.3f}")

    # Données augmentées (les copies d'une même image restent dans le même pli de CV)
    Xa, ya, ga = build_augmented(X_tr, y_tr, copies=6, rng=RNG)
    # exemples "photo" : grande image déformée -> vrai prétraitement -> 8x8
    Xh, yh, gh = [], [], []
    for i, (x, t) in enumerate(zip(X_tr, y_tr)):
        for _ in range(6):
            Xh.append(handwritten_like(x.reshape(8, 8), RNG)); yh.append(t); gh.append(i)
    Xa = np.vstack([Xa, Xh]); ya = np.concatenate([ya, yh]); ga = np.concatenate([ga, gh])
    print(f"Jeu d'entraînement augmenté : {len(Xa)} échantillons")

    param_dist = {
        "scaler": [StandardScaler(), MinMaxScaler()],
        "pca": ["passthrough", PCA(n_components=0.98)],
        "mlp__hidden_layer_sizes": [(128,), (256,), (128, 64), (256, 128), (256, 128, 64)],
        "mlp__activation": ["relu", "tanh"],
        "mlp__solver": ["adam"],
        "mlp__alpha": [1e-4, 1e-3, 1e-2],
        "mlp__learning_rate_init": [1e-3, 3e-3],
    }
    t0 = time.time()
    search = RandomizedSearchCV(make_pipe(), param_dist, n_iter=8, cv=GroupKFold(3),
                                scoring="accuracy", n_jobs=-1, random_state=SEED, verbose=1)
    search.fit(Xa, ya, groups=ga)
    print(f"Recherche terminée en {time.time() - t0:.0f}s")
    print("Meilleurs hyperparamètres :")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    print(f"Accuracy CV (augmenté) : {search.best_score_:.3f}")

    best = search.best_estimator_
    print(f"\n[Avec augmentation] test original : {best.score(X_te, y_te):.3f} | "
          f"manuscrit simulé : {best.score(X_hw, y_te):.3f}")
    print(classification_report(y_te, best.predict(X_te)))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    print(f"Modèle sauvegardé : {MODEL_PATH}")


if __name__ == "__main__":
    main()
