"""
Interface Gradio - Détection de maladies sur feuilles de plantes.

Charge le modèle entraîné (ResNet18 fine-tuné) et propose une interface
simple : upload d'une photo -> prédiction des maladies les plus probables.

Lancement local : python app.py
Déploiement : ce fichier est compatible tel quel avec Hugging Face Spaces
(gr.Interface).
"""

from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "models" / "best_model.pth"
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TOP_K = 3

# Traduction "langage humain" des labels bruts du dataset.
# A completer/affiner librement selon les classes réellement utilisées.
LABEL_TRANSLATIONS = {
    "Pepper__bell___Bacterial_spot": "Poivron — Tache bactérienne",
    "Pepper__bell___healthy": "Poivron — Sain",
    "Potato___Early_blight": "Pomme de terre — Mildiou précoce (Alternariose)",
    "Potato___Late_blight": "Pomme de terre — Mildiou tardif",
    "Potato___healthy": "Pomme de terre — Sain",
    "Tomato_Bacterial_spot": "Tomate — Tache bactérienne",
    "Tomato_Early_blight": "Tomate — Mildiou précoce (Alternariose)",
    "Tomato_Late_blight": "Tomate — Mildiou tardif",
    "Tomato_Leaf_Mold": "Tomate — Moisissure des feuilles",
    "Tomato_Septoria_leaf_spot": "Tomate — Septoriose",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomate — Acariens (tétranyques)",
    "Tomato__Target_Spot": "Tomate — Tache cible (Corynespora)",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomate — Virus des feuilles jaunes en cuillère",
    "Tomato__Tomato_mosaic_virus": "Tomate — Virus de la mosaïque",
    "Tomato_healthy": "Tomate — Sain",
}

DISCLAIMER = (
    "⚠️ **Limites du modèle** : entraîné sur le dataset PlantVillage, "
    "constitué de photos prises en conditions de laboratoire (fond uniforme, "
    "feuille isolée). Les performances peuvent être moins fiables sur des "
    "photos prises en extérieur, avec un fond complexe ou un éclairage "
    "variable. Cet outil est une démonstration technique, pas un outil de "
    "diagnostic agricole professionnel."
)

# ---------------------------------------------------------------------------
# Chargement du modèle (une seule fois, au démarrage de l'app)
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device)
classes = checkpoint["classes"]
img_size = checkpoint.get("img_size", IMG_SIZE)

model = timm.create_model("resnet18", pretrained=False, num_classes=len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Fonction de prédiction
# ---------------------------------------------------------------------------

def predict(image):
    """Prend une image PIL en entrée, retourne un dict {label: proba} pour gr.Label."""
    if image is None:
        return None

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze(0).cpu()

    top_probs, top_idx = torch.topk(probs, k=min(TOP_K, len(classes)))

    results = {}
    for prob, idx in zip(top_probs, top_idx):
        raw_label = classes[idx.item()]
        human_label = LABEL_TRANSLATIONS.get(raw_label, raw_label)
        results[human_label] = float(prob)

    return results


# ---------------------------------------------------------------------------
# Interface Gradio
# ---------------------------------------------------------------------------

with gr.Blocks(title="Détection de maladies sur feuilles de plantes") as demo:
    gr.Markdown(
        """
        # 🌿 Détection de maladies sur feuilles de plantes

        Uploade une photo d'une feuille (tomate, poivron ou pomme de terre) pour identifier
        une éventuelle maladie. Le modèle retourne les hypothèses les plus probables avec
        leur niveau de confiance.
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Photo de la feuille")
            predict_btn = gr.Button("Analyser", variant="primary")
        with gr.Column():
            output_label = gr.Label(num_top_classes=TOP_K, label="Prédictions")

    gr.Markdown(DISCLAIMER)

    predict_btn.click(fn=predict, inputs=image_input, outputs=output_label)
    image_input.change(fn=predict, inputs=image_input, outputs=output_label)


if __name__ == "__main__":
    import os

    # Render fournit le port via la variable d'environnement PORT.
    # En local (pas de variable PORT), on retombe sur le port par défaut de Gradio.
    port = int(os.environ.get("PORT", 4000 ))
    demo.launch(server_name="0.0.0.0", server_port=port)