import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from brainomni.model import BrainOmni
from braintokenizer.model import BrainTokenizer
from braindecode.datautil import load_concat_dataset
from einops import rearrange
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

LATENT_SPACE_DIR = "./latent_space_collection"
PLOTS_DIR = "./umap_plots"

# load braintokenizer model
def get_braintokenizer(ckpt_path) -> BrainTokenizer:
    model_config_path = os.path.join(ckpt_path, "model_cfg.json")
    with open(model_config_path) as f:
        model_config = json.load(f)
    model = BrainTokenizer(**model_config)
    checkpoint = torch.load(
        os.path.join(ckpt_path, "BrainTokenizer.pt"), map_location="cpu",weights_only=True
    )
    model.load_state_dict(checkpoint, strict=False)
    return model

# load brainomni model
def get_brainomni(ckpt_path) -> BrainOmni:
    model_config_path = os.path.join(ckpt_path, "model_cfg.json")
    with open(model_config_path) as f:
        model_config = json.load(f)
    model = BrainOmni(**model_config)
    checkpoint = torch.load(os.path.join(ckpt_path, "BrainOmni.pt"), map_location="cpu",weights_only=True)
    model.load_state_dict(checkpoint, strict=False)
    # when using omni, freeze tokenizer
    for p in model.tokenizer.parameters():
        p.requires_grad = False
    return model

def gen_avg_latent_space(windows_path, z_score = False) -> str:
    LATENT_SPACE_FILE = ""
    if(z_score): LATENT_SPACE_FILE = "latent_space_avg_with_z_score.pt"
    else: LATENT_SPACE_FILE = "latent_space_avg_without_z_score.pt"

    file_path = Path(f"{LATENT_SPACE_DIR}/{LATENT_SPACE_FILE}")
    if file_path.exists(): return file_path
    
    # O diretório onde você salvou as janelas (ex: OUTPUT_DIR = "./pre-processed-pyprep-eeg")
    loaded_windows = load_concat_dataset(
        path=windows_path,
        preload=False  # load by demand (lazy loading), saves RAM
    )

    # 2. Configurar o dispositivo e carregar o BrainOmni pré-treinado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_brainomni('ckpt_collection/base') # Inicialize ou carregue o checkpoint do BrainOmni
    model.to(device)
    model.eval()

    subject_embeddings = []

    with torch.no_grad():
        for dataset_idx, window_ds in enumerate(loaded_windows.datasets):
            # Pega o ch_pos original [64, 3], adiciona o batch -> [1, 64, 3]
            pos_3d = torch.tensor(window_ds.ch_pos, dtype=torch.float32).unsqueeze(0).to(device)
            # Cria um tensor de zeros para as 3 dimensões faltantes [1, 64, 3] e concatena na última dimensão (dim=-1)
            pos = torch.cat([pos_3d, torch.zeros_like(pos_3d)], dim=-1)
            
            print(f"\nGenerating latent space for subject {dataset_idx+1}/{len(loaded_windows.datasets)}\n")
            subject_windows = []
            for idx in range(len(window_ds)):
                x, y, _ = window_ds[idx]
                # CORREÇÃO: Normaliza o sinal por canal (Z-score) para adequar à faixa esperada pelo modelo
                if(z_score): x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)
                # Adiciona dimensão de batch se necessário: [Channels, Time] -> [1, Channels, Time]
                inputs = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
                # Sensor type: 0 = EEG
                sensor_type = torch.zeros((1, x.shape[0]), dtype=torch.long).to(device)

                # Extrai apenas a saída do espaço latente (embeddings/tokens)
                # (O nome do método de extração pode variar dependendo da API do BrainOmni, ex: model.encode(inputs))
                latent_space = model.encode(inputs, pos, sensor_type)
                subject_windows.append(latent_space.cpu())

                if(not idx%500):
                    print(f"{idx+1}/{len(window_ds)}")

            # Concatena as janelas DESTE sujeito específico e tira a média global
            subject_tensor = torch.cat(subject_windows, dim=0).mean(dim=[0, 1]) # Resulta em um vetor 1D de características
            subject_embeddings.append(subject_tensor)
        
    # Matriz final com uma linha por indivíduo
    final_features = torch.stack(subject_embeddings)

    batch_size = final_features.size(0)
    final_features = final_features.reshape(batch_size, -1)

    print("Formato final para o UMAP (por indivíduo):", final_features.shape)
    torch.save(final_features, file_path)

    return file_path

def gen_window_latent_space(windows_path, z_score = False) -> str:
    LATENT_SPACE_FILE = ""
    if(z_score): LATENT_SPACE_FILE = "latent_space_full_windows_with_z_score.pt"
    else: LATENT_SPACE_FILE = "latent_space_full_windows_without_z_score.pt"

    file_path = Path(f"{LATENT_SPACE_DIR}/{LATENT_SPACE_FILE}")
    if file_path.exists(): return file_path

    # O diretório onde você salvou as janelas (ex: OUTPUT_DIR = "./pre-processed-pyprep-eeg")
    loaded_windows = load_concat_dataset(
        path=windows_path,
        preload=False  # load by demand (lazy loading), saves RAM
    )

    # 2. Configurar o dispositivo e carregar o BrainOmni pré-treinado
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_brainomni('ckpt_collection/base') # Inicialize ou carregue o checkpoint do BrainOmni
    model.to(device)
    model.eval()

    latent_outputs = []

    with torch.no_grad():
        for dataset_idx, window_ds in enumerate(loaded_windows.datasets):
            pos_3d = torch.tensor(window_ds.ch_pos, dtype=torch.float32).unsqueeze(0).to(device)
            pos = torch.cat([pos_3d, torch.zeros_like(pos_3d)], dim=-1)
            
            print(f"\nGenerating latent space for subject {dataset_idx+1}/{len(loaded_windows.datasets)}\n")
            for idx in range(len(window_ds)):
                x, y, _ = window_ds[idx]
                
                if(z_score): x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-6)
                inputs = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
                sensor_type = torch.zeros((1, x.shape[0]), dtype=torch.long).to(device)

                latent_space = model.encode(inputs, pos, sensor_type) # Ex: [1, 16, 512] ou [1, 1, 16, 512]
                
                # Achata todas as dimensões exceto o batch/primeira, transformando em um vetor 1D por janela
                latent_vector = latent_space.reshape(latent_space.shape[0], -1) # [1, Total_Features]
                latent_outputs.append(latent_vector.cpu())

                if(not idx%500):
                    print(f"{idx+1}/{len(window_ds)}")

    print(latent_outputs[0].shape)

    # Concatena todas as janelas de todos os sujeitos em uma única matriz 2D: [1876, Total_Features]
    final_features = torch.cat(latent_outputs, dim=0).numpy()

    print("Formato final para o UMAP (por janelas):", final_features.shape)
    torch.save(final_features, f"{LATENT_SPACE_DIR}/{LATENT_SPACE_FILE}")

    return f"{LATENT_SPACE_DIR}/{LATENT_SPACE_FILE}"

def gen_umap(windows_path, option):
    windows_path = Path(windows_path)

    os.makedirs(LATENT_SPACE_DIR, exist_ok=True) # Ensure that the folder already exists
    os.makedirs(PLOTS_DIR, exist_ok=True) # Ensure that the folder already exists
    assert windows_path.is_dir(), f"{windows_path} is not a valid directory"

    options = ["avg_with_z_score", "avg_without_z_score", "windows_with_z_score", "windows_without_z_score"]
    assert option in options, "Wrong option"

    latent_space_file = ""

    if(option == "avg_with_z_score"):
        latent_space_file = gen_avg_latent_space(windows_path, z_score=True)
    elif(option == "avg_without_z_score"):
        latent_space_file = gen_avg_latent_space(windows_path, z_score=False)
    elif(option == "windows_with_z_score"):
        latent_space_file = gen_window_latent_space(windows_path, z_score=True)
    elif(option == "windows_without_z_score"):
        latent_space_file = gen_window_latent_space(windows_path, z_score=False)
    
    final_features = torch.load(latent_space_file)

    # 1. Configurar e rodar o UMAP na matriz de características dos indivíduos
    # (Certifique-se de que final_features é um array numpy com shape [N_indivíduos, Dimensão_Latente])
    reducer = umap.UMAP(
        n_neighbors=min(5, len(final_features) - 1), # Ajuste o n_neighbors se tiver poucos sujeitos
        min_dist=0.1, 
        metric='cosine', 
        random_state=42
    )
    embedding = reducer.fit_transform(final_features)

    # 2. Plotar o gráfico 2D do UMAPs
    fig = plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=embedding[:, 0], 
        y=embedding[:, 1],
        # hue=subject_labels, # Opcional: passe uma lista com os rótulos/classes dos sujeitos se tiver
        palette='viridis',
        s=100,
        alpha=0.8
    )

    title = str(latent_space_file).split(".")[0].split("/")[-1]

    plt.title(f"UMAP - {title} (BrainOmni)", fontsize=14)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    fig.savefig(f"{PLOTS_DIR}/{title}.png", dpi=300, bbox_inches='tight')