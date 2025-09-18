
# 🛰️ Land Use and Land Cover (LULC) Classification with Vision Transformers

This project revisits and rebuilds a previous Land Use and Land Cover (LULC) classification task on the EuroSAT dataset, using **Vision Transformers (ViT)** and **modern PyTorch best practices**.

It covers everything from dataset loading, preprocessing, training with transfer learning, to saving and using the model for inference — all in a clean, modular structure powered by **Pipenv** and reproducible configs.

## 📜  Read  my blog post here

Part 1 : [How I Rebuilt My First LULC Project with a Vision Transformer and Clean Code](https://medium.com/@bernardinligan/how-i-rebuilt-my-first-lulc-project-with-a-vision-transformer-and-clean-code-aa9a06094c89)

Part 2 : [From Model to Map: Automating Land Use Classification with Vision Transformers and Google Earth Engine](https://medium.com/@bernardinligan/from-model-to-map-automating-land-use-classification-with-vision-transformers-and-google-earth-9a5510215054)

## 📂 Project Structure

```

├── config.yaml           # All training and data parameters
├── data/                 # Folder for the EuroSAT dataset (structure preserved, content ignored by Git)
├── datafactory.py        # Custom Dataset and transforms
├── engine.py             # Training and evaluation loops
├── LULC\_3BIMG\_VIT.ipynb  # Main training notebook
├── outputs/              # Folder to save models, plots, reports (content ignored by Git)
├── utils.py              # Utilities (plotting, seeding, inference tools)
├── Pipfile / Pipfile.lock # Reproducible environment with Pipenv
└── .gitignore

````

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/LiganiumInc/LULC-ViT-EuroSAT3B.git
cd LULC-ViT-EuroSAT3B
````

### 2. Set up the Environment

```bash
# Make sure Pipenv is installed
pip install pipenv

# Create and activate virtual environment
export PIPENV_VENV_IN_PROJECT=1
pipenv install
pipenv shell
```

### 3. Download the EuroSAT RGB Dataset

* Link: [https://github.com/phelber/EuroSAT](https://github.com/phelber/EuroSAT)
* Extract into the following structure:

```
data/
└── EuroSAT/
    └── 2750/
        ├── AnnualCrop/
        ├── Forest/
        └── ...
```

---

## ⚙️ Configuration

All settings can be found in `config.yaml`. Example:

```yaml
data_dir: "./data/EuroSAT/2750/"
batch_size: 32
num_epochs: 10
lr: 0.001
weight_decay: 0.05
percentage_per_class: 0.3
```

---

## 🧠 Model: Vision Transformer (ViT)

We use the pretrained `vit_b_16` from `torchvision.models`, and fine-tune it to classify 10 LULC categories.



## 📊 Outputs

After training, you'll find:

* `best_model.pth`: The saved fine-tuned ViT model
* `loss_acc_curves.png`: Visual training metrics
* `train_report.txt`: Final performance logs

---

## 🧑‍💻 Author

**Bernardin Ligan**
PhD Student in AI & Remote Sensing

🌍 Passionate about geospatial ML, open science, and SDGs 

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

```

