# TeXVision


*Automatic LaTeX Generation for Complex Mathematical Formulas*

---

## 🚀 Project Overview

TeXVision converts images of complex mathematical expressions—both printed and handwritten—into clean, compile‑ready LaTeX code using a quantized vision–language model fine‑tuned with LoRA.

## 🔧 Key Features

* **Quantized Qwen2-VL-7B Model**: 4-bit quantized backbone for reduced memory footprint.
* **LoRA Fine‑Tuning**: Lightweight adapters (rank=16, α=32) for efficient domain adaptation on Unsloth dataset.

* **Streamlit Web App**: Interactive drag‑and‑drop interface for instant LaTeX rendering and preview.

## 🧠 Model & Training

| Component | Details                                                      |
| --------- | ------------------------------------------------------------ |
| Backbone  | Qwen2-VL-7B (4-bit quantized)                                |
| Adapter   | LoRA (dropout=0.1)                                           |
| Dataset   | Unloth (printed & handwritten formulas)                      |
| Training  | AdamW, lr=3e-5, cosine decay, 10 epochs, batch size=64, FP16 |

## 📦 Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/kaustubhdeshmukh11/TexVision.git
   cd texvision
   ```
2. **Setup environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Download models**
   Place the quantized Qwen2‑VL‑7B checkpoint and LoRA adapters in `models/quantized_qwen2_vl/`.

## 💻 Usage

1. **Launch Streamlit App**

   ```bash
   streamlit run app/streamlit_main.py
   ```
2. **Open the UI**
   In your browser, go to `http://localhost:8501`, upload an image, and view the generated LaTeX output live.



