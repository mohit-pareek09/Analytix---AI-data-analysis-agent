# Analytix---AI-data-analysis-agent

# 📘 **README — Local Data Analysis Agent using Ollama + Gemma 3**

This project implements a **local AI-powered data analysis assistant** that works entirely offline using the **Gemma 3** model running on **Ollama**.
It can:

* Clean columns
* Generate histograms
* Compute correlations
* Automatically choose the correct tool based on your natural-language question

Works with any CSV dataset.

---

# 🚀 **Features**

### ✅ Natural Language → Tool Selection

Ask things like:

* “Plot a histogram of age”
* “Clean the weight column”
* “What columns are present?”

The agent interprets your query, selects the right tool, and runs Python code.

### ✅ Tools Implemented

* **clean_feature** → fills missing numeric/categorical values
* **plot_histogram** → generates `.png` plots
* **calculate_correlation** → Pearson correlation between two numeric columns
* **help** → explains available tools

### ✅ Fully Offline

Runs entirely on your system using **Ollama + Gemma 3**, no cloud required.

---

# 📦 **Installation**

### **1. Install Ollama**

Download from: [https://ollama.com/download](https://ollama.com/download)

Check installation:

```bash
ollama version
```

### **2. Pull the Gemma 3 model**

```bash
ollama pull gemma3:1b
```

(Your machine must show this model in `ollama list`)

### **3. Install Python dependencies**

```bash
pip install ollama pandas matplotlib pydantic langchain-ollama
```

---

# 📁 **Project Structure**

```
agent_pai/
│
├── app.py          # Main AI agent
├── data.csv        # Your dataset
└── README.md       # This file
```

---

# ▶️ **How to Run**

```bash
python app.py data.csv
```

You’ll see:

```
Ready. Ask something about the dataset (type 'exit' to quit).
You:
```

Now you can chat with your dataset!

---

# 💬 **Example Questions to Ask**

### 📊 **Histograms**

```
Plot a histogram of age
Plot a histogram of weight with 20 bins
Visualize the height distribution
```

### 🧹 **Cleaning**

```
Clean the column weight
Remove missing values from height
Fix NaN in score
```

### 📈 **Correlation**

```
Calculate the correlation between weight and score
Find relation between age and height
```

### 🔍 **Smart Tool Selection**

```
Which feature is most correlated with age?
Visualize anything interesting
What analysis can you do?
```

### 🧨 **Error Handling**

```
Clean abcdef
Plot histogram of hello
```

---

# 🛠 **How It Works**

### **1. Your question is fed to Gemma 3 via Ollama**

### **2. Model returns a JSON object:**

```json
{
  "tool": "plot_histogram",
  "feature": "age",
  "feature2": null,
  "bins": 10
}
```

### **3. Python executes the chosen tool**

* Pandas handles cleaning & data operations
* Matplotlib generates plots
* Output printed in terminal

---

# 🧪 **Test Dataset**

A 200-row dataset (`data.csv`) is included with columns:

```
id, age, weight, height, score
```

---

# 🐞 **Debug Mode**

The app prints extra debug information if a response cannot be parsed.
This helps diagnose any issues with model output.

---

# 🔧 **Troubleshooting**

### ❌ **Model error: model 'gemma3:1b' not found**

Run:

```bash
ollama pull gemma3:1b
ollama list
```

### ❌ **Model error: does not support tools**

This project **does not use tools** — everything is JSON mode.
So you’re safe.

### ❌ **JSON parsing failed**

Model responded with non-JSON.
Just re-ask, or tighten instructions.

---

# ⭐ **Future Improvements**

* Add scatter plots
* Add full statistical summaries
* Export cleaned data automatically
* Build a web UI (Streamlit)

---

# 🙌 **Credits**

* **Gemma 3** model by Google DeepMind
* **Ollama** for local model serving
* **LangChain-Ollama** for integration
* **Pandas + Matplotlib** for data analysis

---

