# How to View the Architecture Diagram

## ✅ Method 1: VS Code Extension (RECOMMENDED - Best Quality)

### Step 1: Install PlantUML Extension
1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for **"PlantUML"** by jebbs
4. Click Install

### Step 2: Install Java (Required)
PlantUML needs Java to render diagrams.
- Download from: https://www.java.com/download/
- Install and restart VS Code

### Step 3: View the Diagram
1. Open `architecture_diagram_simple.puml` in VS Code
2. Press **Alt+D** to preview
3. Right-click on preview → **Export Current Diagram**
4. Choose format: PNG, SVG, or PDF

---

## ✅ Method 2: Online Viewer (Quick & Easy)

### Option A: PlantText (Works for smaller diagrams)
1. Go to: https://www.planttext.com/
2. Open `architecture_diagram_simple.puml`
3. Copy ALL the content (Ctrl+A, Ctrl+C)
4. Paste into PlantText
5. Click "Refresh" to generate
6. Download as PNG

### Option B: PlantUML.com
1. Go to: http://www.plantuml.com/plantuml/uml/
2. Paste the code from `architecture_diagram_simple.puml`
3. The diagram will render automatically
4. Right-click → Save image

---

## ✅ Method 3: Local PlantUML Installation

### Windows:
```powershell
# Install Java first
# Then download PlantUML JAR
Invoke-WebRequest -Uri "https://sourceforge.net/projects/plantuml/files/plantuml.jar/download" -OutFile "plantuml.jar"

# Generate PNG from .puml file
java -jar plantuml.jar architecture_diagram_simple.puml

# This creates: architecture_diagram_simple.png
```

---

## ✅ Method 4: Draw.io Import

1. Go to: https://app.diagrams.net/
2. File → Import from → Text
3. Paste the PlantUML code
4. It will convert to Draw.io format
5. Edit, customize, and export

---

## ✅ Method 5: Markdown Preview (GitHub)

1. Upload the .puml file to GitHub repository
2. GitHub will auto-render PlantUML diagrams
3. View directly in browser

---

## 📊 Which File to Use?

### `architecture_diagram_simple.puml`
- **Recommended for online viewers**
- Simplified, compact version
- Works with PlantText, PlantUML.com
- ~100 lines, small header size

### `system_architecture_diagram.puml`
- **Recommended for VS Code Extension**
- Full detailed version with all components
- 500+ lines, very comprehensive
- May be too large for online tools

---

## 🎯 Quick Start (Fastest Method)

### For Immediate Viewing:
1. **Go to**: https://www.planttext.com/
2. **Open**: `architecture_diagram_simple.puml`
3. **Copy**: Press Ctrl+A, then Ctrl+C
4. **Paste** into PlantText website
5. **Click**: "Refresh" button
6. **Download**: Right-click image → Save

### For High-Quality Export:
1. **Install**: PlantUML extension in VS Code
2. **Install**: Java Runtime (JRE)
3. **Open**: Either .puml file in VS Code
4. **Press**: Alt+D to preview
5. **Export**: Right-click → Export as PNG/SVG/PDF

---

## 🔧 Troubleshooting

### "Request header too large" error:
- **Solution**: Use `architecture_diagram_simple.puml` instead
- The simpler file has fewer lines and works with online tools

### "Syntax error" in online viewer:
- **Solution**: Make sure you copied the ENTIRE file including:
  - `@startuml` at the beginning
  - `@enduml` at the end

### Nothing appears in VS Code preview:
- **Solution**: Install Java Runtime Environment
- Download from: https://www.java.com/download/

### Export button grayed out:
- **Solution**: Wait for diagram to fully render
- Check Java is installed correctly

---

## 📐 Diagram Contents

Both diagrams include:

✅ **7 Major Layers**:
1. Data Layer (CSV files, preprocessing)
2. Model Training (10+ ML/DL models)
3. Advanced AI (MAML, NAS, Federated, Causal)
4. Prediction Engine (Single model, Ensemble)
5. Flask API (20+ endpoints)
6. Web Interface (6 pages)
7. Outputs (Results, files, reports)

✅ **Complete Data Flow**: 
- CSV → Preprocessing → Models → Predictions → API → UI → Results

✅ **Technology Stack**:
- Backend: Python, Flask, PyTorch
- Models: LSTM, GRU, XGBoost, etc.
- Frontend: HTML5, Chart.js, D3.js

✅ **Performance Metrics**:
- 95.8% Accuracy
- 2.4ms Response Time
- 342K Data Points

---

## 💡 Tips

1. **For presentations**: Export as SVG (vector format, scales perfectly)
2. **For documents**: Export as PNG (good quality, widely compatible)
3. **For printing**: Export as PDF (high resolution)
4. **For editing**: Use Draw.io to import and customize

---

## 🎨 Color Legend

- 🔵 **Blue** (#E3F2FD) - Data Layer
- 🟠 **Orange** (#FFF3E0) - Model Training
- 🟢 **Green** (#E8F5E9) - Advanced AI
- 🔴 **Pink** (#FCE4EC) - Prediction Engine
- 🔵 **Cyan** (#E1F5FE) - API Layer
- 🟣 **Purple** (#F3E5F5) - Frontend
- 🟡 **Yellow** (#FFF9C4) - Outputs

---

Need help? The diagram visualizes your complete pharmaceutical sales forecasting system architecture with all components and data flows!
