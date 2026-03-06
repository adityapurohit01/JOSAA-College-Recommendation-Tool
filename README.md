# 🎯 JoSAA Rank Predictor & IIT College Recommendation Tool

<div align="center">
  <p><strong>🚀 Empowering JEE Aspirants with Data-Driven College Recommendations</strong></p>
  <p>
    <a href="#features">✨ Features</a> •
    <a href="#installation">⚙️ Setup</a> •
    <a href="#usage">🎮 Usage</a> •
    <a href="#roadmap">🗺️ Roadmap</a> •
    <a href="#contributing">🤝 Contribute</a>
  </p>
</div>

---

## 📋 Overview

The **JoSAA Rank Predictor & IIT College Recommendation Tool** is a Python-based utility that helps **JEE aspirants, parents, and counsellors** make informed decisions during JoSAA counselling.

✅ **Input:** Your JEE rank, category, and preferences  
✅ **Output:** Realistic IIT, NIT, and IIIT options with branch-wise cutoffs  
✅ **Data Source:** Official JoSAA Excel files (historical data)  
✅ **Result:** Fast, transparent, offline predictions

> **Why this tool?** Instead of jumping between scattered PDFs and guessing, get all possibilities ranked and organized in seconds! 🏃⚡

---

## ✨ Key Features

🎓 **IIT-First Recommendations**  
Prioritises IIT options based on your rank, then shows NIT and IIIT possibilities.

📊 **Official JoSAA Data**  
All cutoffs and seat information come from JoSAA's published Excel files—100% transparent and trustworthy.

🏷️ **Smart Filtering**  
Filter by category (GEN, OBC-NCL, SC, ST, EWS, etc.) and quota type with official JoSAA data.

🔧 **Branch & Institute Options**  
Explore which branches across IITs are within your reach for a given rank.

⚡ **Fast Offline Processing**  
No internet required after downloading data. Pure Python logic runs on your system.

📁 **Simple Excel-Based Workflow**  
Drop your JoSAA Excel file, run the script, get results.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | 🐍 Python 3.8+ |
| **Data Format** | 📊 Excel (.xlsx) |
| **Core Dependencies** | `pandas`, `openpyxl` |
| **Architecture** | Rank-based filtering on official JoSAA cutoffs |

---

## 📁 Project Structure

```
JOSAA-College-Recommendation-Tool/
│
├── 📂 FORM_KART/
│   └── main.py              # 🎯 Main entry script
│
├── 📂 data/                 # 📊 Place JoSAA Excel files here
│   └── josaa_cutoff.xlsx    # Example: Official JoSAA data
│
├── README.md                # 📖 Documentation
└── requirements.txt         # 📦 Python dependencies
```

---

## 🚀 Quick Start

### 1️⃣ Clone Repository
```bash
git clone https://github.com/adityapurohit01/JOSAA-College-Recommendation-Tool.git
cd JOSAA-College-Recommendation-Tool
```

### 2️⃣ Set Up Virtual Environment (Recommended)
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install pandas openpyxl
```

### 4️⃣ Add JoSAA Data
1. Download official JoSAA cutoff/allotment Excel from [JoSAA Official Website](https://www.josaa.ac.in/)
2. Place in `data/` folder (e.g., `data/josaa_2025_cutoff.xlsx`)
3. Update file path in `FORM_KART/main.py` if needed

### 5️⃣ Run the Tool
```bash
python FORM_KART/main.py
```

---

## 🎮 Usage Guide

### Basic Workflow

```bash
$ python FORM_KART/main.py

📊 JoSAA Rank Predictor Tool
═════════════════════════════════════

🎯 Enter your JEE rank: 15000
🏷️ Enter your category: OBC-NCL
🎓 Choose mode:
   1) IIT Only
   2) IIT + NIT + IIIT
   👉 Select (1/2): 1

🔍 Processing your profile...
```

### Example Output

```
✅ REALISTIC PREDICTIONS FOR RANK 15000 (OBC-NCL)
═════════════════════════════════════════════════

🏆 IIT OPTIONS:
├─ IIT Delhi | CSE | Closing Rank: 14,500
├─ IIT Bombay | ECE | Closing Rank: 14,800
├─ IIT Kanpur | Mechanical | Closing Rank: 15,200
├─ IIT Kharagpur | Civil | Closing Rank: 15,500
└─ IIT BHU | Electrical | Closing Rank: 16,100

💡 Tip: These are based on previous year cutoffs.
        Actual admission may vary based on:
        - Number of applicants
        - Difficulty level
        - Seat matrix changes
```

---

## 📊 How It Works

### Algorithm Overview

```
┌─────────────────────┐
│  User Input         │
│ • Rank: 15000       │
│ • Category: OBC-NCL │
│ • Preference: IIT   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Load JoSAA Data    │
│  (Excel file)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Filter by:         │
│ • Category Match    │
│ • Institute Type    │
│ • Closing Rank ≥ 15K│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Rank & Display     │
│  Recommendations    │
│  (Closest match)    │
└─────────────────────┘
```

### Data Filtering Logic

✔️ **Closing Rank Filter:**  
If your rank < closing rank, you have a realistic chance.

✔️ **Category Filter:**  
Matches your category quota from official JoSAA data.

✔️ **Institute Priority:**  
IITs listed first, then NITs, then IIITs.

---

## 📝 Example Scenarios

### Scenario 1: IIT-Focused
**Profile:**
- Rank: 8,000 (JEE Advanced)
- Category: GEN
- Goal: Top IITs with CSE/ECE

**Result:**
```
✅ IIT Delhi - CSE
✅ IIT Bombay - ECE
✅ IIT Kanpur - CSE (Open)
```

### Scenario 2: Balanced Approach
**Profile:**
- Rank: 45,000 (JEE Mains)
- Category: SC
- Goal: IIT/NIT with good branch

**Result:**
```
✅ NIT Warangal - CSE
✅ NIT Surathkal - IT
✅ IIIT Hyderabad - CSE
```

---

## ⚠️ Important Disclaimers

⚡ **Historical Data:** Predictions are based on previous years' cutoffs. Current year results may vary significantly.

🔄 **Annual Variations:** Cutoffs fluctuate due to:
- Number of candidates taking exam
- Difficulty level of exam
- Changes in seat matrix
- Policy updates

🎯 **Not Official:** This tool is a **guidance aid**, not an official JoSAA counselling engine.

📖 **Use Alongside Official Resources:** Always cross-check with official JoSAA portals and college websites.

---

## 🗺️ Roadmap

### Phase 1 ✅ (Current)
- [x] Core rank-filtering logic
- [x] Excel-based data support
- [x] CLI interface

### Phase 2 🚧 (Planned)
- [ ] 🌐 Web-based GUI (Streamlit/Flask)
- [ ] 📅 Multi-year data selector
- [ ] 🎯 Advanced filters (state, gender-neutral, branch preference)
- [ ] 📊 Visualization (cutoff trends, choice fill strategy)
- [ ] 💾 Export to CSV/PDF

### Phase 3 💡 (Future)
- [ ] 🤖 ML-based predictive modeling
- [ ] 📱 Mobile app
- [ ] 🔔 Notification system for cutoff updates
- [ ] 🌍 Support for IIIT & SFTI colleges

---

## 🤝 Contributing

We ❤️ contributions! Here's how to get started:

### Steps

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/JOSAA-College-Recommendation-Tool.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
   - Add new features
   - Fix bugs
   - Improve documentation

4. **Commit with clear messages**
   ```bash
   git commit -m "✨ Add feature X" # Use meaningful emoji prefixes
   ```

5. **Push and create Pull Request**
   ```bash
   git push origin feature/amazing-feature
   ```

### Contribution Ideas 💡
- 🐛 **Bug fixes:** Report and fix issues
- 📝 **Documentation:** Improve README, add examples
- 🌐 **Web interface:** Build a Streamlit/Flask frontend
- 📊 **Features:** Add new filtering options
- 🧪 **Testing:** Write unit tests

---

## 📄 License

This project is open-source and available under the **MIT License**. See LICENSE file for details.

---

## 📞 Support & Feedback

💬 **Have questions?**
- Open an [Issue](https://github.com/adityapurohit01/JOSAA-College-Recommendation-Tool/issues)
- 📧 Email: aditya@example.com

⭐ **Like this tool?**
- Please star ⭐ the repository
- Share with friends & counsellors
- Spread the word! 📣

---

## 🙏 Acknowledgments

- 🏛️ **JoSAA (Joint Seat Allocation Authority)** for official counselling data
- 🎓 **IIT Council** for maintaining consistent counselling processes
- 💻 **Open-source community** for amazing tools like Pandas
- 👥 **Contributors** who help improve this tool

---

<div align="center">
  <p><strong>Made with ❤️ by Aditya Purohit</strong></p>
  <p>
    <a href="https://github.com/adityapurohit01">GitHub</a> •
    <a href="https://twitter.com/adityapurohit01">Twitter</a> •
    <a href="https://linkedin.com/in/adityapurohit01">LinkedIn</a>
  </p>
  <p>⭐ If this helped you, please star the repo! ⭐</p>
</div>
