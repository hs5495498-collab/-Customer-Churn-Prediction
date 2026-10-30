# Setup and Execution Guide — Fresh Windows Machine

A step-by-step walkthrough for running this project from scratch on a Windows computer that has nothing installed yet. Commands are shown for PowerShell.

---

## 1. Install Python

1. Download Python 3.12 from [python.org/downloads](https://www.python.org/downloads/).
2. Run the installer. **Check "Add python.exe to PATH"** on the first screen before clicking Install.
3. Verify the install in a new PowerShell window:
   ```powershell
   python --version
   ```
   This should print something like `Python 3.12.x`.

*Why:* every tool in this project (data cleaning, model training, the notebook, the Streamlit app, tests) is Python code, so Python must be installed first.

## 2. Install Git

1. Download Git from [git-scm.com/download/win](https://git-scm.com/download/win) and run the installer (default options are fine).
2. Verify:
   ```powershell
   git --version
   ```

*Why:* Git is needed to download (clone) the project's source code from GitHub and to track any changes you make.

## 3. Clone the GitHub Repository

```powershell
git clone https://github.com/hs5495498-collab/-Customer-Churn-Prediction.git
cd -Customer-Churn-Prediction
```

*Why:* `git clone` downloads a full copy of the repository (all commits, branches, and the current state of every tracked file) onto your machine. `cd` moves your terminal into that new folder so subsequent commands run in the right place.

## 4. Open the Project in VS Code

1. Install [Visual Studio Code](https://code.visualstudio.com/) if you don't have it.
2. From inside the cloned folder:
   ```powershell
   code .
   ```
   (Or open VS Code and use File → Open Folder, selecting the cloned `-Customer-Churn-Prediction` folder.)
3. Install the "Python" and "Jupyter" extensions from the Extensions panel if prompted.

## 5. Create a Virtual Environment

```powershell
python -m venv .venv
```

*Why:* a virtual environment is an isolated, project-specific copy of Python and its installed packages. This keeps this project's exact dependency versions separate from any other Python project on your machine, avoiding version conflicts.

## 6. Activate the Environment

```powershell
.venv\Scripts\activate
```

You should see `(.venv)` appear at the start of your terminal prompt, confirming it's active. **Every command below assumes this environment is active** — if you close and reopen your terminal, re-run this activation command before continuing.

*Why:* activating tells your terminal to use this project's isolated Python and packages instead of your system-wide Python installation.

## 7. Install Requirements

```powershell
pip install -r requirements.txt
```

This installs pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter, notebook, streamlit, joblib, and pytest — all pinned to specific tested versions in `requirements.txt`. This can take a few minutes the first time, especially for Jupyter and Streamlit.

*Why:* these are the exact libraries the project's code imports; without them, nothing in the project will run.

## 8. Download and Place the Kaggle Dataset

1. Go to the [Telco Customer Churn dataset page on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (you'll need a free Kaggle account, and may need to agree to the dataset's terms).
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
3. Create a `data` folder in the project root if it doesn't already exist, and place the file there, so the final path is:
   ```
   -Customer-Churn-Prediction\data\WA_Fn-UseC_-Telco-Customer-Churn.csv
   ```

*Why:* the raw dataset is **not included in this Git repository** (by design — see the README's Dataset section), so it must be downloaded separately before any cleaning, training, or notebook code can run.

## 9. Generate the Final Model Artifact

The trained model file (`models/final_churn_model.joblib`) is also **not included in Git** (it's a large, regenerable binary, excluded via `.gitignore`). **Do not assume it exists after cloning** — generate it yourself with one of:

**Option A — standalone script (faster):**
```powershell
python -m src.validation
```

**Option B — run the notebook (see Step 10 below), executing all cells.**

Both reproduce the exact same documented Stage 5 results (best C=10, selected threshold=0.60) because every random seed in the pipeline is fixed. This step requires the dataset from Step 8 to already be in place.

## 10. Running the Notebook

```powershell
jupyter notebook notebooks/churn_analysis.ipynb
```

This opens Jupyter in your browser. Use the menu **Cell → Run All** (or **Kernel → Restart & Run All**) to execute every cell from top to bottom — this reproduces all cleaning, EDA, modeling, validation, and the final model artifact. Running the full notebook can take a couple of minutes, mostly due to the `GridSearchCV` step in Section 11.

## 11. Running the Streamlit Application

```powershell
streamlit run app/app.py
```

Run this from the **project root** (the folder containing `app/`, not from inside `app/` itself). Streamlit will print a local URL (typically `http://localhost:8501`) and should open it in your default browser automatically. Requires the model artifact from Step 9 to exist — if it's missing, the app will show a clear on-screen error rather than crashing unexpectedly or training a substitute model.

## 12. Running Tests

```powershell
pytest tests/ -v
```

Runs all 69 automated tests. Some tests (those checking the real model artifact and the full Streamlit app) are automatically skipped if `models/final_churn_model.joblib` doesn't exist yet — complete Step 9 first to run the full suite.

## 13. Troubleshooting Common Errors

**`python : The term 'python' is not recognized...`**
Python wasn't added to PATH during installation. Reinstall Python and make sure to check "Add python.exe to PATH," or restart your terminal after installing.

**`.venv\Scripts\activate` does nothing, or PowerShell shows a script-execution error.**
PowerShell's execution policy may be blocking scripts. Run PowerShell as Administrator once and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
then try activating again.

**`ModuleNotFoundError: No module named 'pandas'` (or similar) when running a script.**
The virtual environment isn't activated, or `pip install -r requirements.txt` wasn't run (or was run before activating). Re-run Step 6, then Step 7.

**`FileNotFoundError` mentioning `WA_Fn-UseC_-Telco-Customer-Churn.csv`.**
The dataset hasn't been downloaded and placed yet — see Step 8. Check the exact filename and path match precisely (case-sensitive on some systems).

**The Streamlit app shows "Model artifact not found."**
`models/final_churn_model.joblib` doesn't exist yet — see Step 9. This is expected immediately after a fresh clone.

**`ImportError: pyarrow requires NumPy 2.0 or newer...` (or similar version mismatch).**
This can occur if a package was installed outside the pinned `requirements.txt` versions. Try reinstalling cleanly:
```powershell
pip install -r requirements.txt --force-reinstall
```
If the issue persists, it may indicate your environment picked up a newer version of a transitive dependency (like `pyarrow`) than was tested with this project; the application itself was specifically written to avoid this issue for its own features (see `app/app.py`'s sidebar metrics table, which uses markdown rather than `st.table()` for exactly this reason).

**Jupyter opens but a cell fails with an import error for `src` or `app`.**
Make sure you're running Jupyter from the project root (the folder containing both `notebooks/` and `src/`), not from inside the `notebooks/` folder directly in a terminal — the notebook's own `sys.path.insert(0, "..")` cells assume this.

**Tests related to the model or app are being skipped.**
This is expected if `models/final_churn_model.joblib` doesn't exist — those specific tests are designed to skip gracefully rather than fail, with a message explaining why. Complete Step 9 to un-skip them.
