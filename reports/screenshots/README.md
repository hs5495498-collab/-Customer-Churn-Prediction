# Screenshots — Manual Capture Required

No automated screenshots have been captured for this folder. The browser automation tool available in this development environment was not connected when Stage 6 (Streamlit application) and Stage 7 (documentation) were completed, so no screenshot files exist here yet. This is stated honestly rather than faking placeholder images — **do not** assume any `.png` files exist in this folder until you add them yourself.

The application itself was verified working using Streamlit's headless `AppTest` framework (see `tests/test_app.py`), which confirmed the app starts, the form renders, a prediction is produced, and the reset/validation paths work — but that verification does not produce visual screenshots.

## What to capture

Please capture the following 6 screenshots yourself and save them into this folder with these exact filenames (so `README.md` can reference them):

1. **`01_main_page.png`** — The main application page on first load (title "Customer Churn Prediction System", subtitle, and the empty customer input form).
2. **`02_input_form.png`** — The customer input form filled in with example values (any realistic combination, e.g. a month-to-month customer with short tenure).
3. **`03_prediction_result.png`** — The prediction result after clicking "Predict Churn Risk," showing the churn probability percentage, the progress bar, and the High Risk/Low Risk badge.
4. **`04_retention_recommendations.png`** — The "Retention Recommendations" section shown below a prediction result.
5. **`05_model_info_sidebar.png`** — The sidebar showing the "About the Model" section and the held-out test metrics table.
6. **`06_eda_or_evaluation_chart.png`** — Any one relevant chart from `reports/figures/` (for example `churn_rate_by_contract.png` or `final_model_evaluation.png`), shown either from the notebook or copied directly — this one may reuse an existing file from `reports/figures/` instead of a fresh screenshot, since those are already real, verified chart outputs.

## How to capture them on Windows

1. Activate the project's virtual environment and start the app:
   ```powershell
   .venv\Scripts\activate
   streamlit run app/app.py
   ```
2. Your default browser should open automatically to `http://localhost:8501`. If not, open that URL manually.
3. Resize the browser window to a reasonable desktop width (e.g. maximize it) so the dashboard layout looks clean.
4. For each screenshot:
   - Arrange the page/state as described above (e.g. fill the form before capturing #2, submit it before capturing #3 and #4).
   - Press **Win + Shift + S** to open the Windows Snipping Tool, select the browser window or a rectangular region, and the snip is copied to your clipboard.
   - Open Paint (or any image editor), paste (**Ctrl+V**), and save as a `.png` with the exact filename listed above into this folder (`reports/screenshots/`).
5. Repeat for all 6 screenshots.

Once the files exist, you can reference them from `README.md` using relative paths, e.g.:
```markdown
![Main application page](reports/screenshots/01_main_page.png)
```

`README.md` in this project currently does **not** embed any screenshot images, specifically because none exist yet — add the `![...]` lines yourself once you've captured and saved the files, so the README never references a missing image.
