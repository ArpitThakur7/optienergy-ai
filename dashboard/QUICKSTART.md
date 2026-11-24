# Start here

1. Open this repository's **dashboard** folder in VS Code.
2. Use Python 3.11-3.13. In a terminal, run:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

3. Open the local address printed by Streamlit. The demo data, results, and model are already included; no initial training or download is needed.
4. Read `README.md` for the complete setup and experimental protocol.

Useful commands (use your environment's Python):

```bash
python -m pytest -q
python -m src.experiments --mode demo
python -m src.build_report --mode demo
```

The report is `reports/Project_Report.pdf`. The timed presentation is `reports/Presentation_Content.md`. Maths, measured analysis, architecture, and 40 viva questions are beside them. The notebook is `notebooks/OptiEnergy_Colab.ipynb`.

The included measurements are a 2,400-hour real-UCI demo with three seeds. The complete-history experiment is implemented, but you must run `python -m src.experiments --mode full` separately. Full mode can be a long CPU job.
