import json
import os
from urllib.request import urlopen
from pathlib import Path
from playwright.sync_api import sync_playwright

out = Path("evidencias")
info = json.loads((out / "resultados.json").read_text())["info"]
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get("CHROME","/usr/bin/google-chrome"),args=["--no-sandbox"])
    page = browser.new_page(viewport={"width":1120, "height":740}, device_scale_factor=1.25)
    page.goto("http://localhost:8000/docs")
    page.get_by_text("Clasificar una flor").wait_for()
    page.screenshot(path=out/"swagger.png")
    page.get_by_text("Clasificar una flor").click()
    page.get_by_role("button",name="Try it out").click()
    page.get_by_role("button",name="Execute").click()
    live = page.locator(".live-responses-table")
    live.get_by_text("setosa", exact=False).first.wait_for()
    live.screenshot(path=out/"swagger-respuesta.png")
    page.goto("http://localhost:8000/model-info")
    page.locator("pre").wait_for()
    page.screenshot(path=out/"model-info.png")
    tracking = json.load(urlopen("http://localhost:5000/api/2.0/mlflow/runs/get?run_id="+info["run_id"]))["run"]["info"]
    page.goto(f"http://localhost:5000/#/experiments/{tracking['experiment_id']}/runs/{info['run_id']}")
    page.get_by_text(tracking["run_name"],exact=False).first.wait_for(timeout=90000)
    page.screenshot(path=out/"mlflow.png")
    browser.close()
