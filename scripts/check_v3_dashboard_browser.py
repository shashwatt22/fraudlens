"""Exercise the verified V3 dashboard in Chromium and optionally capture share-safe screenshots."""

import argparse
from pathlib import Path
import re

from playwright.sync_api import expect, sync_playwright


def heading(page, title):
    expect(page.get_by_role("heading", name=title, exact=True)).to_be_visible()


def clean(page):
    expect(page.get_by_test_id("stException")).to_have_count(0)
    expect(page.get_by_test_id("stAlert").filter(has_text=re.compile("unavailable|failed", re.I))).to_have_count(0)


def text(page, value, exact=False):
    expect(page.get_by_text(value, exact=exact).first).to_be_visible()


def radio(page, name):
    page.locator('[role="radiogroup"] label').filter(has_text=re.compile(f"^{re.escape(name)}$")).click()


def choose(page, key, option):
    picker = page.locator(f".st-key-{key}")
    picker.get_by_role("button", name="Open", exact=True).click()
    page.get_by_role("option", name=option, exact=True).click()
    expect(picker.get_by_role("combobox")).to_have_value(option)


def enter(field, value):
    field.fill(value)
    field.press("Enter")


def plot(page, key):
    expect(page.locator(f".st-key-{key} .js-plotly-plot")).to_be_visible(timeout=30000)


def screenshot(page, output, name):
    if output:
        page.evaluate("window.scrollTo(0, 0); document.querySelector('[data-testid=stMain]').scrollTop=0")
        page.mouse.move(2, 2)
        page.screenshot(path=output / name, full_page=True, animations="disabled")


def run(browser, url, private_output):
    page = browser.new_page(viewport={"width": 1440, "height": 1040})
    page.set_default_timeout(15000)
    page_errors = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.goto(url, wait_until="domcontentloaded")
    heading(page, "Command Center")
    text(page, "0.56989", exact=True)
    text(page, "0.1938110143", exact=True)
    text(page, "3,378 matching")
    page.get_by_text("Chronological partitions", exact=True).click()
    expect(page.get_by_test_id("stDataFrame")).to_have_count(2)
    screenshot(page, private_output, "command-center-local.png")
    search = page.get_by_role("textbox", name="Search transaction ID")
    enter(search, "abc")
    text(page, "Enter a numeric transaction ID.")
    enter(search, "1")
    text(page, "No transactions match")
    enter(search, "")
    text(page, "3,378 matching")
    choose(page, "v3_queue_decision", "Below threshold")
    text(page, "114,730 matching")
    choose(page, "v3_queue_product", "W")
    choose(page, "v3_queue_order", "Largest amount")
    enter(page.get_by_role("spinbutton", name="Minimum risk"), "0.50")
    text(page, "No transactions match")
    enter(page.get_by_role("spinbutton", name="Minimum risk"), "0")
    expect(page.get_by_role("button", name="Open investigation")).to_be_visible()
    choose(page, "v3_queue_decision", "Flagged")
    choose(page, "v3_queue_product", "All products")
    choose(page, "v3_queue_order", "Highest risk")
    enter(page.get_by_role("spinbutton", name="Queue page"), "2")
    text(page, "showing 13")
    tx_id = page.get_by_role("combobox", name="Transaction to investigate").input_value()
    enter(search, tx_id)
    text(page, "1 matching")
    page.get_by_role("button", name="Open investigation").click()
    heading(page, "Transaction Investigator")
    heading(page, f"Transaction {tx_id}")
    heading(page, "Behavioral context")
    text(page, "Frozen V3 risk score", exact=True)
    expect(page.get_by_text(re.compile(r"^Observed label: (Fraud|Legitimate)$"))).to_have_count(0)
    page.get_by_text("Reveal observed label", exact=True).click()
    expect(page.get_by_text(re.compile(r"^Observed label: (Fraud|Legitimate)$"))).to_be_visible()
    page.get_by_role("button", name="Calculate TreeSHAP").click()
    plot(page, "v3_local_shap")
    text(page, "Score reconstruction error")
    screenshot(page, private_output, "investigator-local.png")
    if private_output:
        page.locator(".st-key-v3_local_shap .js-plotly-plot").screenshot(path=private_output / "investigator-shap-local.png")
    enter(page.get_by_role("textbox", name="Transaction ID", exact=True), "1")
    text(page, "outside the saved retrospective holdout")
    enter(page.get_by_role("textbox", name="Transaction ID", exact=True), tx_id)
    heading(page, f"Transaction {tx_id}")
    clean(page)
    print("PASS Command Center: metrics, chronology, decision/product/minimum/sort, pagination, exact and invalid search, case handoff")
    print("PASS Transaction Investigator: lookup, behavior, label toggle, native V3 TreeSHAP and score reconstruction")

    page.get_by_role("switch", name="Hide local case details").press("Space")
    text(page, "Local case details are hidden")
    expect(page.get_by_role("textbox", name="Transaction ID", exact=True)).to_have_count(0)
    heading(page, "Investigation workflow")
    radio(page, "Command Center")
    heading(page, "Command Center")
    text(page, "Frauds flagged", exact=True)
    expect(page.get_by_role("button", name="Open investigation")).to_have_count(0)

    radio(page, "Model Intelligence")
    heading(page, "Model Intelligence")
    heading(page, "Final V3 model comparison")
    heading(page, "Precision–recall")
    expect(page.get_by_test_id("stPlotlyChart")).to_have_count(2)
    text(page, "improved AP by +0.05842")
    for label in ("ROC", "Calibration", "Confusion matrix"):
        radio(page, label)
        if label != "Confusion matrix":
            heading(page, label)
            expect(page.get_by_test_id("stPlotlyChart")).to_have_count(2)
    text(page, "cutoff 0.1938110143")
    plot(page, "v3_confusion")
    page.locator(".st-key-v3_operating_point").get_by_text("0.5 for all models", exact=True).click()
    text(page, "cutoff 0.5000000000")
    choose(page, "v3_confusion_model", "Random Forest")
    heading(page, "Random Forest · confusion matrix")
    page.locator(".st-key-v3_models_view").get_by_text("Development evidence", exact=True).click()
    heading(page, "Neural network")
    text(page, "224 → 192 → 96 → 48 → 1", exact=True)
    text(page, "67,105", exact=True)
    heading(page, "Development experiments")
    expect(page.get_by_test_id("stPlotlyChart")).to_have_count(2)
    page.get_by_role("switch", name="Open development threshold sandbox").press("Space")
    text(page, "Frozen cutoff would review 575")
    page.get_by_role("slider").press("End")
    text(page, "Explorer reviews", exact=True)
    text(page, "0.1938110143", exact=True)
    clean(page)
    print("PASS Model Intelligence: PR, ROC, calibration, confusion model/cutoff, V1/V2/V3, MLP learning curves, development sandbox")

    radio(page, "Behavioral Analytics")
    heading(page, "Behavioral Analytics")
    page.get_by_role("button", name="Calculate global TreeSHAP").click()
    plot(page, "v3_global_shap")
    text(page, "24 rows explained")
    radio(page, "Isolation Forest")
    plot(page, "v3_isolation")
    radio(page, "Reused retrospective holdout")
    text(page, "118,108 comparison rows")
    radio(page, "DBSCAN & PCA")
    plot(page, "v3_cluster_aggregate")
    page.get_by_role("switch", name="Hide local case details").press("Space")
    plot(page, "v3_pca")
    page.locator(".st-key-v3_cluster_groups [role=combobox]").fill("Noise")
    page.get_by_role("option", name="Noise", exact=True).click()
    expect(page.get_by_test_id("stPlotlyChart")).to_have_count(1)
    radio(page, "Apriori rules")
    heading(page, "Apriori association rules")
    search_rules = page.get_by_role("textbox", name="Search rule conditions")
    enter(search_rules, "zzzz")
    text(page, "No saved rules")
    enter(search_rules, "")
    expect(page.locator(".st-key-v3_rule")).to_be_visible()
    radio(page, "V1 global SHAP")
    plot(page, "v3_v1_shap")
    archive = page.get_by_text("Archived V2 development patterns", exact=True)
    expect(archive).to_have_count(1)
    archive.click()
    page.get_by_role("switch", name="Load V2 pattern summary").press("Space")
    text(page, "V2 · three expanding development folds")
    clean(page)
    assert not page_errors, page_errors
    print("PASS Behavioral Analytics: exact V3 global SHAP, Isolation Forest cohorts, DBSCAN/PCA group filter, Apriori search, V1 SHAP, V2 archive")

    mobile = browser.new_page(viewport={"width": 430, "height": 900}, is_mobile=True, has_touch=True)
    mobile.goto(url, wait_until="domcontentloaded")
    heading(mobile, "Command Center")
    heading(mobile, "Transaction review queue")
    assert mobile.evaluate("document.documentElement.scrollWidth") <= 431
    assert mobile.get_by_role("heading", name="Command Center").bounding_box()["width"] < 430
    mobile.get_by_role("button", name="Expand sidebar", exact=True).tap()
    expect(mobile.get_by_test_id("stSidebar")).to_have_attribute("data-collapsed", "false")
    mobile.get_by_text("Model Intelligence", exact=True).tap()
    heading(mobile, "Model Intelligence")
    clean(mobile)
    print("PASS Mobile: 430 px viewport, responsive Command Center without page overflow, sidebar navigation")


def capture_aggregate_pages(browser, url, output):
    """Keep the screenshot session separate from the real-case verification."""
    page = browser.new_page(viewport={"width": 1440, "height": 1040})
    separator = "&" if "?" in url else "?"
    page.goto(f"{url}{separator}aggregate=1", wait_until="domcontentloaded")
    heading(page, "Command Center")
    text(page, "Frauds flagged", exact=True)
    expect(page.get_by_test_id("stDataFrame")).to_have_count(0)
    expect(page.get_by_role("button", name="Open investigation")).to_have_count(0)
    screenshot(page, output, "01-command-center.png")

    radio(page, "Transaction Investigator")
    heading(page, "Investigation workflow")
    expect(page.get_by_role("textbox", name="Transaction ID", exact=True)).to_have_count(0)
    expect(page.get_by_role("heading", name="Behavioral context", exact=True)).to_have_count(0)
    screenshot(page, output, "02-transaction-investigator.png")

    radio(page, "Model Intelligence")
    heading(page, "Final V3 model comparison")
    expect(page.locator(".js-plotly-plot").first).to_be_visible()
    page.set_viewport_size({"width": 1440, "height": 1450})
    screenshot(page, output, "03-model-intelligence.png")
    page.set_viewport_size({"width": 1440, "height": 1040})

    radio(page, "Behavioral Analytics")
    page.get_by_role("button", name="Calculate global TreeSHAP").click()
    plot(page, "v3_global_shap")
    text(page, "24 rows explained")
    screenshot(page, output, "04-behavioral-analytics.png")
    page.close()
    print("PASS Aggregate screenshots: four pages captured separately from the case session.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8518")
    parser.add_argument("--chrome", type=Path, required=True)
    parser.add_argument("--screenshots", type=Path)
    parser.add_argument("--local-screenshots", action="store_true", help="Save case screenshots in the ignored Step 8 artifacts directory.")
    args = parser.parse_args()
    private = Path(__file__).resolve().parents[1] / "artifacts" / "v3" / "step8-dashboard" if args.local_screenshots else None
    if private:
        private.mkdir(parents=True, exist_ok=True)
    if args.screenshots:
        args.screenshots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(args.chrome))
        print(f"Chrome {browser.version}")
        run(browser, args.url, private)
        if args.screenshots:
            capture_aggregate_pages(browser, args.url, args.screenshots)


if __name__ == "__main__":
    main()
