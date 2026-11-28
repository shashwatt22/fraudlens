"""Check the standalone two-page public demo in an ordinary browser."""

import argparse
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

from check_v3_dashboard_browser import heading, radio, screenshot, text


def verify(browser, url, directory):
    page = browser.new_page(viewport={"width": 1440, "height": 1170})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(f"{url.rstrip('/')}/?demo=0&aggregate=0", wait_until="domcontentloaded")
    heading(page, "Dashboard")
    expect(page.get_by_test_id("stSidebar").get_by_role("radio")).to_have_count(2)
    text(page, "0.56989", exact=True)
    text(page, "Threshold 0.1938110143")
    expect(page.locator(".st-key-public_confusion .js-plotly-plot")).to_be_visible()
    for chart in ("ROC", "Calibration", "Precision–recall"):
        radio(page, chart)
        expect(page.locator(".st-key-public_performance .js-plotly-plot")).to_have_count(1)
    radio(page, "Confusion matrix")
    page.get_by_text("0.5 for all models", exact=True).click()
    text(page, "Threshold 0.5000000000")
    page.get_by_text("Frozen thresholds", exact=True).click()
    text(page, "Threshold 0.1938110143")
    page.get_by_text("Research methods, neural learning and behavioral analysis", exact=True).click()
    heading(page, "Development experiments")
    heading(page, "V3 performance by proxy history")
    page.get_by_text("Research methods, neural learning and behavioral analysis", exact=True).click()
    expect(page.get_by_role("heading", name="Development experiments", exact=True)).to_have_count(0)
    page.set_viewport_size({"width": 1440, "height": 1550})
    screenshot(page, directory, "01-dashboard.png")
    page.set_viewport_size({"width": 1440, "height": 1170})
    radio(page, "Check Transaction & Alerts")
    heading(page, "Check Transaction & Alerts")
    text(page, "Transaction assessment and real email are unavailable here.")
    expect(page.locator('input[type="file"]')).to_have_count(0)
    expect(page.get_by_role("spinbutton")).to_have_count(0)
    for name in ("Send Test Email", "Activate real delivery", "Assess Transaction"):
        expect(page.get_by_role("button", name=name, exact=True)).to_have_count(0)
    screenshot(page, directory, "02-check-transaction.png")
    expect(page.get_by_test_id("stException")).to_have_count(0)
    assert not errors, errors
    print("PASS standalone Dashboard and Check Transaction & Alerts, aggregate charts and public isolation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8521")
    parser.add_argument("--chrome", type=Path, required=True)
    parser.add_argument("--screenshots", type=Path)
    args = parser.parse_args()
    if args.screenshots:
        args.screenshots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(args.chrome))
        print(f"Chrome {browser.version}")
        verify(browser, args.url, args.screenshots)
