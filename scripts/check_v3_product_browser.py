"""Verify the two-page workspace with real local inference and dry-run email."""

import argparse
from pathlib import Path
import re

from playwright.sync_api import expect, sync_playwright

from check_v3_dashboard_browser import heading, radio, screenshot, text
from check_v3_replay_browser import button


def card(page, name, value):
    target = page.locator(".fl-card").filter(has=page.get_by_text(name, exact=True)).locator(".fl-card-value")
    expect(target).to_have_text(value)


def verify(browser, url, output):
    page = browser.new_page(viewport={"width": 1510, "height": 1100})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(url, wait_until="domcontentloaded")
    heading(page, "Dashboard")
    options = page.get_by_test_id("stSidebar").get_by_role("radio")
    expect(options).to_have_count(2)
    text(page, "Check Transaction & Alerts", exact=True)
    card(page, "Historical average precision", "0.56989")
    card(page, "Historical ROC-AUC", "0.91798")
    card(page, "Historical F1", "0.55254")
    card(page, "Manual assessments", "0")
    text(page, "+0.38211 AP from V1 to V3")
    text(page, "Threshold 0.1938110143")
    expect(page.locator(".st-key-public_confusion .js-plotly-plot")).to_be_visible()
    radio(page, "Precision–recall")
    expect(page.locator(".st-key-public_performance .js-plotly-plot")).to_be_visible()
    radio(page, "Confusion matrix")
    expect(page.locator(".st-key-public_confusion .js-plotly-plot")).to_be_visible()
    heading(page, "Manual model validation")
    text(page, "23,622 later development rows")
    card(page, "Average precision", "0.15605")
    card(page, "Precision", "22.60%")
    card(page, "Recall", "23.01%")
    page.get_by_text("Research methods, learning curves and supplementary analysis", exact=True).click()
    heading(page, "Development experiments")
    page.get_by_text("Research methods, learning curves and supplementary analysis", exact=True).click()
    expect(page.get_by_role("heading", name="Development experiments", exact=True)).to_have_count(0)

    radio(page, "Check Transaction & Alerts")
    heading(page, "Check Transaction & Alerts")
    expect(button(page, "Activate real delivery")).to_have_count(0)
    text(page, "Authorized recipient: unavailable.")
    button(page, "Assess Transaction").click()
    text(page, re.compile(r"Enter (a valid amount|an amount between)"))
    page.get_by_text("Use a prefilled example", exact=True).click()
    button(page, "Debit · $49.95").click()
    expect(page.get_by_role("spinbutton", name="Amount (USD)")).to_have_value("49.95")
    button(page, "Assess Transaction").click()
    card(page, "Model risk score", "2.05%")
    card(page, "Decision", "Not flagged")
    text(page, "Continue standard checks")
    button(page, "Credit · $899.99").click()
    expect(page.get_by_role("spinbutton", name="Amount (USD)")).to_have_value("899.99")
    button(page, "Assess Transaction").click()
    card(page, "Model risk score", "18.45%")
    card(page, "Decision", "Flagged")
    text(page, "Hold for manual review")
    expect(page.get_by_text(re.compile(r"MA-[A-F0-9]{16}.*UTC")).first).to_be_visible()
    shap = page.locator(".st-key-fl_manual_shap .js-plotly-plot")
    expect(shap).to_have_count(1)
    expect(shap).to_be_visible()
    heading(page, "Recent assessments")
    page.get_by_text("Optional context", exact=True).click()
    page.locator(".st-key-fl_device").get_by_text("Desktop or laptop", exact=True).click()
    page.locator(".st-key-fl_recipient").get_by_text("Microsoft", exact=True).click()
    button(page, "Assess Transaction").click()
    card(page, "Model risk score", "9.04%")
    card(page, "Decision", "Not flagged")
    text(page, "Latest notification · Dry run")
    button(page, "Send Test Email").click()
    text(page, "Latest notification · Dry run")
    button(page, "Disable alerts").click()
    expect(button(page, "Send Test Email")).to_be_disabled()
    text(page, "Disabled · 0/5 real attempts")
    button(page, "Dry run").click()
    text(page, "Dry run · 0/5 real attempts")
    page.set_viewport_size({"width": 1510, "height": 1310})
    heading(page, "Notification history")
    page.set_viewport_size({"width": 1510, "height": 1600})
    screenshot(page, output, "02-check-transaction.png")

    radio(page, "Dashboard")
    card(page, "Manual assessments", "3")
    card(page, "Notifications", "2")
    card(page, "Emails accepted", "0")
    heading(page, "Manual model validation")
    expect(page.locator(".st-key-public_versions .js-plotly-plot")).to_be_visible()
    page.set_viewport_size({"width": 1510, "height": 1750})
    screenshot(page, output, "01-dashboard.png")
    assert not errors, errors
    expect(page.get_by_test_id("stException")).to_have_count(0)
    print("PASS desktop: verified research metrics and plots, both real assessments, SHAP, history, and dry-run alerts")
    page.close()

    guest = browser.new_page(viewport={"width": 430, "height": 900}, is_mobile=True, has_touch=True)
    guest.goto(f"{url.rstrip('/')}/?demo=1", wait_until="domcontentloaded")
    heading(guest, "Dashboard")
    assert guest.evaluate("document.documentElement.scrollWidth") <= 431
    guest.get_by_role("button", name="Expand sidebar", exact=True).tap()
    expect(guest.get_by_test_id("stSidebar")).to_have_attribute("data-collapsed", "false")
    guest.get_by_text("Check Transaction & Alerts", exact=True).tap()
    heading(guest, "Check Transaction & Alerts")
    text(guest, "Transaction assessment and real email are unavailable here.")
    expect(guest.get_by_role("spinbutton")).to_have_count(0)
    expect(button(guest, "Send Test Email")).to_have_count(0)
    print("PASS mobile: two-page navigation and public-demo isolation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8525")
    parser.add_argument("--chrome", type=Path, required=True)
    parser.add_argument("--screenshots", type=Path)
    args = parser.parse_args()
    if args.screenshots:
        args.screenshots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(args.chrome))
        print(f"Chrome {browser.version}")
        verify(browser, args.url, args.screenshots)
