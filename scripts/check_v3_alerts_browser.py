"""Check local dry-run workflows and public isolation without sending email."""

import argparse
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

from check_v3_dashboard_browser import clean, heading, radio, screenshot, text
from check_v3_replay_browser import button, check


def run(browser, url, output, local_output, smtp_configured):
    page = browser.new_page(viewport={"width": 1440, "height": 1080})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(url, wait_until="domcontentloaded")
    heading(page, "Command Center")
    radio(page, "Alert Center")
    heading(page, "Alert Center")
    check(page, "Notification mode", "Dry run")
    check(page, "Notifications logged", 0)
    check(page, "Real attempt quota", "0 / 5")
    if smtp_configured:
        expect(button(page, "Activate real delivery")).to_be_disabled()
        page.get_by_text("I authorize delivery to the recipient configured on this server.", exact=True).click()
        expect(button(page, "Activate real delivery")).to_be_enabled()
        page.get_by_text("I authorize delivery to the recipient configured on this server.", exact=True).click()
        expect(button(page, "Activate real delivery")).to_be_disabled()
    else:
        text(page, "Server SMTP opt-in is off.")
        expect(button(page, "Activate real delivery")).to_have_count(0)

    radio(page, "Live Monitor")
    text(page, "Alert Center mode · Dry run")
    for step in range(1, 5):
        button(page, "Step one").click()
        check(page, "Processed transactions", step)
    check(page, "Flagged transactions", 2)
    radio(page, "Alert Center")
    check(page, "Notifications logged", 2)
    text(page, "Latest notification · Dry run · Preview only.")
    check(page, "SMTP accepted", 0)
    expect(page.get_by_test_id("stDataFrame")).to_have_count(1)
    button(page, "Send Test Email").click()
    check(page, "Notifications logged", 3)
    check(page, "Real attempt quota", "0 / 5")
    screenshot(page, local_output, "alert-center-local.png")
    radio(page, "Command Center")
    heading(page, "Command Center")
    radio(page, "Alert Center")
    check(page, "Notifications logged", 3)
    button(page, "Disable alerts").click()
    check(page, "Notification mode", "Disabled")
    expect(button(page, "Send Test Email")).to_be_disabled()
    radio(page, "Live Monitor")
    text(page, "Alert Center mode · Disabled")
    button(page, "Reset").click()
    check(page, "Processed transactions", 0)
    for step in range(1, 4):
        button(page, "Step one").click()
        check(page, "Processed transactions", step)
    check(page, "Flagged transactions", 1)
    radio(page, "Alert Center")
    check(page, "Notifications logged", 4)
    check(page, "Notification mode", "Disabled")
    text(page, "Latest notification · Disabled")
    check(page, "Real attempt quota", "0 / 5")
    button(page, "Dry run").click()
    check(page, "Notification mode", "Dry run")
    clean(page)
    assert not errors, errors
    print("PASS Chrome local: real replay flags, dry-run test, no duplicate records on navigation, disabled mode, and reset")
    if smtp_configured:
        print("PASS Chrome opt-in: real activation requires explicit confirmation and was left inactive")
    page.close()

    public = browser.new_page(viewport={"width": 1440, "height": 1020})
    separator = "&" if "?" in url else "?"
    public.goto(f"{url}{separator}demo=1", wait_until="domcontentloaded")
    heading(public, "Command Center")
    radio(public, "Alert Center")
    heading(public, "Alert Center")
    text(public, "Dry run only.")
    heading(public, "Email preview")
    for label in ("Activate real delivery", "Send Test Email", "Disable alerts"):
        expect(button(public, label)).to_have_count(0)
    expect(public.get_by_role("switch", name="Hide local case details")).to_have_count(0)
    expect(public.get_by_test_id("stDataFrame")).to_have_count(0)
    screenshot(public, output, "06-alert-center.png")
    radio(public, "Live Monitor")
    expect(button(public, "Start")).to_be_disabled()
    expect(public.get_by_test_id("stDataFrame")).to_have_count(0)
    clean(public)
    print("PASS Chrome public: dry-run only, delivery configuration and controls hidden, no local records")
    public.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8520")
    parser.add_argument("--chrome", type=Path, required=True)
    parser.add_argument("--screenshots", type=Path)
    parser.add_argument("--local-screenshots", action="store_true")
    parser.add_argument("--smtp-configured", action="store_true",
                        help="Check activation UI with reserved dummy settings without activating delivery.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    local = root / "artifacts" / "v3" / "step10-alerts" if args.local_screenshots else None
    for folder in (args.screenshots, local):
        if folder:
            folder.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(args.chrome))
        print(f"Chrome {browser.version}")
        run(browser, args.url, args.screenshots, local, args.smtp_configured)


if __name__ == "__main__":
    main()
