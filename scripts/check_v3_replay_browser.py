"""Verify the local Step 9 simulator in Chrome and optionally capture screenshots."""

import argparse
from pathlib import Path
import re

from playwright.sync_api import expect, sync_playwright

from check_v3_dashboard_browser import clean, heading, plot, radio, screenshot, text


def value(page, label):
    return page.locator(".fl-card").filter(has=page.get_by_text(label, exact=True)).locator(".fl-card-value")


def count(page, label="Processed transactions"):
    return int(value(page, label).inner_text().replace(",", ""))


def check(page, label, expected):
    expect(value(page, label)).to_have_text(str(expected), timeout=15_000)


def button(page, label):
    return page.get_by_role("button", name=label, exact=True)


def run(browser, url, local_output):
    page = browser.new_page(viewport={"width": 1440, "height": 1280})
    page.set_default_timeout(15_000)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(url, wait_until="domcontentloaded")
    heading(page, "Command Center")
    radio(page, "Live Monitor")
    heading(page, "Live Monitor")
    text(page, "Historical replay", exact=True)
    text(page, "0.1938110143", exact=True)
    check(page, "Replay status", "Ready")
    check(page, "Processed transactions", 0)
    check(page, "Flagged transactions", 0)
    expect(button(page, "Pause")).to_be_disabled()

    for step, flags in enumerate((0, 0, 1, 2), start=1):
        button(page, "Step one").click()
        check(page, "Processed transactions", step)
        check(page, "Flagged transactions", flags)
    check(page, "Replay status", "Paused")
    plot(page, "v3_replay_risk")
    expect(page.get_by_text("Reveal observed label", exact=True)).to_have_count(0)
    picker = page.get_by_role("combobox", name="Flagged transaction")
    expect(picker).to_have_value(re.compile(r"Transaction \d+ · row (3|4)$"))
    picker_root = page.locator(".st-key-v3_replay_pick")
    picker_root.get_by_role("button", name="Open", exact=True).click()
    options = page.get_by_role("option")
    expect(options).to_have_count(2)
    assert len(set(options.all_text_contents())) == 2
    chosen = page.get_by_role("option", name=re.compile(r"Transaction [0-9]+ · row 3$"))
    selected = re.search(r"[0-9]+", chosen.inner_text()).group()
    chosen.click()
    expect(picker).to_have_value(f"Transaction {selected} · row 3")
    screenshot(page, local_output, "live-monitor-classroom-local.png")

    button(page, "Investigate flagged transaction").click()
    heading(page, "Transaction Investigator")
    heading(page, f"Transaction {selected}")
    text(page, "replay inference matches the frozen case score")
    expect(page.get_by_text(re.compile(r"^Observed label: (Fraud|Legitimate)$"))).to_have_count(0)
    button(page, "Calculate TreeSHAP").click()
    plot(page, "v3_local_shap")
    text(page, "Score reconstruction error")
    if local_output:
        page.locator(".st-key-v3_local_shap .js-plotly-plot").screenshot(path=local_output / "replay-shap-local.png")
    page.get_by_text("Reveal observed label", exact=True).click()
    expect(page.get_by_text(re.compile(r"^Observed label: (Fraud|Legitimate)$"))).to_be_visible()
    radio(page, "Live Monitor")
    check(page, "Replay status", "Paused")
    check(page, "Processed transactions", 4)
    clean(page)
    print("PASS Classroom steps: two flags among four real transactions, no duplicate queue entries, investigator/SHAP and explicit label reveal")

    button(page, "Reset").click()
    check(page, "Replay status", "Ready")
    check(page, "Processed transactions", 0)
    check(page, "Flagged transactions", 0)
    expect(picker).to_have_count(0)
    button(page, "Start").click()
    check(page, "Replay status", "Running")
    expect(value(page, "Processed transactions")).not_to_have_text("0")
    button(page, "Pause").click()
    check(page, "Replay status", "Paused")
    before = count(page)
    page.wait_for_timeout(2300)  # Longer than two fragment intervals.
    check(page, "Processed transactions", before)

    page.get_by_role("slider").press("End")
    text(page, "Speed setting · 100 transactions/s")
    check(page, "Processed transactions", before)
    button(page, "Resume").click()
    check(page, "Replay status", "Running")
    expect(value(page, "Processed transactions")).not_to_have_text(str(before))
    advanced = count(page)
    assert advanced > before and (advanced - before) % 100 == 0
    button(page, "Investigate flagged transaction").click()
    heading(page, "Transaction Investigator")
    text(page, "replay inference matches the frozen case score")
    radio(page, "Live Monitor")
    check(page, "Replay status", "Paused")
    button(page, "Resume").click()
    check(page, "Replay status", "Running")
    radio(page, "Command Center")
    heading(page, "Command Center")
    radio(page, "Live Monitor")
    check(page, "Replay status", "Paused")
    assert count(page) >= advanced
    button(page, "Reset").click()
    check(page, "Processed transactions", 0)
    check(page, "Flagged transactions", 0)
    clean(page)
    assert not errors, errors
    print("PASS Timed replay: start, pause, idle refresh, speed change, resume, automatic pause on investigation and navigation, and reset")
    page.close()


def capture(browser, url, output):
    page = browser.new_page(viewport={"width": 1440, "height": 1040})
    separator = "&" if "?" in url else "?"
    page.goto(f"{url}{separator}aggregate=1", wait_until="domcontentloaded")
    heading(page, "Command Center")
    radio(page, "Live Monitor")
    heading(page, "Live Monitor")
    expect(button(page, "Start")).to_be_disabled()
    expect(button(page, "Step one")).to_be_disabled()
    expect(page.get_by_test_id("stDataFrame")).to_have_count(0)
    expect(page.get_by_role("heading", name="Transaction feed")).to_have_count(0)
    text(page, "Local transaction feed and review details are hidden")
    clean(page)
    screenshot(page, output, "05-live-monitor.png")
    print("PASS Aggregate launch: replay controls disabled and no transaction records rendered")
    page.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8519")
    parser.add_argument("--chrome", type=Path, required=True)
    parser.add_argument("--screenshots", type=Path)
    parser.add_argument("--local-screenshots", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    local = root / "artifacts" / "v3" / "step9-replay" if args.local_screenshots else None
    for folder in (args.screenshots, local):
        if folder:
            folder.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(args.chrome))
        print(f"Chrome {browser.version}")
        run(browser, args.url, local)
        capture(browser, args.url, args.screenshots)


if __name__ == "__main__":
    main()
