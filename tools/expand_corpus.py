"""Write the longer near-duplicate pages added in Phase 1.5.

Re-running this script rewrites the files it owns. Hand-written pages are left
as they are, except the six that gain a parallel tail. The tail is replaced
on each run, starting at a fixed heading, so the opening fact stays put.
"""

from __future__ import annotations

import json
from pathlib import Path

from supportdesk.config import get_settings
from supportdesk.ingest.loaders import load_corpus
from supportdesk.ingest.prepare import prepare_documents
from supportdesk.ingest.tokenize import HuggingFaceCodec, token_count

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "corpus"
FORBIDDEN = ("student", "mesh", "fax", "referral", "airline")
CANARIES = (
    "FINANCE_CANARY_7K2M",
    "AGENT_CANARY_4QP9",
    "WRITEOFF_CANARY_8HN4",
    "EXECUTIVE_CANARY_2VX6",
)

PAD = (
    " Staff write the clock time and the ticket id on the note for {label}. "
    "They compare {label} with the neighboring page before they quote it. "
    "The neighboring page uses the same steps and a different figure. "
    "Quoting the neighboring figure on a {label} note is a misread. "
    "The note stays on the account for 24 months. "
    "A later reader looks for {label} and does not copy the note onto another page."
)


def pad(codec: HuggingFaceCodec, text: str, label: str) -> str:
    extra = PAD.format(label=label)
    while token_count(codec, text) < 280:
        candidate = text.rstrip() + extra
        count = token_count(codec, candidate)
        if count > 390:
            if token_count(codec, text) < 220:
                text = candidate
            break
        text = candidate
    return text.strip()


def section(codec: HuggingFaceCodec, heading: str, body: str, label: str) -> str:
    return f"## {heading}\n\n{pad(codec, body.strip(), label)}\n"


def document(level: str, slug: str, title: str, body: str) -> str:
    return (
        "---\n"
        "synthetic: true\n"
        "company: Northline Telecom\n"
        "access_level: "
        f"{level}\n"
        "doc_slug: "
        f"{slug}\n"
        "title: "
        f"{title}\n"
        "---\n\n"
        f"{body.strip()}\n"
    )


def write_new(level: str, slug: str, title: str, body: str) -> None:
    path = CORPUS / level / f"{slug}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document(level, slug, title, body), encoding="utf-8", newline="\n")


def append_tail(relative: str, first_heading: str, tail: str) -> None:
    path = CORPUS / relative
    text = path.read_text(encoding="utf-8")
    marker = f"\n## {first_heading}\n"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    path.write_text(text.rstrip() + "\n\n" + tail.strip() + "\n", encoding="utf-8", newline="\n")


def fiber_tail(codec: HuggingFaceCodec, spec: dict) -> str:
    label = f"{spec['tier']} at {spec['price']} dollars"
    neighbor = spec["neighbor"]
    blocks = [
        (
            "Neighboring tier",
            f"The monthly figure on the {spec['tier']} page is {spec['price']} dollars. "
            f"The neighboring tier is {neighbor} at {spec['neighbor_price']} dollars. "
            f"A household comparing the two should read both pages. "
            f"The {spec['price']} dollar figure is only the {spec['tier']} recurring wireline charge. "
            f"Taxes are extra on both pages. "
            f"A quote that mixes {spec['price']} dollars with the {neighbor} speed is a misread.",
        ),
        (
            "Wired speed test",
            f"{spec['tier']} is advertised at {spec['down']} Mbps toward the home and "
            f"{spec['up']} Mbps toward the network. "
            "The desk measures that on a cable plugged into the gateway, not over the air. "
            f"A result at or above 80 percent of {spec['down']} Mbps is treated as inside the home. "
            "A result below that line can become a plant ticket. "
            f"Do not quote the {neighbor} speed of {spec['neighbor_speed']} Mbps on a {spec['tier']} ticket.",
        ),
        (
            "Included gateway",
            f"The residential gateway on {spec['tier']} stays with the household for the length of the subscription. "
            f"{spec['static']} "
            "The same metal box is used on the neighboring tier, and the box does not tell you the price. "
            "Price and speed come from the plan page, not from the label on the box. "
            f"Returning the box follows the equipment page and is not a {spec['tier']} credit.",
        ),
        (
            "What this page is not",
            f"This page is not the {neighbor} page and it is not a mobile plan. "
            "It does not include a handset or a second address. "
            f"{spec['install']} "
            "An autopay reduction, if any, is a different page and is not subtracted inside this price. "
            "One-time work is its own line and is not part of the monthly figure.",
        ),
    ]
    return "\n".join(section(codec, heading, body, label) for heading, body in blocks)


def fiber_opening(spec: dict) -> str:
    return (
        f"# {spec['title']}\n\n"
        f"The {spec['tier']} plan costs {spec['price']} dollars per month. "
        f"It is a residential wireline tier and is not {spec['neighbor']}. "
        f"Download and upload peaks are {spec['down']} Mbps and {spec['up']} Mbps. "
        f"{spec['static']}"
    )


FIBER = [
    {
        "slug": "fiber-500-plan",
        "existing": True,
        "title": "Fiber 500 residential plan",
        "tier": "Fiber 500",
        "price": 79,
        "down": 500,
        "up": 500,
        "neighbor": "Fiber 1000",
        "neighbor_price": 99,
        "neighbor_speed": 1000,
        "static": "No static address is included on this tier.",
        "install": "A truck is not required when the optical terminal already lights this tier.",
    },
    {
        "slug": "fiber-1000-plan",
        "existing": True,
        "title": "Fiber 1000 residential plan",
        "tier": "Fiber 1000",
        "price": 99,
        "down": 1000,
        "up": 1000,
        "neighbor": "Fiber 500",
        "neighbor_price": 79,
        "neighbor_speed": 500,
        "static": "No static address is included on this tier.",
        "install": "Households already on the slower tier can move up without a truck when the optical terminal supports it.",
    },
    {
        "slug": "fiber-300-plan",
        "existing": False,
        "title": "Fiber 300 residential plan",
        "tier": "Fiber 300",
        "price": 59,
        "down": 300,
        "up": 300,
        "neighbor": "Fiber 500",
        "neighbor_price": 79,
        "neighbor_speed": 500,
        "static": "No static address is included on this tier.",
        "install": "A truck is scheduled only when no optical terminal is already in the home.",
    },
    {
        "slug": "fiber-2000-plan",
        "existing": False,
        "title": "Fiber 2000 residential plan",
        "tier": "Fiber 2000",
        "price": 129,
        "down": 2000,
        "up": 2000,
        "neighbor": "Fiber 1000",
        "neighbor_price": 99,
        "neighbor_speed": 1000,
        "static": "One static address is included on this tier and on no slower tier.",
        "install": "A truck is required when the optical terminal is older than the 2024 shelf.",
    },
]


def outage_tail(codec: HuggingFaceCodec, spec: dict) -> str:
    label = f"{spec['name']} outage credit of {spec['size']}"
    blocks = [
        (
            "How the request is opened",
            f"A {spec['name']} outage credit starts from {spec['opened']}. "
            f"The qualifying continuous gap is {spec['hours']} hours, which is not the figure on the neighboring page. "
            f"The size of the credit is {spec['size']}. "
            "Planned work that was announced at least 24 hours ahead does not qualify on this page. "
            f"{spec['who']} "
            "Cash is not offered on this page.",
        ),
        (
            "What gets confused with the neighboring page",
            f"The neighboring page is {spec['neighbor']}. "
            f"That page uses a different size ({spec['neighbor_size']}) and a different audience. "
            f"Quoting {spec['neighbor_size']} on a {spec['name']} note is a misread. "
            f"This page's size stays {spec['size']} even when the caller asks for the other figure. "
            "Banner id, clock time, and plan name are written on the ticket before the credit is discussed.",
        ),
        (
            "What does not qualify",
            "A gap shorter than the figure on this page does not qualify. "
            "A single television app, a handset on wireless, or a gateway that was unplugged does not qualify. "
            "The credit is service days or a stated fraction of a day, posted on a later invoice. "
            f"It is not a refund of a payment, and it is not the {spec['neighbor']} rule.",
        ),
    ]
    return "\n".join(section(codec, heading, body, label) for heading, body in blocks)


OUTAGE = [
    {
        "slug": "outage-credit-customer",
        "level": "public",
        "existing": True,
        "title": "Outage credit for residential customers",
        "name": "residential customer",
        "hours": 12,
        "size": "one day of the plan recurring charge",
        "neighbor": "the agent outage-credit page",
        "neighbor_size": "at most 3 days",
        "opened": "the outage banner or a billing ticket the household opens",
        "who": "The household requests it. This page does not state what frontline staff may apply alone.",
    },
    {
        "slug": "outage-credit-agent",
        "level": "agent",
        "existing": True,
        "title": "Agent outage credits",
        "name": "residential agent",
        "hours": 12,
        "size": "at most 3 days",
        "neighbor": "the customer outage-credit page",
        "neighbor_size": "one day, and only when the household asks",
        "opened": "an open banner or a closed banner whose duration met the line",
        "who": "Frontline staff may apply the credit without the ledger desk. Above that size, the ticket goes to the ledger desk with the banner id attached.",
    },
    {
        "slug": "outage-credit-business",
        "level": "public",
        "existing": False,
        "title": "Outage credit for small business customers",
        "name": "small-business customer",
        "hours": 8,
        "size": "half a day of the business recurring charge",
        "neighbor": "the residential customer outage page",
        "neighbor_size": "one full day after 12 hours",
        "opened": "the business banner in the account app",
        "who": "The business account holder requests it. Residential rules are not used on a business account.",
    },
    {
        "slug": "outage-credit-business-agent",
        "level": "agent",
        "existing": False,
        "title": "Agent outage credits for small business",
        "name": "small-business agent",
        "hours": 8,
        "size": "at most 2 days",
        "neighbor": "the residential agent outage page",
        "neighbor_size": "at most 3 days",
        "opened": "a business banner id",
        "who": "Frontline staff may apply at most 2 days on a business banner. A larger request goes to the ledger desk. Do not apply the residential 3 day cap on a business account.",
    },
]


def refund_tail(codec: HuggingFaceCodec, spec: dict) -> str:
    label = spec["label"]
    blocks = [
        (
            "What a refund is not",
            f"{spec['not_a']} "
            "A service credit reduces a later invoice and is described on a different page. "
            f"The neighboring page is {spec['neighbor']}. "
            f"{spec['neighbor_rule']} "
            "Mixing the two pages produces the wrong answer.",
        ),
        (
            "What the ticket must contain",
            f"{spec['ticket']} "
            "The invoice id and the charge line are required before anyone reviews the request. "
            "A request without those two fields is sent back. "
            f"{spec['audience_limit']}",
        ),
        (
            "Timing versus approval",
            f"{spec['timing']} "
            "Timing and approval are different questions. "
            "A page that states a clock does not state who may approve, and a page that states approval does not promise a clock unless it says so.",
        ),
    ]
    return "\n".join(section(codec, heading, body, label) for heading, body in blocks)


REFUND = [
    {
        "slug": "refund-request-customer",
        "level": "public",
        "existing": True,
        "title": "How a customer requests a refund",
        "label": "customer refund request",
        "not_a": "This page tells a household how to ask. It does not list approval amounts.",
        "neighbor": "the finance refund matrix",
        "neighbor_rule": "That matrix is not on this page, and its desk code is not repeated here.",
        "ticket": "The household names the invoice and the charge in the app, or asks frontline staff to open a billing ticket.",
        "audience_limit": "The household is told the ticket number and can ask for status with that number.",
        "timing": "This page does not promise a number of days.",
    },
    {
        "slug": "refund-authority",
        "level": "finance",
        "existing": True,
        "title": "Refund authority matrix",
        "label": "finance refund matrix",
        "not_a": "This matrix is approval authority for a posted payment. It is not a service credit.",
        "neighbor": "the customer refund-request page",
        "neighbor_rule": "The customer page has no dollar caps. This page does: 25 dollars for a specialist, 100 dollars for an analyst, and the finance director above that.",
        "ticket": "The ticket must include the invoice id and the charge line before an approval is recorded.",
        "audience_limit": "Frontline staff do not see this matrix and must not be read the approval code.",
        "timing": "Approval on this page is not the same as the posting clock on the ledger timing page.",
    },
]


def refund_new(codec: HuggingFaceCodec) -> None:
    customer = (
        "# When a reviewed refund shows up\n\n"
        "A reviewed refund of a posted payment shows on the account within 7 business days. "
        "The clock starts when the review is finished, not when the household asked. "
        "This page does not say who is allowed to approve the review.\n\n"
        + refund_tail(
            codec,
            {
                "label": "7 business day refund posting",
                "not_a": "Seven business days is a posting clock for the household. It is not an approval cap.",
                "neighbor": "the finance same-day posting page",
                "neighbor_rule": "The ledger page describes the day the approval is recorded, which is a different clock from these 7 business days.",
                "ticket": "The household can see the ticket number but not the approval code.",
                "audience_limit": "No dollar limit appears on this page.",
                "timing": "Count 7 business days from the end of review. Do not count from the day the household tapped the button.",
            },
        )
    )
    finance = (
        "# When the ledger posts an approved reversal\n\n"
        "The ledger posts an approved reversal on the same business day the approval is recorded. "
        "That clock is not the 7 business days a household is told. "
        "This page does not replace the approval matrix.\n\n"
        + refund_tail(
            codec,
            {
                "label": "same-business-day ledger posting",
                "not_a": "Same-day posting is a ledger clock. It is not permission to approve.",
                "neighbor": "the household 7 business day page",
                "neighbor_rule": "Quoting 7 business days on a ledger note, or quoting same-day posting to a household, is a misread.",
                "ticket": "The ledger entry names the approval id, the invoice id, and the charge line.",
                "audience_limit": "The approval matrix still sets the dollar cap. This page only sets the posting clock.",
                "timing": "Post on the same business day the approval is recorded. Do not wait 7 business days on the ledger.",
            },
        )
    )
    write_new("public", "refund-timing", "When a reviewed refund shows up", customer)
    write_new("finance", "refund-posting", "When the ledger posts an approved reversal", finance)


def money_confusers(codec: HuggingFaceCodec) -> None:
    public_write = (
        "# Asking for a balance review\n\n"
        "A household asks for a balance review from the billing page in the app. "
        "This page does not say who may erase a balance, and it does not give a desk code.\n\n"
        + section(
            codec,
            "What an agent may say",
            "Frontline staff open a billing ticket and stop. They do not promise that a balance will be erased. Dollar lines live on a finance page that is not this one.",
            "customer balance review",
        )
        + section(
            codec,
            "What this is not",
            "A balance review is not a refund of a posted payment and it is not an outage credit. The household is given a ticket number and no promise of a dollar outcome.",
            "customer balance review",
        )
    )
    agent_write = (
        "# What frontline staff may say about a balance\n\n"
        "Frontline staff do not erase a balance. They open a billing ticket and stop. "
        "The dollar line and the desk code are on the finance page, which is not read aloud.\n\n"
        + section(
            codec,
            "What an agent may say",
            "Tell the caller the ticket number. Do not quote a threshold, a number of days, or a desk code. Those live on the finance write-off page.",
            "agent balance handoff",
        )
        + section(
            codec,
            "What this is not",
            "This page is not the finance write-off page. A caller who asks for the code is not given it. The ticket goes to the ledger queue.",
            "agent balance handoff",
        )
    )
    public_exec = (
        "# Asking the executive office for a courtesy adjustment\n\n"
        "A household may write to the executive office. "
        "This page does not say who must sign, and it does not give a desk code or a dollar line.\n\n"
        + section(
            codec,
            "What an agent may say",
            "Frontline staff can open a ticket and mark that the caller asked for the executive office. They cannot create the credit.",
            "customer executive request",
        )
        + section(
            codec,
            "What this is not",
            "This is not the finance page that names the signer. A service credit and a refund remain different, and neither dollar line is on this page.",
            "customer executive request",
        )
    )
    write_new("public", "balance-review-customer", "Asking for a balance review", public_write)
    write_new("agent", "balance-review-agent", "What frontline staff may say about a balance", agent_write)
    write_new(
        "public",
        "executive-courtesy-customer",
        "Asking the executive office for a courtesy adjustment",
        public_exec,
    )
    writeoff_tail = "\n".join(
        [
            section(
                codec,
                "What an agent may say",
                "Frontline staff are not given the desk code and are not asked to promise an erasure on a call. They open a billing ticket. The threshold on this page stays with the finance director.",
                "finance write-off",
            ),
            section(
                codec,
                "What is not erased",
                "An erasure does not clear a suspicion marker. The account is closed after the entry posts. A smaller balance follows the analyst path on this same page and is not the director path.",
                "finance write-off",
            ),
        ]
    )
    executive_tail = "\n".join(
        [
            section(
                codec,
                "What an agent may say",
                "Frontline staff cannot create this credit type and are not given the desk code. A forwarding note from the executive office is stored on the ticket before finance reviews it.",
                "finance executive courtesy",
            ),
            section(
                codec,
                "What is not erased",
                "A courtesy adjustment is still a service credit unless a refund was approved on the refund matrix. The signer on this page is not the signer on the write-off page, and the dollar lines differ.",
                "finance executive courtesy",
            ),
        ]
    )
    append_tail("finance/write-off-policy.md", "What an agent may say", writeoff_tail)
    append_tail("finance/executive-credits.md", "What an agent may say", executive_tail)


PLACES = [
    ("north-mill", "North Mill", "NM-14", "3.2", "Mill Night", "40"),
    ("east-yard", "East Yard", "EY-22", "2.8", "Yard Night", "35"),
    ("west-pier", "West Pier", "WP-07", "4.1", "Pier Night", "50"),
    ("south-junction", "South Junction", "SJ-18", "3.6", "Junction Night", "45"),
    ("cedar-ridge", "Cedar Ridge", "CR-03", "2.5", "Ridge Night", "30"),
    ("lake-switch", "Lake Switch", "LS-11", "3.9", "Switch Night", "55"),
    ("old-foundry", "Old Foundry", "OF-09", "4.4", "Foundry Night", "60"),
    ("brick-wharf", "Brick Wharf", "BW-16", "2.9", "Wharf Night", "25"),
    ("maple-cut", "Maple Cut", "MC-21", "3.4", "Cut Night", "42"),
    ("iron-gate", "Iron Gate", "IG-05", "4.0", "Gate Night", "38"),
]


def neighborhood_docs(codec: HuggingFaceCodec) -> None:
    for slug, place, code, db, crew, minutes in PLACES:
        public_body = (
            f"# {place} service banner\n\n"
            f"When the {place} banner {code} is up, the household checks the power light on the optical terminal before opening a ticket. "
            f"A dark power light is a local problem. A lit power light with no service is banner {code}. "
            f"The household quotes banner {code} and does not quote another spur's code.\n\n"
            + section(
                codec,
                "Recognizing the banner",
                f"Banner {code} is the {place} code only. A neighboring spur uses a different code and a different first check. "
                f"Quote {code} when the app shows that banner. Do not quote a code from a street outside {place}.",
                f"{place} banner {code}",
            )
            + section(
                codec,
                "First check",
                f"The first check in {place} is the power light, then the app banner. "
                f"If the banner is {code}, wait for the posted restoration note. "
                "A gateway that was unplugged is not this banner.",
                f"{place} banner {code}",
            )
            + section(
                codec,
                "What not to borrow",
                f"Do not borrow a crew name, a dB line, or a minute count from another spur page. "
                f"Those figures are not the {place} figures. The household page does not list the night crew.",
                f"{place} banner {code}",
            )
        )
        agent_body = (
            f"# {place} dispatch\n\n"
            f"For banner {code}, dispatch {crew} when optical loss stays above {db} dB for {minutes} minutes. "
            f"Do not dispatch {crew} for a single apartment under that line. "
            f"The {db} dB line is {place} only.\n\n"
            + section(
                codec,
                "Recognizing the banner",
                f"Banner {code} maps to {place}. A ticket that names a different banner is a different spur and a different crew. "
                f"Write {code}, the dB reading, and the clock time before paging {crew}.",
                f"{place} {code} at {db} dB",
            )
            + section(
                codec,
                "Who rolls",
                f"{crew} rolls for {place} only. The roll starts after {minutes} minutes above {db} dB. "
                f"A shorter spike does not page {crew}. A single apartment is a premises visit, not a spur roll.",
                f"{place} {code} at {db} dB",
            )
            + section(
                codec,
                "What not to borrow",
                f"Do not copy another spur's dB line onto a {code} ticket. "
                f"The household is told banner {code} and is not told the {db} dB line or the name {crew}.",
                f"{place} {code} at {db} dB",
            )
        )
        write_new("public", f"banner-{slug}", f"{place} service banner", public_body)
        write_new("agent", f"dispatch-{slug}", f"{place} dispatch", agent_body)


SYMPTOMS = [
    ("slow-evenings", "Slow only in the evening", "evenings after 19:00", "move the gateway off the floor and away from the television"),
    ("slow-all-day", "Slow all day", "every hour of the day", "run a wired measurement before any other step"),
    ("upload-only", "Slow toward the network only", "upload and not download", "read the optical terminal light before a plant ticket"),
    ("wifi-drops-hourly", "Wireless drops every hour", "a drop about once an hour", "forget the wireless network and join it again"),
    ("wired-only", "Only wired devices are slow", "wired devices and not wireless ones", "replace the house cable between the gateway and the device"),
    ("stream-buffer", "The television app buffers", "buffering on the television app", "put the streamer on a cable and do not use a phone test as evidence"),
    ("calling-drops", "Calls drop on home wireless", "calls that drop only on home wireless", "stand next to the gateway and try one call"),
    ("nightly-reboot", "Gateway restarts at night", "a restart that happens at night", "write down the clock time and do not factory-reset on the first call"),
]


def symptom_docs(codec: HuggingFaceCodec) -> None:
    for slug, title, pattern, first in SYMPTOMS:
        body = (
            f"# {title}\n\n"
            f"The pattern on this page is {pattern}. The first step is: {first}. "
            "A different pattern has a different first step, even when the household describes it with the same word.\n\n"
            + section(
                codec,
                "First step",
                f"Do this first for {pattern}: {first}. "
                "Skip this step only when the household has already done it and can say when. "
                "Do not import the first step from a neighboring symptom page.",
                f"{title}: {first}",
            )
            + section(
                codec,
                "What a neighboring page owns",
                "Evening slowness, all-day slowness, upload-only slowness, hourly drops, wired-only slowness, "
                "television buffering, home-wireless call drops, and nightly restarts each have their own page. "
                f"This page owns only {pattern}.",
                f"{title}: {first}",
            )
            + section(
                codec,
                "When to open a plant ticket",
                f"Open a plant ticket for {pattern} only after the first step is done and a wired result is still far below the plan. "
                "A premises issue stays a premises visit. Quote the pattern name in the ticket so the wrong page is not used.",
                f"{title}: {first}",
            )
        )
        write_new("public", f"symptom-{slug}", title, body)


FEES = [
    ("reconnect-fee", "Reconnect fee", "35 dollars", "one reconnect fee in a year", "Frontline staff cannot waive it."),
    ("paper-bill-fee", "Paper bill fee", "5 dollars each invoice", "a hardship arrangement only", "Frontline staff cannot waive it."),
    ("static-address-addon", "Static address add-on", "15 dollars", "only when the optical terminal cannot host one", "Frontline staff open a ticket and do not remove the charge."),
    ("vacation-hold", "Vacation hold", "10 dollars per month while the hold lasts", "90 days", "Frontline staff may start a hold of at most 30 days and cannot extend it."),
    ("residential-deposit", "Residential deposit", "100 dollars", "after 12 invoices paid on time", "Frontline staff cannot send the deposit back."),
    ("missed-visit-fee", "Missed visit fee", "25 dollars", "when the technician arrived outside the window", "Frontline staff credit it only when the window log shows the miss."),
    ("directory-listing", "Directory listing", "3 dollars", "when the listing was added in error", "Frontline staff cannot drop it on the call."),
    ("call-forward-addon", "Call forward add-on", "4 dollars", "prorated to the day it was removed", "Frontline staff can remove it and cannot prorate it."),
]


def fee_docs(codec: HuggingFaceCodec) -> None:
    for slug, title, price, finance_rule, agent_rule in FEES:
        public = (
            f"# {title}\n\n"
            f"The {title.lower()} is {price}. "
            "This page states the charge a household sees. It does not state who may waive it.\n\n"
            + section(
                codec,
                "What the household is billed",
                f"The billed figure is {price}. A neighboring fee page uses a different figure and a different name. "
                "Do not subtract this figure from a plan price. It is its own line.",
                f"{title} at {price}",
            )
            + section(
                codec,
                "What this page does not decide",
                f"{agent_rule} Waiver rules are on the finance page for {title.lower()}. "
                "A household who wants a waiver opens a billing ticket.",
                f"{title} at {price}",
            )
        )
        finance = (
            f"# Waiving the {title.lower()}\n\n"
            f"The ledger name for this line is {title.lower()}, billed at {price}. "
            f"Finance may waive or cap it under this rule: {finance_rule}. "
            f"{agent_rule}\n\n"
            + section(
                codec,
                "Who may change the line",
                f"The finance rule is {finance_rule}. "
                f"That rule is not the rule on the household page, which only states {price}. "
                "A waiver note names the invoice, the line, and the rule that was used.",
                f"waive {title} ({finance_rule})",
            )
            + section(
                codec,
                "What not to borrow",
                "Goodwill caps, outage credits, and refund caps are different pages. "
                f"A {title.lower()} waiver is not a goodwill credit and is not an outage credit. "
                f"Keep the rule {finance_rule} on this line only.",
                f"waive {title} ({finance_rule})",
            )
        )
        write_new("public", f"fee-{slug}", title, public)
        write_new("finance", f"waive-{slug}", f"Waiving the {title.lower()}", finance)


ACCOUNTS = [
    ("authorized-user", "Authorized user", "a person the account holder named in writing", "the account PIN plus the written name"),
    ("bankruptcy-notice", "Bankruptcy notice", "a written notice from a court", "stop outbound collection the day the notice is logged"),
    ("seasonal-suspend", "Seasonal suspend", "a suspend of wireline service for a vacant month", "at most 2 months, and the number is kept"),
    ("language-preference", "Language preference", "the language marked on the account", "do not switch it because a caller asks in another language"),
    ("number-change", "Changing the telephone number", "a new number on the same account", "the old number is released after 1 business day"),
    ("toll-block", "Toll block", "a block on paid long-distance calls", "local calls stay open"),
    ("international-dial-block", "International dial block", "a block on calls outside the country", "roaming data is a different page"),
    ("caller-id-block", "Caller id block", "hiding the number on outbound calls", "it does not hide the number from emergency services"),
]


def account_docs(codec: HuggingFaceCodec) -> None:
    for slug, title, definition, rule in ACCOUNTS:
        body = (
            f"# {title}\n\n"
            f"{title} means {definition}. The rule on this page is: {rule}. "
            "A neighboring account page uses a different rule.\n\n"
            + section(
                codec,
                "The rule",
                f"Apply only this rule: {rule}. "
                f"The definition is {definition}. "
                "Write the ticket note with the page name so a later reader does not apply a neighboring rule.",
                f"{title}: {rule}",
            )
            + section(
                codec,
                "What not to borrow",
                "Authorized user, bankruptcy notice, seasonal suspend, language preference, number change, "
                "toll block, international dial block, and caller id block are separate pages. "
                f"This page is only {title.lower()}.",
                f"{title}: {rule}",
            )
        )
        write_new("agent", f"account-{slug}", title, body)


LEDGER = [
    ("duplicate-payment", "Duplicate payment", "the same amount posted twice on one invoice", "reverse the second posting the day it is found"),
    ("deposit-application", "Applying a deposit to a past-due invoice", "using a deposit already on the account against an overdue invoice", "apply it only after day 30 past due"),
    ("tiny-balance", "Balance under 5 dollars", "a residual under 5 dollars after a payment", "finance may clear it without the goodwill cap"),
    ("third-party-payment", "Payment from a third party", "a payment that arrives from someone who is not the account holder", "hold it until the account holder confirms it in the app"),
    ("broken-arrangement", "Broken payment arrangement", "a written arrangement whose payment did not post", "the arrangement ends after one missed payment"),
    ("returned-payment", "Returned payment", "a payment the bank sent back", "the returned-payment line is 15 dollars and is not the late fee"),
    ("credit-balance", "Credit balance left after closing", "money left on an account that has closed", "send it only after 30 days with no further charge"),
    ("misapplied-payment", "Payment posted to the wrong account", "a payment that landed on a different account number", "move it the same day and do not treat it as a refund"),
]


def ledger_docs(codec: HuggingFaceCodec) -> None:
    for slug, title, definition, rule in LEDGER:
        body = (
            f"# {title}\n\n"
            f"{title} means {definition}. The ledger rule is: {rule}. "
            "This is not the refund matrix, the goodwill cap, or the write-off page.\n\n"
            + section(
                codec,
                "The ledger rule",
                f"Follow this rule only: {rule}. "
                f"The definition is {definition}. "
                "Name the invoice and the payment id on the note.",
                f"{title}: {rule}",
            )
            + section(
                codec,
                "What not to borrow",
                "Duplicate payment, deposit application, a balance under 5 dollars, a third-party payment, "
                "a broken arrangement, a returned payment, a leftover credit balance, and a misapplied payment "
                f"are separate pages. This page is only {title.lower()}.",
                f"{title}: {rule}",
            )
        )
        write_new("finance", f"ledger-{slug}", title, body)


def check(codec: HuggingFaceCodec) -> None:
    settings = get_settings()
    documents = load_corpus(CORPUS)
    prepared = prepare_documents(
        documents,
        codec=codec,
        budget=settings.chunk_token_budget,
        overlap=settings.chunk_overlap_tokens,
    )
    chunks = [chunk for document in prepared for chunk in document.chunks]
    blob = "\n".join(chunk.text for chunk in chunks).lower()
    for term in FORBIDDEN:
        if term in blob:
            raise SystemExit(f"forbidden term present: {term}")
    counts = dict.fromkeys(CANARIES, 0)
    for path in CORPUS.rglob("*"):
        if not path.is_file() or path.suffix == ".json":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for canary in counts:
            if canary in text:
                counts[canary] += 1
    if any(value != 1 for value in counts.values()):
        raise SystemExit(f"canary file counts: {counts}")
    gold = json.loads((ROOT / "data" / "gold" / "questions.json").read_text(encoding="utf-8"))
    by_id = {chunk.id: chunk.text for chunk in chunks}
    problems: list[str] = []
    for question in gold["questions"]:
        for support in question["supports"]:
            fact = support["fact"]
            hits = [chunk_id for chunk_id, text in by_id.items() if fact in text]
            if hits != [support["chunk_id"]]:
                problems.append(f"{question['id']} {fact!r} -> {hits} expected {support['chunk_id']}")
    if problems:
        raise SystemExit("gold facts:\n" + "\n".join(problems))
    if len(chunks) < 250:
        raise SystemExit(f"only {len(chunks)} chunks")
    print(
        json.dumps(
            {"documents": len(documents), "chunks": len(chunks), "canaries": counts},
            indent=2,
        )
    )


def main() -> None:
    settings = get_settings()
    codec = HuggingFaceCodec(settings.embedding_model, settings.embedding_revision)
    for spec in FIBER:
        tail = fiber_tail(codec, spec)
        if spec["existing"]:
            append_tail(f"public/{spec['slug']}.md", "Neighboring tier", tail)
        else:
            write_new("public", spec["slug"], spec["title"], fiber_opening(spec) + "\n\n" + tail)
    for spec in OUTAGE:
        tail = outage_tail(codec, spec)
        if spec["existing"]:
            append_tail(f"{spec['level']}/{spec['slug']}.md", "How the request is opened", tail)
        else:
            opening = (
                f"# {spec['title']}\n\n"
                f"A continuous outage of {spec['hours']} hours or more on a {spec['name']} account "
                f"is handled as {spec['size']}. "
                f"This is not the residential customer rule and it is not the residential agent rule.\n\n"
            )
            write_new(spec["level"], spec["slug"], spec["title"], opening + tail)
    for spec in REFUND:
        append_tail(
            f"{spec['level']}/{spec['slug']}.md",
            "What a refund is not",
            refund_tail(codec, spec),
        )
    refund_new(codec)
    money_confusers(codec)
    neighborhood_docs(codec)
    symptom_docs(codec)
    fee_docs(codec)
    account_docs(codec)
    ledger_docs(codec)
    check(codec)


if __name__ == "__main__":
    main()
