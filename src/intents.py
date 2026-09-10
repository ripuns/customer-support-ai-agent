"""Intent taxonomy for the AppleSupport agent.

Defined by sampling and manually reviewing customer_msg values from
data/processed/apple_triples.csv (2 random samples of 60, seeds 1 and 7 —
see scripts/_sample_msgs.txt / _sample_msgs2.txt, not committed). The
dataset is from the iOS 11 era and is dominated by update-related bug and
battery complaints; the taxonomy reflects that real distribution rather
than a generic guessed set of support categories.

Scope note: this taxonomy classifies the first customer message in a
thread (customer_msg). Short conversational follow-up turns (e.g. "Yes",
"Thanks!", "Nope") that appear as customer_followup values are not new
intents and are out of scope for classification.
"""

INTENTS = {
    "software_bug": (
        "App, OS, or feature crashing, freezing, glitching, or otherwise "
        "misbehaving (e.g. keyboard glitches, notifications not working, "
        "Bluetooth/Wi-Fi toggling itself, iMessage/FaceTime issues). Usually "
        "tied to a specific iOS/app version or a recent update."
    ),
    "battery_performance": (
        "Battery draining faster than expected, or the device/app running "
        "slow, lagging, or feeling generally degraded after an update. "
        "Kept separate from software_bug because it is an extremely common, "
        "distinct complaint pattern in this dataset."
    ),
    "account_security": (
        "Apple ID login problems, password reset, two-factor/verification "
        "codes not received, iCloud account locked, suspected phishing, or "
        "a compromised/hacked account or payment method."
    ),
    "billing_purchase": (
        "Unauthorized or unexpected charges, subscription access issues "
        "(e.g. paid for Apple Music but can't access it), refund requests, "
        "or questions about pricing/plans (e.g. iCloud storage plans)."
    ),
    "hardware_issue": (
        "A physical device defect: touchscreen not responding, screen "
        "damage, a charger/accessory not working as expected, or similar "
        "hardware-level complaints not explained by a software update."
    ),
    "how_to_info": (
        "A genuine question or feature request with no bug being reported "
        "-- e.g. 'is there a way to...', 'how do I...', asking for setup or "
        "usage advice."
    ),
    "store_order_service": (
        "Issues related to the Apple Store, AppleCare, order/delivery "
        "status, or scheduling an appointment/service, rather than the "
        "device or software itself."
    ),
}

INTENT_LABELS = list(INTENTS.keys())
