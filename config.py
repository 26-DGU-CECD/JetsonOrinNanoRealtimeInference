from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EFFICIENTAT_DIR = BASE_DIR / "EfficientAT"

SAMPLE_RATE = 16000
MODEL_SAMPLE_RATE = 32000
CHUNK_SECONDS = 2
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS
MODEL_INPUT_SECONDS = 10
MODEL_INPUT_SAMPLES = MODEL_SAMPLE_RATE * MODEL_INPUT_SECONDS

REQUIRED_INPUT_CHANNELS = 6
MIC_CHANNEL_INDEX = 0
MIC_NAME_KEYWORDS = ("respeaker", "re speaker", "seeed", "array v3")

MODEL_NAME = "mn10_as"
AUDIOSET_CLASS_COUNT = 527

DB_EPSILON = 1e-12
DEFAULT_MIN_DB = 30.0
DEFAULT_ENHANCE_THRESHOLD_DB = 35.0
DEFAULT_NOISE_REDUCTION_DB = 18.0
DEFAULT_MAIN_GAIN_DB = 8.0
DEFAULT_ENHANCE_SHARPNESS = 2.0
DEFAULT_MIN_SCORE = 0.05

ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_RESET = "\033[0m"

# EfficientAT AudioSet pretrained models use the official 32 kHz frontend.
N_MELS = 128
WINDOW_SIZE = 800
HOP_SIZE = 320
N_FFT = 1024
FMAX = MODEL_SAMPLE_RATE // 2 - 1000

LABEL_MAPPING = {
    "construction": ["Tools", "Power tool", "Jackhammer", "Drill", "Chainsaw", "Hammer", "Sawing"],
    "gunshot": ["Gunshot, gunfire"],
    "alarm_siren": ["Siren", "Alarm", "Alarm clock"],
    "horn": ["Vehicle horn, car horn, honking"],
    "water": [
        "Water",
        "Rain",
        "Raindrop",
        "Rain on surface",
        "Stream",
        "Waterfall",
        "Gurgling",
        "Water tap, faucet",
        "Sink (filling or washing)",
        "Liquid",
        "Splash, splatter",
        "Pour",
    ],
    "knock": ["Knock"],
    "appliances": ["Vacuum cleaner"],
    "baby_cry": ["Baby cry, infant cry"],
    "animal_cry": ["Dog", "Cat", "Caterwaul"],
    "glass_shatter": ["Glass", "Shatter"],
}

DANGER_LABELS = {
    "gunshot",
    "alarm_siren",
    "horn",
    "glass_shatter",
}

CAUTION_LABELS = {
    "construction",
    "water",
    "knock",
    "appliances",
    "baby_cry",
    "animal_cry",
}

BLUEZ_SERVICE_NAME = "org.bluez"
DBUS_OM_IFACE = "org.freedesktop.DBus.ObjectManager"
DBUS_PROP_IFACE = "org.freedesktop.DBus.Properties"
ADAPTER_IFACE = "org.bluez.Adapter1"
LE_ADVERTISING_MANAGER_IFACE = "org.bluez.LEAdvertisingManager1"
LE_ADVERTISEMENT_IFACE = "org.bluez.LEAdvertisement1"
GATT_MANAGER_IFACE = "org.bluez.GattManager1"
GATT_SERVICE_IFACE = "org.bluez.GattService1"
GATT_CHRC_IFACE = "org.bluez.GattCharacteristic1"

APP_PATH = "/com/bugless/bleinference"
ADVERTISEMENT_PATH = "/com/bugless/bleinference/advertisement0"

# UUIDs expected by the EdgeAudioRecognition Flutter client.
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
INFERENCE_CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"

RESPEAKER_USB_VENDOR_ID = 0x2886
RESPEAKER_USB_PRODUCT_ID = 0x0018
SPEED_OF_SOUND_M_S = 343.0
RAW_DOA_CHANNELS = (1, 2, 3, 4)
RAW_DOA_MIC_POSITIONS_M = (
    (-0.032, 0.000),
    (0.000, -0.032),
    (0.032, 0.000),
    (0.000, 0.032),
)

CARDINAL_SUFFIX = {
    "북": "북쪽",
    "동": "동쪽",
    "남": "남쪽",
    "서": "서쪽",
}
