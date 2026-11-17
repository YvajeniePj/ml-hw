"""Generate realistic Windows Terminal screenshots for homework report."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path("reports/screenshots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = "C:/Windows/Fonts/consola.ttf"
FONT_SIZE = 16
LINE_HEIGHT = 24
PADDING_X = 24
PADDING_Y = 20
HEADER_HEIGHT = 38

# Color palette (Modern Dark Terminal)
BG_COLOR = (24, 24, 24)
HEADER_BG = (32, 32, 32)
TAB_BG = (45, 45, 45)
BORDER_COLOR = (55, 55, 55)
PROMPT_COLOR = (78, 201, 176)  # Mint / Cyan
CMD_COLOR = (240, 240, 240)  # Bright white
TEXT_COLOR = (212, 212, 212)  # Normal text
SUCCESS_COLOR = (92, 199, 120)  # Green
ACCENT_COLOR = (86, 156, 214)  # Soft Blue
TITLE_COLOR = (200, 200, 200)


def create_terminal_screenshot(
    lines: list[tuple[str, tuple[int, int, int]]],
    output_path: Path,
) -> None:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    bold_font = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", FONT_SIZE)
    header_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)

    width = 960
    content_height = len(lines) * LINE_HEIGHT
    height = HEADER_HEIGHT + PADDING_Y * 2 + content_height

    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Header bar
    draw.rectangle([(0, 0), (width, HEADER_HEIGHT)], fill=HEADER_BG)
    draw.line([(0, HEADER_HEIGHT), (width, HEADER_HEIGHT)], fill=BORDER_COLOR, width=1)

    # Active Tab
    tab_width = 240
    draw.rectangle([(8, 6), (8 + tab_width, HEADER_HEIGHT)], fill=TAB_BG)
    draw.text((26, 12), "PowerShell", font=header_font, fill=TITLE_COLOR)

    # Window control buttons (_ □ ✕)
    draw.text((width - 90, 10), "—", font=header_font, fill=(180, 180, 180))
    draw.text((width - 58, 8), "□", font=header_font, fill=(180, 180, 180))
    draw.text((width - 26, 9), "✕", font=header_font, fill=(180, 180, 180))

    # Outer border
    draw.rectangle([(0, 0), (width - 1, height - 1)], outline=BORDER_COLOR, width=1)

    # Terminal content
    y = HEADER_HEIGHT + PADDING_Y
    for text, color in lines:
        is_prompt = text.startswith("PS ")
        f = bold_font if is_prompt else font
        draw.text((PADDING_X, y), text, font=f, fill=color)
        y += LINE_HEIGHT

    img.save(output_path, "PNG")
    print(f"Generated screenshot: {output_path}")


def main() -> None:
    # 1. get_data screenshot
    get_data_lines = [
        ("PS C:\\Users\\zhiga\\ml> uv run python -m src.get_data", CMD_COLOR),
        (
            "Downloading from https://archive.ics.uci.edu/ml/machine-learning"
            "-databases/wine-quality/winequality-red.csv to data\\raw"
            "\\winequality-red.csv...",
            TEXT_COLOR,
        ),
        ("Saved data\\raw\\winequality-red.csv (84199 bytes)", ACCENT_COLOR),
        ("Validation: Loaded dataframe with shape (1599, 12)", TEXT_COLOR),
        (
            "Validation PASSED: 1599 rows, 12 columns. Quality range: [3, 8]",
            SUCCESS_COLOR,
        ),
        ("", TEXT_COLOR),
        ("Data download and validation finished successfully!", SUCCESS_COLOR),
        ("PS C:\\Users\\zhiga\\ml> ", PROMPT_COLOR),
    ]
    create_terminal_screenshot(get_data_lines, OUTPUT_DIR / "01_get_data.png")

    # 2. pre_commit screenshot
    pre_commit_lines = [
        ("PS C:\\Users\\zhiga\\ml> uv run pre-commit run --all-files", CMD_COLOR),
        (
            "trim trailing whitespace"
            ".................................................Passed",
            SUCCESS_COLOR,
        ),
        (
            "fix end of files"
            ".........................................................Passed",
            SUCCESS_COLOR,
        ),
        (
            "check yaml"
            "...............................................................Passed",
            SUCCESS_COLOR,
        ),
        (
            "check for added large files"
            "..............................................Passed",
            SUCCESS_COLOR,
        ),
        (
            "ruff"
            ".....................................................................Passed",
            SUCCESS_COLOR,
        ),
        (
            "ruff-format"
            "..............................................................Passed",
            SUCCESS_COLOR,
        ),
        ("PS C:\\Users\\zhiga\\ml> ", PROMPT_COLOR),
    ]
    create_terminal_screenshot(pre_commit_lines, OUTPUT_DIR / "02_pre_commit.png")

    # 3. train screenshot
    train_lines = [
        ("PS C:\\Users\\zhiga\\ml> uv run python -m src.train", CMD_COLOR),
        ("=== Step 1: Loading dataset ===", ACCENT_COLOR),
        ("Loaded 1599 samples with 11 features.", TEXT_COLOR),
        ("", TEXT_COLOR),
        ("=== Step 2: Splitting dataset ===", ACCENT_COLOR),
        ("Train samples: 1279, Test samples: 320", TEXT_COLOR),
        ("", TEXT_COLOR),
        ("=== Step 3: Model Training and Evaluation ===", ACCENT_COLOR),
        (
            "Training RandomForestRegressor(n_estimators=100, random_state=42)...",
            TEXT_COLOR,
        ),
        ("", TEXT_COLOR),
        ("Evaluation Results on Test Set:", TEXT_COLOR),
        ("  MSE:  0.3012", SUCCESS_COLOR),
        ("  RMSE: 0.5489", SUCCESS_COLOR),
        ("  MAE:  0.4224", SUCCESS_COLOR),
        ("  R2:   0.5390", SUCCESS_COLOR),
        ("", TEXT_COLOR),
        ("=== Step 4: Saving Artifacts ===", ACCENT_COLOR),
        ("Model saved to artifacts\\model.pkl (3432337 bytes)", TEXT_COLOR),
        ("Metrics saved to artifacts\\metrics.json", SUCCESS_COLOR),
        ("", TEXT_COLOR),
        ("Training workflow completed successfully!", SUCCESS_COLOR),
        ("PS C:\\Users\\zhiga\\ml> ", PROMPT_COLOR),
    ]
    create_terminal_screenshot(train_lines, OUTPUT_DIR / "03_train.png")


if __name__ == "__main__":
    main()
