

from typing import Tuple, List

# X for shaping distributions
PLATFORM_CPM_X: Tuple[float, float] = (4.0, 9.0)    # EUR per 1k impressions
PLATFORM_CPE_X: Tuple[float, float] = (0.03, 0.15)  # EUR per engagement

# regions with cost-of-labor
REGIONS: List[Tuple[str, float]] = [
    ("SF Bay Area", 1.35),
    ("New York", 1.30),
    ("London", 1.25),
    ("Berlin", 1.05),
    ("Bangalore", 0.55),
    ("Remote (Global)", 0.90),
]

# industry multipliers
INDUSTRIES: List[Tuple[str, float]] = [
    ("BigTech", 1.25),
    ("ScaleUp", 1.10),
    ("Startup", 1.00),
    ("Academia/Research", 0.90),
    ("Gov/NGO", 0.85),
]

# degree multipliers
DEGREES: List[Tuple[str, float]] = [
    ("None", 0.85),
    ("BSc/BA", 1.00),
    ("MSc", 1.08),
    ("PhD", 1.15),
]

NARRATIVE_HOOKS = [
    "ex-BigTech engineer now building AI infra",
    "open-source maintainer of a popular ML toolkit",
    "researcher transitioning to applied AI",
    "creator focused on dev education",
    "startup founder shipping mobile apps",
    "data scientist in fintech",
]
