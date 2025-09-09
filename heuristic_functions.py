# scaling, income proxy, EMV(X), and heuristic scoring fallback.

import math
import random
from typing import Dict, Any

from distribution import DEGREES, REGIONS, INDUSTRIES, PLATFORM_CPM_X, PLATFORM_CPE_X
from models import ProfileFeatures

def log_scale_to_1_10(x: float, lo: float, hi: float) -> int:
    x = max(lo, min(hi, x))
    s = 1 + 9 * ((math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo)))
    return int(round(max(1, min(10, s))))

def minmax_to_1_10(x: float, lo: float, hi: float) -> int:
    x = max(lo, min(hi, x))
    s = 1 + 9 * (x - lo) / (hi - lo)
    return int(round(max(1, min(10, s))))

def sample_income_base(rng: random.Random, years: int, degree: str, region: str, industry: str) -> float:
    d_mult = dict(DEGREES)[degree]
    r_mult = dict(REGIONS)[region]
    i_mult = dict(INDUSTRIES)[industry]
    base = 28_000.0 + 4_500.0 * years
    base *= d_mult * r_mult * i_mult
    base *= rng.uniform(0.85, 1.15)
    return max(18_000.0, base)

def emv_x(impressions: int, engagements: int, q1: float, q2: float, rng: random.Random) -> float:
    cpm = rng.uniform(*PLATFORM_CPM_X)
    cpe = rng.uniform(*PLATFORM_CPE_X)
    emv = (impressions / 1000.0) * cpm * q1 + engagements * cpe * q2
    return emv

def heuristic_scores(feat: ProfileFeatures, rng: random.Random) -> Dict[str, Any]:
    # human
    if feat.income_known and feat.income_eur > 0:
        human = log_scale_to_1_10(feat.income_eur, 18_000, 250_000)
    else:
        approx = sample_income_base(rng, feat.years_exp, feat.degree, feat.region, feat.industry)
        human = log_scale_to_1_10(approx, 18_000, 250_000)

    # social
    total_emv = emv_x(feat.x_impressions, feat.x_engagements, feat.audience_quality, feat.authenticity, rng)
    social = log_scale_to_1_10(1.0 + total_emv, 1.0 + 50.0, 1.0 + 2_000_000.0)

    # network
    network = min(10, minmax_to_1_10(feat.centrality, 0.0, 1.0) + (1 if feat.near_superstar else 0))

    # contribution
    comp = 0.5 * math.log1p(feat.oss_stars) + 0.35 * math.log1p(feat.downloads_mo) + 0.25 * (feat.hindex / 20.0) + 0.2 * feat.products_shipped
    contribution = minmax_to_1_10(comp, 0.0, 8.0)

    # confidence (data richness + proximity)
    richness = (
        (1 if feat.income_known else 0) +
        (1 if feat.x_impressions > 0 else 0) +
        (1 if feat.oss_stars > 200 else 0) +
        (1 if feat.downloads_mo > 5000 else 0) +
        (1 if feat.hindex >= 10 else 0)
    )
    confidence = int(max(40, min(92, 50 + richness * 8 + (10 if feat.near_superstar else 0))))

    reasons = []
    reasons.append("Income known; strong human capital" if (feat.income_known and human >= 7) else
                   ("Income known; developing human capital" if feat.income_known else
                    "Human capital inferred from degree/experience"))
    reasons.append("High EMV on X" if social >= 8 else ("Moderate EMV on X" if social >= 5 else "Low EMV on X"))
    reasons.append("Central network ties" if network >= 8 else ("Growing network" if network >= 5 else "Limited network"))
    reasons.append("Solid OSS/citations" if contribution >= 7 else ("Some contributions" if contribution >= 4 else "Limited contributions"))

    return {
        "human": int(human),
        "social": int(social),
        "network": int(network),
        "contribution": int(contribution),
        "confidence": int(confidence),
        "reasons": reasons[:4],
    }
