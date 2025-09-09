
import random
from dataclasses import dataclass

from distribution import REGIONS, INDUSTRIES, DEGREES, NARRATIVE_HOOKS

@dataclass
class ProfileFeatures:
    name: str
    years_exp: int
    degree: str
    region: str
    industry: str
    income_known: bool
    income_eur: float  # 0.0 if unknown
    oss_stars: int
    downloads_mo: int
    hindex: int
    products_shipped: int
    # we will also consider other social platforms to have a more robust profiling
    x_impressions: int
    x_engagements: int
    audience_quality: float  # 0.65-1.20
    authenticity: float      # 0.65-1.20
    centrality: float        # 0.0-1.0
    near_superstar: bool
    narrative_hook: str

def rand_name(rng: random.Random) -> str:
    first = ["Alex", "Sam", "Taylor", "Jordan", "Casey", "Avery", "Jamie", "Riley", "Morgan", "Quinn", "Aditya"]
    last = ["Lee", "Kim", "Patel", "Garcia", "Ivanov", "Schmidt", "Nguyen", "Brown", "Khan", "Novak", "Kayle", "Bruno"]
    return f"{rng.choice(first)} {rng.choice(last)}"

def sample_profile(rng: random.Random) -> ProfileFeatures:
    # ref for math functions used:
    # beta distribution: https://en.wikipedia.org/wiki/Beta_distribution
    # gaussian distribution: https://en.wikipedia.org/wiki/Gaussian_distribution
    # lognormvariate: https://en.wikipedia.org/wiki/Log-normal_distribution
    # https://ucilnica.fri.uni-lj.si/pluginfile.php/1147/course/section/1510/Bonacich%20-%20Power%20and%20centrality%20-%20A%20family%20of%20measures%2C%201987.pdf

    name = rand_name(rng)
    years = int(rng.triangular(0, 7, 18))  # skew mid-career
    degree, d_mult = rng.choice(DEGREES)
    region, r_mult = rng.choice(REGIONS)
    industry, i_mult = rng.choice(INDUSTRIES)

    # 65% chance income is explicitly known
    income_known = rng.random() < 0.65
    income = 0.0  # set by heuristic layer if unknown

    # contrib.
    oss_stars = int(max(0, rng.lognormvariate(1.2, 1.0)))      # long tail
    downloads_mo = int(max(0, rng.lognormvariate(3.0, 1.1)))   # long tail
    hindex = int(max(0, rng.gauss(5 if degree != "PhD" else 10, 4)))
    products_shipped = rng.randint(0, 6)

    # X
    x_impressions = int(max(0, rng.lognormvariate(12.0, 1.2)))  # ~e^12 median-ish
    x_engagements = int(x_impressions * rng.uniform(0.005, 0.06))

    # quality
    audience_quality = rng.uniform(0.65, 1.20)
    authenticity = rng.uniform(0.65, 1.20)

    # network centrality ~ Beta; elites in tail
    centrality = min(1.0, max(0.0, rng.betavariate(1.5, 4.0)))
    near_superstar = (rng.random() < (0.06 + 0.12 * centrality))

    narrative_hook = rng.choice(NARRATIVE_HOOKS)

    return ProfileFeatures(
        name=name,
        years_exp=years,
        degree=degree,
        region=region,
        industry=industry,
        income_known=income_known,
        income_eur=float(round(income, 2)),
        oss_stars=oss_stars,
        downloads_mo=downloads_mo,
        hindex=hindex,
        products_shipped=products_shipped,
        x_impressions=x_impressions,
        x_engagements=x_engagements,
        audience_quality=audience_quality,
        authenticity=authenticity,
        centrality=centrality,
        near_superstar=near_superstar,
        narrative_hook=narrative_hook,
    )
