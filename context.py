from models import ProfileFeatures

def format_social_block_x(feat: ProfileFeatures) -> str:
    return f"X: {feat.x_impressions:,} impressions / {feat.x_engagements:,} engagements"

def build_user_text(feat: ProfileFeatures) -> str:
    income_str = f"Salary: €{feat.income_eur:,.0f}" if feat.income_known and feat.income_eur > 0 else "Salary: (unknown)"
    near = "directly mentored by a high-influence figure" if feat.near_superstar else "no direct ties to top celebrities"
    social_block = format_social_block_x(feat)
    return (
        f"Bio: {feat.narrative_hook}. Name: {feat.name}. Region: {feat.region}. Industry: {feat.industry}. "
        f"Degree: {feat.degree}. Experience: {feat.years_exp} years. {income_str}. "
        f"Contributions: {feat.products_shipped} products shipped; {feat.oss_stars} GitHub stars; "
        f"{feat.downloads_mo} monthly downloads; h-index {feat.hindex}. "
        f"Social (last 90d): {social_block}. Audience quality: {feat.audience_quality:.2f}; authenticity: {feat.authenticity:.2f}. "
        f"Network: eigenvector centrality {feat.centrality:.2f}; {near}."
    )
