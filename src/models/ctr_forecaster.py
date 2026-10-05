from typing import Dict, Any, List

# Industry benchmark organic CTR by Google SERP rank position
BENCHMARK_CTR_MAP = {
    1: 0.285,
    2: 0.157,
    3: 0.110,
    4: 0.080,
    5: 0.058,
    6: 0.044,
    7: 0.035,
    8: 0.028,
    9: 0.023,
    10: 0.019
}

class CTRForecaster:
    """Estimates expected CTR, organic traffic, and conversion volume by ranking tier."""

    def forecast_traffic(self, ranking_prob: float, monthly_search_volume: int = 2400) -> Dict[str, Any]:
        """Calculates expected position bracket, expected CTR, and forecast monthly visits."""
        if ranking_prob >= 0.75:
            expected_pos = 2
            expected_ctr = BENCHMARK_CTR_MAP[2]
        elif ranking_prob >= 0.60:
            expected_pos = 5
            expected_ctr = BENCHMARK_CTR_MAP[5]
        elif ranking_prob >= 0.45:
            expected_pos = 8
            expected_ctr = BENCHMARK_CTR_MAP[8]
        elif ranking_prob >= 0.30:
            expected_pos = 14
            expected_ctr = 0.008
        else:
            expected_pos = 25
            expected_ctr = 0.002

        expected_monthly_clicks = int(round(monthly_search_volume * expected_ctr * (0.8 + 0.4 * ranking_prob)))

        # Forecast across top 10 positions
        curve_data = []
        for pos, ctr in BENCHMARK_CTR_MAP.items():
            curve_data.append({
                "position": pos,
                "benchmark_ctr_pct": round(ctr * 100, 1),
                "potential_monthly_clicks": int(round(monthly_search_volume * ctr))
            })

        return {
            "monthly_search_volume": monthly_search_volume,
            "expected_ranking_position": expected_pos,
            "expected_ctr_pct": round(expected_ctr * 100, 2),
            "forecast_monthly_clicks": expected_monthly_clicks,
            "serp_ctr_curve": curve_data
        }
