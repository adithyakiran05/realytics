import pandas as pd

def run_lbo():
    df = pd.read_csv("data/processed/final_output.csv")

    results = []

    for _, row in df.iterrows():

        # --- Base Inputs ---
        base_ebitda = row["EBITDA"] if not pd.isna(row["EBITDA"]) else 1000
        base_growth = row["Predicted_Growth"]

        # --- Assumptions ---
        exit_multiple = 12
        tax_rate = 0.25
        interest_rate = 0.08
        capex_pct = 0.05
        wc_pct = 0.02

        # --- Scenario + Sensitivity ---
        scenarios = {
            "Base": base_growth,
            "Upside": base_growth + 0.03,
            "Downside": max(base_growth - 0.03, -0.05)  # avoid extreme negative
        }

        entry_multiples = [8, 10, 12]

        for entry_multiple in entry_multiples:

            entry_value = base_ebitda * entry_multiple
            debt = entry_value * 0.6
            equity = entry_value * 0.4

            for scenario_name, growth in scenarios.items():

                ebitda = base_ebitda
                remaining_debt = debt

                # --- 5 Year Projection ---
                for year in range(5):

                    ebitda = ebitda * (1 + growth)

                    depreciation = 0.05 * ebitda
                    ebit = ebitda - depreciation

                    interest = remaining_debt * interest_rate

                    ebt = ebit - interest
                    taxes = max(0, ebt * tax_rate)

                    net_income = ebt - taxes

                    capex = capex_pct * ebitda
                    wc_change = wc_pct * ebitda

                    cash_flow = net_income + depreciation - capex - wc_change

                    # Debt repayment
                    repayment = max(0, min(cash_flow, remaining_debt))
                    remaining_debt -= repayment

                # --- Exit ---
                exit_value = ebitda * exit_multiple
                equity_exit = exit_value - remaining_debt

                # Avoid invalid IRR
                if equity <= 0 or equity_exit <= 0:
                    irr = -1
                else:
                    irr = (equity_exit / equity) ** (1/5) - 1

                results.append({
                    "Company": row["Company"],
                    "Scenario": scenario_name,
                    "Entry_Multiple": entry_multiple,
                    "IRR": irr,
                    "Exit_Value": exit_value,
                    "Remaining_Debt": remaining_debt
                })

    lbo_df = pd.DataFrame(results)
    lbo_df.to_csv("data/processed/lbo_output.csv", index=False)

    print("Enhanced LBO with scenarios completed")


if __name__ == "__main__":
    run_lbo()