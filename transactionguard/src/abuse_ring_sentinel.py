from __future__ import annotations

import argparse

import networkx as nx
import numpy as np
import pandas as pd

MIN_CLUSTER_SIZE_TO_SCORE = 2      
SIZE_SUSPICION_KNEE = 4            
SIGNUP_WINDOW_SUSPICION_DAYS = 3   
VELOCITY_SUSPICION_TXNS = 6        

WEIGHT_SIZE = 0.30
WEIGHT_SIGNUP_CONCENTRATION = 0.40
WEIGHT_VELOCITY = 0.30

MANUAL_REVIEW_THRESHOLD = 0.5


def build_account_graph(df: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(df["account_id"])
    for _, group in df.groupby("device_id"):
        ids = group["account_id"].tolist()
        for i in range(len(ids) - 1):
            g.add_edge(ids[i], ids[i + 1], reason="shared_device")
    for _, group in df.groupby("linked_bank_account"):
        ids = group["account_id"].tolist()
        for i in range(len(ids) - 1):
            g.add_edge(ids[i], ids[i + 1], reason="shared_bank_account")
    return g


def score_clusters(df: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
    df = df.set_index("account_id")
    components = list(nx.connected_components(graph))

    rows = []
    for i, component in enumerate(components):
        members = sorted(component)
        size = len(members)
        if size < MIN_CLUSTER_SIZE_TO_SCORE:
            rows.append({"graph_cluster_id": f"CLUSTER-{i:04d}", "members": members,
                         "size": size, "suspicion_score": 0.0, "recommended_action": "accept",
                         "reasons": ["Single account — no shared-infrastructure cluster"]})
            continue

        sub = df.loc[members]
        signup_spread_days = float(sub["signup_day"].max() - sub["signup_day"].min())
        avg_velocity = float(sub["txn_count_first_7d"].mean())

        size_score = min(1.0, max(0.0, (size - SIZE_SUSPICION_KNEE) / 8.0 + 0.3))
        # tight signup window -> high concentration score; spread out -> low
        signup_score = min(1.0, max(0.0, (SIGNUP_WINDOW_SUSPICION_DAYS - signup_spread_days)
                                      / SIGNUP_WINDOW_SUSPICION_DAYS))
        velocity_score = min(1.0, max(0.0, (avg_velocity - VELOCITY_SUSPICION_TXNS) / 6.0))

        composite = (WEIGHT_SIZE * size_score + WEIGHT_SIGNUP_CONCENTRATION * signup_score
                     + WEIGHT_VELOCITY * velocity_score)

        reasons = []
        if size_score > 0.3:
            reasons.append(f"{size} accounts share infrastructure — larger than a typical "
                            f"shared-device household")
        if signup_score > 0.3:
            reasons.append(f"All accounts opened within {signup_spread_days:.0f} day(s) of "
                            f"each other — signup burst")
        if velocity_score > 0.3:
            reasons.append(f"Average {avg_velocity:.1f} transactions in first 7 days — "
                            f"elevated cash-out velocity")
        if not reasons:
            reasons.append("Shared infrastructure detected but consistent with ordinary "
                            "household/shop sharing, not a coordinated ring")

        rows.append({
            "graph_cluster_id": f"CLUSTER-{i:04d}",
            "members": members,
            "size": size,
            "signup_spread_days": signup_spread_days,
            "avg_txn_velocity_7d": avg_velocity,
            "suspicion_score": round(float(composite), 4),
            "recommended_action": "manual_review" if composite >= MANUAL_REVIEW_THRESHOLD else "accept",
            "reasons": reasons,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Abuse-ring sentinel — graph-based account cluster scorer (defense-only).")
    parser.add_argument("--input", default="data/abuse_ring_accounts.csv")
    parser.add_argument("--output", default="reports/abuse_ring_scored.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    graph = build_account_graph(df)
    scored = score_clusters(df, graph)

    flagged = scored[scored["recommended_action"] == "manual_review"]
    print(f"Built graph over {len(df):,} accounts -> {graph.number_of_nodes():,} nodes, "
          f"{nx.number_connected_components(graph):,} clusters")
    print(f"Flagged {len(flagged)} clusters for manual review "
          f"({flagged['size'].sum() if len(flagged) else 0} accounts total)")

    scored_out = scored.drop(columns=["members"])  
    scored_out.to_csv(args.output, index=False)
    print(f"Wrote -> {args.output}")


if __name__ == "__main__":
    main()
