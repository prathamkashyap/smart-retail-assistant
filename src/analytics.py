"""
analytics.py -- Post-run summary analytics from the event log.

Reads the CSV log produced during a session and generates a short
statistical summary: total interactions, removals, dwell-time stats,
empty-zone durations, and the most active zone.

Can be run standalone or called at the end of main.py.
"""

import os
import sys
import pandas as pd


def generate_summary(log_path="outputs/logs.csv",
                     output_path="outputs/analytics_summary.csv"):
    """
    Read the event log and print a tabular summary to the console.
    Also saves the summary to a separate CSV for easy reference.
    """
    if not os.path.exists(log_path):
        print("[analytics] No log file found. Nothing to summarize.")
        return

    df = pd.read_csv(log_path)
    if df.empty:
        print("[analytics] Log file is empty. Nothing to summarize.")
        return

    print("\n" + "=" * 60)
    print("  RETAIL SHELF INTELLIGENCE -- SESSION ANALYTICS")
    print("=" * 60)

    total_events = len(df)
    print(f"\nTotal events logged: {total_events}")

    # Break down by event type
    print("\n-- Event breakdown --")
    type_counts = df["event_type"].value_counts()
    for evt, cnt in type_counts.items():
        print(f"  {evt:30s} : {cnt}")

    # Interaction counts per zone
    interactions = df[df["event_type"] == "INTERACTION_START"]
    if not interactions.empty:
        print("\n-- Interactions per zone --")
        zone_interactions = interactions["zone_id"].value_counts()
        for zid, cnt in zone_interactions.items():
            print(f"  {zid:30s} : {cnt}")

        most_active = zone_interactions.idxmax()
        print(f"\n  Most active zone: {most_active} "
              f"({zone_interactions.max()} interactions)")
    else:
        most_active = "N/A"

    # Product removals per zone
    removals = df[df["event_type"] == "PRODUCT_REMOVED"]
    if not removals.empty:
        print("\n-- Product removals per zone --")
        zone_removals = removals["zone_id"].value_counts()
        for zid, cnt in zone_removals.items():
            print(f"  {zid:30s} : {cnt}")
    else:
        print("\n  No product removals recorded.")

    # Dwell time estimates from DWELL_ALERT events
    dwell_events = df[df["event_type"] == "DWELL_ALERT"]
    if not dwell_events.empty:
        print("\n-- Dwell alerts per zone --")
        dwell_zones = dwell_events["zone_id"].value_counts()
        for zid, cnt in dwell_zones.items():
            print(f"  {zid:30s} : {cnt} alerts")

    # Anomaly summary
    anomalies = df[df["event_type"].str.startswith("ANOMALY")]
    if not anomalies.empty:
        print("\n-- Anomalies --")
        anom_counts = anomalies["event_type"].value_counts()
        for evt, cnt in anom_counts.items():
            print(f"  {evt:30s} : {cnt}")
    else:
        print("\n  No anomalies recorded.")

    # Repeated attention
    repeats = df[df["event_type"] == "REPEATED_ATTENTION"]
    if not repeats.empty:
        print(f"\n  Repeated attention events: {len(repeats)}")

    print("\n" + "=" * 60)

    # Save summary to CSV
    summary_rows = []
    summary_rows.append({
        "metric": "total_events", "value": total_events
    })
    summary_rows.append({
        "metric": "total_interactions",
        "value": len(interactions)
    })
    summary_rows.append({
        "metric": "total_removals",
        "value": len(removals)
    })
    summary_rows.append({
        "metric": "total_anomalies",
        "value": len(anomalies)
    })
    summary_rows.append({
        "metric": "most_active_zone",
        "value": most_active
    })
    summary_rows.append({
        "metric": "dwell_alerts",
        "value": len(dwell_events)
    })
    summary_rows.append({
        "metric": "repeated_attention_events",
        "value": len(repeats)
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False)
    print(f"\nSummary saved to: {output_path}\n")


if __name__ == "__main__":
    # Allow running standalone: python src/analytics.py [path_to_logs.csv]
    log_file = sys.argv[1] if len(sys.argv) > 1 else "outputs/logs.csv"
    generate_summary(log_path=log_file)
