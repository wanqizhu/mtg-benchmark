"""Freeze compact, attributed prerelease ratings from already downloaded sources.

The collector's current October joint LLU list and LR's postlaunch rare episode
are deliberately excluded. No article prose or transcript is reproduced.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--community", type=Path, required=True)
    parser.add_argument("--scryfall", type=Path, help="Optional frozen raw main-set print metadata for loyalty")
    args = parser.parse_args()
    unique = {}
    for card in json.loads(args.cards.read_text()):
        unique.setdefault(card["name"], card)
    assert len(unique) == 295
    cards = list(unique.values())
    if args.scryfall:
        printed = {c['name']: c for c in json.loads(args.scryfall.read_text())}
        for card in cards:
            card['loyalty'] = printed.get(card['name'], {}).get('loyalty')
    (ROOT / "inputs/cards.json").write_text(json.dumps(cards, ensure_ascii=False, indent=2))
    feed = json.loads(args.community.read_text())
    dates = {
        "andrewquinn": "2026-09-21", "cardgamebase": "2026-09-23",
        "casualspike": "2026-09-23", "draftsim": "2026-09-25",
        "drosan": "2026-09-25", "mtgazone": "2026-09-21",
        "nizzahon": "2026-09-20", "limitedresources": "2026-09-21",
    }
    sources = {}
    for key, date in dates.items():
        source = feed["sources"][key]
        sources[key] = {
            "id": key, "name": source["name"], "url": source["url"],
            "publication_or_recorded_date": date,
            "date_basis": "collector's source-activity ledger, checked against dated original reviews",
            "scale": source["scale"], "kind": "creator", "rows": [],
            "edition": "prerelease published grades; current collector snapshot",
            "caveat": "provisional live source; no immutable publisher archive" if key == "drosan" else "",
        }
    lr_evidence = {
        row["cardId"]: row for video in feed["videoRatings"]
        if video["sourceKey"] == "limitedresources"
        for row in video["ratings"] if row["videoId"] == "mLABrUxyTKw"
    }
    for card in feed["cards"]:
        for key in dates:
            rating = card["ratings"].get(key)
            if rating is None:
                continue
            if key == "limitedresources" and card["id"] not in lr_evidence:
                continue
            row = {
                "name": card["name"], "native": rating["native"],
                "score": rating.get("normalized") if rating["scope"] in ["ordinary", "ordinary_plus_conditional"] else None,
                "scope": rating["scope"], "ordinary_native": rating.get("ordinaryNative"),
                "contexts": rating.get("contexts", []),
                "source_url": rating.get("sourceUrl", sources[key]["url"]),
            }
            if key == "limitedresources":
                evidence = lr_evidence[card["id"]]
                row["source_url"] = "https://www.youtube.com/watch?v=mLABrUxyTKw&t=" + str(evidence["timestampSeconds"])
                row["evidence"] = evidence["certainty"] + "; caption extraction, not independently heard audio"
            sources[key]["rows"].append(row)
    history = feed["historicalReviews"]
    by_id = {card["id"]: card for card in feed["cards"]}
    for key in history["sourceOrder"]:
        source = history["sources"][key]
        assert source["editionDate"] < "2026-09-29"
        item = {
            "id": key, "name": source["name"], "url": source["url"],
            "publication_or_recorded_date": source["editionDate"],
            "date_basis": "preserved prerelease edition in collector's historicalReviews",
            "scale": source["scale"], "kind": "creator", "rows": [],
            "edition": "preserved September 25 edition; October 4 joint edition excluded", "caveat": "",
        }
        for card_id, ratings in history["cards"].items():
            rating = ratings.get(key)
            if rating is not None:
                item["rows"].append({
                    "name": by_id[card_id]["name"], "native": rating["native"],
                    "score": rating.get("normalized") if rating["scope"] in ["ordinary", "ordinary_plus_conditional"] else None,
                    "scope": rating["scope"], "ordinary_native": rating.get("ordinaryNative"),
                    "contexts": rating.get("contexts", []),
                    "source_url": rating.get("sourceUrl", item["url"]),
                })
        sources[key] = item
    data = {
        "collector_url": "https://chunk.science/mtga-reality-fracture.html",
        "collector_data_url": "https://chunk.science/mtga-reality-fracture/data/fra.e5c2a826e60d.js?v=20261005-llu-joint-edition",
        "collector_generated_at": feed["generatedAt"],
        "local_snapshot_sha256": hashlib.sha256(args.community.read_bytes()).hexdigest(),
        "cutoff": "before 2026-09-29 Arena launch; dated prerelease editions only",
        "excluded": ["October 4 joint Limited Level-Ups list", "September 30 Limited Resources 873 rares/mythics", "Jim Davis commentary without explicit grades"],
        "grade_index_policy": "Retain the collector's monotone 0–100 ordinary grade index, including an explicit ordinary grade accompanied by a conditional alternative or build-around flag. This index is never a win rate. Sideboard-only and conditional-only grades stay missing; native grades and contextual alternatives are preserved.",
        "provenance_caveat": "Creator grades transcribed by a community collector; original source links are retained. Historical-review edition dates are verified collector metadata, not cryptographic proof of publication. No postlaunch editions are intentionally admitted.",
        "sources": list(sources.values()),
    }
    (ROOT / "inputs/creator_ratings.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print([(key, len(source["rows"]), sum(r["score"] is not None for r in source["rows"])) for key, source in sources.items()])


if __name__ == "__main__":
    main()
