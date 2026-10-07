"""Freeze compact, attributed prerelease ratings from already downloaded sources.

The default excludes the October joint LLU list and LR's release-day rare review.
An explicit option adds the rare review to a separate input. No review prose is reproduced.
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
    parser.add_argument("--include-lr-rares", action="store_true", help="Include episode 873 under the user's release-day exception")
    parser.add_argument("--output", type=Path, default=ROOT / 'inputs/creator_ratings.json')
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
        for row in video["ratings"] if row["videoId"] in (['mLABrUxyTKw', 'Pf4DvBpzH7g'] if args.include_lr_rares else ['mLABrUxyTKw'])
    }
    if args.include_lr_rares:
        sources['limitedresources'].update({
            'publication_or_recorded_date': '2026-09-21 CU; 2026-09-29 rare audio / 2026-09-30 rare video',
            'date_basis': 'Official episode pages; collector video metadata; release-day review explicitly included by user',
            'edition': 'Episodes 872 and 873, one joint LSV + Marshall source',
            'rare_episode_url': 'https://lrcast.com/limited-resources-873-reality-fracture-set-review-rare-and-mythic-rare/',
            'caveat': 'Rare review is a release-day edition; it is not certified to precede the exact Arena release time.'
        })
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
                row["source_url"] = "https://www.youtube.com/watch?v=" + evidence['videoId'] + "&t=" + str(evidence["timestampSeconds"])
                row['video_id'] = evidence['videoId']
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
        "cutoff": "Dated prerelease editions, plus user-authorized LR episode 873 release-day review" if args.include_lr_rares else "before 2026-09-29 Arena launch; dated prerelease editions only",
        "excluded": ["October 4 joint Limited Level-Ups list", "Jim Davis commentary without explicit grades"] + ([] if args.include_lr_rares else ["September 30 Limited Resources 873 rares/mythics"]),
        "grade_index_policy": ("Retain the collector's index for provenance only. creator_native.py fits raw numeric or ordered letter grades with eligible explicit context indicators; it does not use this index. Bare SB remains unmapped." if args.include_lr_rares else "Retain the collector's monotone 0–100 ordinary grade index, including an explicit ordinary grade accompanied by a conditional alternative or build-around flag. This index is never a win rate. Sideboard-only and conditional-only grades stay missing; native grades and contextual alternatives are preserved."),
        "provenance_caveat": "Creator grades transcribed by a community collector; original source links are retained. Historical-review edition dates are verified collector metadata, not cryptographic proof of publication. " + ("LR's release-day rare review is explicitly included by the user." if args.include_lr_rares else "No postlaunch editions are intentionally admitted."),
        "sources": list(sources.values()),
    }
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print([(key, len(source["rows"]), sum(r["score"] is not None for r in source["rows"])) for key, source in sources.items()])


if __name__ == "__main__":
    main()
