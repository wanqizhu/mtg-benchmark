import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "dataset"))

from build_dataset import parse_post_links, persist_source_url, public_page_url, season_from_title


def test_parse_legacy_patreon_posts_url():
    html = '<a href="https://www.patreon.com/posts/159788995">View Solution</a>'
    _, solution_url = parse_post_links(html)
    assert solution_url == "https://www.patreon.com/posts/159788995"


def test_parse_creator_scoped_patreon_posts_url():
    html = (
        '<a class="thumbnail" href="https://www.patreon.com/mtgpuzzles/posts/162351525" '
        'target="_blank">solution</a>'
    )
    _, solution_url = parse_post_links(html)
    assert solution_url == "https://www.patreon.com/posts/162351525"


def test_season_from_title_prefers_set_name_over_stale_wp_tags():
    assert season_from_title("302: Marvel Super Heroes #1") == "Marvel Super Heroes"
    assert season_from_title("306: The Hobbit #1") == "The Hobbit"
    assert season_from_title("008. Amonkhet Teaser") is None


def test_public_page_url_uses_wordpress_link():
    assert public_page_url({"link": "https://www.possibilitystorm.com/aer1/"}) == (
        "https://www.possibilitystorm.com/aer1/"
    )
    assert public_page_url({"link": "  "}) is None


def test_persist_source_url_writes_missing_link(tmp_path):
    meta_path = tmp_path / "metadata.json"
    meta = {"puzzle_id": "001"}
    persist_source_url(meta_path, meta, {"link": "https://www.possibilitystorm.com/aer1/"})
    assert meta["source_url"] == "https://www.possibilitystorm.com/aer1/"
    written = json.loads(meta_path.read_text(encoding="utf-8"))
    assert written["source_url"] == "https://www.possibilitystorm.com/aer1/"
