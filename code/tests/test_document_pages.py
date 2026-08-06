from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_PAGE_STYLES = {
    "docs/Model-Aligned-Wiki.html": "assets/document.css",
    "docs/related-work/related-work.html": "../assets/document.css",
    "docs/related-work/survey.html": "../assets/document.css",
    "docs/method/scriptorium-method.html": "../assets/document.css",
    "docs/experiments/experiment.html": "../assets/document.css",
    "docs/research/research-plan.html": "../assets/document.css",
    "docs/research/longmemeval-smoke-plan.html": "../assets/document.css",
}
REQUIRED_SHELL_CLASSES = (
    "document-shell",
    "document-sidebar",
    "document-sidebar-brand",
    "document-main",
    "document-content",
    "document-header",
    "document-breadcrumb",
    "document-eyebrow",
    "document-lead",
    "document-meta",
)
REQUIRED_FIGURE_LABELS = {
    "Agent Operations",
    "Memory Writer",
    "Memory Manager",
    "Query Navigator",
    "File-Native Multi-View Memory",
    "Source Memory",
    "Topical View",
    "Timeline View",
    "Recent Memory",
    "Core Memory",
    "Hyperlink Relations",
    "grep",
    "BM25",
    "Embedding",
    "Selected Evidence",
    "Evidence-Grounded Answer",
}
IGNORED_SCHEMES = {"http", "https", "mailto", "data", "javascript"}


class DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lang = ""
        self.class_count: Counter[str] = Counter()
        self.ids: list[str] = []
        self.fragment_links: list[str] = []
        self.local_targets: list[str] = []
        self.stylesheets: list[str] = []
        self.tag_count: Counter[str] = Counter()
        self.text: list[str] = []
        self.image_hrefs: list[str] = []
        self.table_count = 0
        self.focusable_table_count = 0
        self.metric_tab_count = 0
        self.accessible_metric_tab_count = 0
        self.current_page_link_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = dict(attrs)
        self.tag_count[tag] += 1
        if tag == "table":
            self.table_count += 1
            if attributes.get("tabindex") == "0":
                self.focusable_table_count += 1
        if "metric-tab" in (attributes.get("class") or "").split():
            self.metric_tab_count += 1
            if tag == "button" and attributes.get("role") == "tab":
                self.accessible_metric_tab_count += 1
        if attributes.get("aria-current") == "page":
            self.current_page_link_count += 1
        if tag == "html":
            self.lang = attributes.get("lang") or ""
        for class_name in (attributes.get("class") or "").split():
            self.class_count[class_name] += 1
        if element_id := attributes.get("id"):
            self.ids.append(element_id)
        href = attributes.get("href")
        if href is not None:
            if tag == "link" and "stylesheet" in (attributes.get("rel") or "").lower():
                self.stylesheets.append(href)
            self._add_target(href)
            if tag == "image":
                self.image_hrefs.append(href)
        src = attributes.get("src")
        if src is not None:
            self._add_target(src)
            if tag == "image":
                self.image_hrefs.append(src)

    def handle_data(self, data: str) -> None:
        self.text.append(data)

    def _add_target(self, value: str) -> None:
        target = urlsplit(value)
        if target.scheme.lower() in IGNORED_SCHEMES or value.startswith("//"):
            return
        if target.path:
            self.local_targets.append(value)
        if not target.path and target.fragment:
            self.fragment_links.append(unquote(target.fragment))


def parse_document(relative_path: str) -> tuple[Path, DocumentParser]:
    path = REPOSITORY_ROOT / relative_path
    parser = DocumentParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return path, parser


def assert_local_targets_exist(page_path: Path, document: DocumentParser) -> None:
    missing = []
    for value in document.local_targets:
        target_path = unquote(urlsplit(value).path)
        resolved = (page_path.parent / target_path).resolve()
        if not resolved.exists():
            missing.append(f"{value} -> {resolved}")
    assert not missing, "missing local targets:\n" + "\n".join(missing)


def test_public_pages_share_the_document_shell() -> None:
    for relative_path, stylesheet in PUBLIC_PAGE_STYLES.items():
        _, document = parse_document(relative_path)
        assert document.lang == "zh-CN", relative_path
        assert stylesheet in document.stylesheets, relative_path
        for class_name in REQUIRED_SHELL_CLASSES:
            assert document.class_count[class_name] == 1, f"{relative_path}: {class_name}"
        assert document.current_page_link_count == 1, relative_path
        assert "scriptorium" in " ".join(document.text), relative_path
        assert document.tag_count["h1"] == 1, relative_path


def test_public_page_anchors_ids_and_local_targets_are_valid() -> None:
    for relative_path in PUBLIC_PAGE_STYLES:
        page_path, document = parse_document(relative_path)
        assert len(document.ids) == len(set(document.ids)), relative_path
        assert set(document.fragment_links) <= set(document.ids), relative_path
        assert_local_targets_exist(page_path, document)


def test_tables_and_metric_tabs_are_keyboard_accessible() -> None:
    for relative_path in PUBLIC_PAGE_STYLES:
        _, document = parse_document(relative_path)
        assert document.focusable_table_count == document.table_count, relative_path
    _, experiment = parse_document("docs/experiments/experiment.html")
    assert experiment.metric_tab_count == 9
    assert experiment.accessible_metric_tab_count == experiment.metric_tab_count


def test_experiment_page_retains_tab_and_relocation_behavior() -> None:
    experiment_html = (REPOSITORY_ROOT / "docs/experiments/experiment.html").read_text(encoding="utf-8")
    for token in (
        "function moveSection(",
        "function switchTab(",
        "moveSection('protocol-block','protocol-slot')",
        "moveSection('benchmark-master-results','main-results-primary')",
    ):
        assert token in experiment_html


def test_overview_figure_is_editable_and_complete() -> None:
    figure_path = REPOSITORY_ROOT / "code/figures/scriptorium_overview.svg"
    assert figure_path.exists()
    _, figure = parse_document("code/figures/scriptorium_overview.svg")
    figure_source = figure_path.read_text(encoding="utf-8")
    assert "viewBox=" in figure_source
    assert not figure.image_hrefs
    figure_text = " ".join(figure.text)
    for label in REQUIRED_FIGURE_LABELS:
        assert label in figure_text


def test_public_pages_reference_the_svg_overview_figure() -> None:
    overview_html = (REPOSITORY_ROOT / "docs/Model-Aligned-Wiki.html").read_text(encoding="utf-8")
    method_html = (REPOSITORY_ROOT / "docs/method/scriptorium-method.html").read_text(encoding="utf-8")
    assert "../code/figures/scriptorium_overview.svg" in overview_html
    assert "../assets/scriptorium_overview.svg" in method_html


def test_method_page_records_the_current_topic_and_timeline_contract() -> None:
    method = (REPOSITORY_ROOT / "docs/method/scriptorium-method.html").read_text(
        encoding="utf-8"
    )
    for token in (
        "^new-block-label",
        "[^new-evidence-label]",
        "claim-adjacent evidence footnotes",
        "relations.json",
        ".scriptorium/runtime.json",
        "不生成文件或条目",
        "shell-only Writer/Manager",
    ):
        assert token in method
    assert "Writer 的内部 proposal 可以是结构化 JSON" not in method
    assert "timeline/undated.md" not in method
