from __future__ import annotations

import html
import re
from html.parser import HTMLParser

from .schemas import CleanDocument, RawDocument


class _ReadableHTMLParser(HTMLParser):
    block_tags = {
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }
    skip_tags = {"script", "style", "noscript", "svg"}
    heading_tags = {"h1", "h2", "h3", "h4"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.headings: list[str] = []
        self._skip_depth = 0
        self._current_heading: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.skip_tags:
            self._skip_depth += 1
            return
        if tag in self.heading_tags:
            self._current_heading = []
        if tag in self.block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.skip_tags and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in self.heading_tags and self._current_heading is not None:
            heading = _normalize_space(" ".join(self._current_heading))
            if heading:
                self.headings.append(heading)
                self.parts.append(f"\n## {heading}\n")
            self._current_heading = None
        if tag in self.block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = html.unescape(data)
        if not text.strip():
            return
        if self._current_heading is not None:
            self._current_heading.append(text)
        else:
            self.parts.append(text)
            self.parts.append(" ")


def _normalize_space(text: str) -> str:
    text = text.replace("\u200b", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_title(html_text: str, fallback: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html_text, flags=re.I | re.S)
    if not match:
        return fallback
    title = re.sub(r"<[^>]+>", "", match.group(1))
    title = html.unescape(title)
    title = re.sub(r"\s+", " ", title).strip()
    return title or fallback


def clean_document(raw: RawDocument) -> CleanDocument:
    html_text = _extract_article_html(raw.html)
    parser = _ReadableHTMLParser()
    parser.feed(html_text)
    text = _normalize_space("".join(parser.parts))
    title = extract_title(raw.html, raw.title)
    return CleanDocument(
        id=raw.id,
        title=title,
        url=raw.url,
        text=text,
        headings=parser.headings,
    )


def _extract_article_html(html_text: str) -> str:
    """Keep only the Docusaurus article body when possible."""
    start_match = re.search(
        r'<div class="theme-doc-markdown markdown"', html_text, flags=re.I
    )
    if not start_match:
        article_match = re.search(r"<article[^>]*>(.*?)</article>", html_text, flags=re.I | re.S)
        return article_match.group(1) if article_match else html_text
    start = start_match.start()
    end_match = re.search(
        r'<footer class="theme-doc-footer|</article>',
        html_text[start:],
        flags=re.I,
    )
    end = start + end_match.start() if end_match else len(html_text)
    return html_text[start:end]
