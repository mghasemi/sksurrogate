#!/usr/bin/env python3
"""Render index.html's visible text as Markdown into llms-full.txt.

Strips markup but keeps heading levels, lists and tables, so an answer engine can
read the whole page without executing JavaScript. Run after any copy change:

    python scripts/make_llms_full.py
"""
import html
import os
import re
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = {"script", "style", "noscript", "svg", "head", "nav", "header", "footer", "kbd"}
BLOCK = {"p", "div", "section", "li", "td", "th", "h1", "h2", "h3", "h4", "h5",
         "pre", "blockquote", "table", "tr", "ul", "ol", "br", "nav", "footer", "header"}


class MD(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out = []
        self.skip = 0
        self.buf = []
        self.in_pre = False
        self.table_row = None
        self.in_table = False
        self.table_rows = 0

    def flush(self):
        text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
        self.buf = []
        return text

    def handle_starttag(self, tag, attrs):
        if tag == "br" and not self.skip:
            self.buf.append(" ")
            return
        if tag in SKIP:
            self.skip += 1
            return
        if self.skip:
            return
        if tag == "pre":
            self.in_pre = True
            self.close_block()
            self.out.append("")
        elif tag in ("h1", "h2", "h3", "h4"):
            self.close_block()
            self.pending_heading = int(tag[1])
        elif tag == "li":
            self.close_block()
            self.out.append("- ")
        elif tag == "tr":
            self.close_block()
            self.table_row = []
        elif tag == "table":
            self.close_block()
            self.in_table = True
            self.out.append("")
        elif tag in ("td", "th"):
            self.flush()

    def handle_endtag(self, tag):
        if tag in SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if tag == "pre":
            self.in_pre = False
            self.out.append("")
        elif tag in ("h1", "h2", "h3", "h4"):
            level = getattr(self, "pending_heading", 2)
            self.out.append("#" * level + " " + self.flush())
            self.out.append("")
        elif tag == "span" and not self.in_pre:
            self.buf.append(" ")
        elif tag in ("p", "blockquote", "div", "section"):
            self.close_block()
        elif tag == "li":
            self.close_block()
        elif tag in ("td", "th") and self.table_row is not None:
            self.table_row.append(self.flush())
        elif tag == "tr" and self.table_row is not None:
            row = "| " + " | ".join(c.replace("|", "\\|") for c in self.table_row) + " |"
            self.out.append(row)
            if self.table_rows == 0:
                self.out.append("|" + " --- |" * len(self.table_row))
            self.table_rows += 1
            self.table_row = None
        elif tag == "table":
            self.in_table = False
            self.table_rows = 0
            self.out.append("")

    def close_block(self):
        text = self.flush()
        if text:
            self.out.append(text)
            self.out.append("")

    def handle_data(self, data):
        if self.skip:
            return
        if self.in_pre:
            self.out.append(data.rstrip("\n"))
        else:
            self.buf.append(data)


def main():
    src = open(os.path.join(ROOT, "index.html")).read()
    body = src[re.search(r"<body[^>]*>", src).end():]
    p = MD()
    p.feed(body)
    p.close_block()
    md = "\n".join(line for line in p.out)
    md = re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"
    out = os.path.join(ROOT, "llms-full.txt")
    open(out, "w").write(md)
    print("wrote llms-full.txt", len(md.splitlines()), "lines,",
          len(md.split()), "words")


if __name__ == "__main__":
    main()
