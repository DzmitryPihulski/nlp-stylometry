import logging
import re

logger = logging.getLogger(__name__)

_ISBN = re.compile(r"^ISBN", re.IGNORECASE)
# Matches Roman numerals I–MMMCMXCIX and plain Arabic numbers
_ROMAN = re.compile(
    r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$",
    re.IGNORECASE,
)
_CONTENT_THRESHOLD = 30   # poetry: min chars to consider a line actual content
_PROSE_THRESHOLD = 50     # prose: min chars to distinguish a paragraph from an epigraph


def _is_chapter_marker(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 10:
        return False
    return s.isdigit() or bool(_ROMAN.match(s))


def clean_text(text: str) -> str:
    lines = text.splitlines()

    # Locate and strip title (first non-blank line)
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        return ""
    logger.debug("Stripping title: %r", lines[i].strip())
    i += 1

    # Detect prose structure: look for a chapter marker preceded by a blank line
    chapter_idx = None
    for j in range(i, len(lines)):
        s = lines[j].strip()
        if s and _is_chapter_marker(s):
            if j == 0 or not lines[j - 1].strip():
                chapter_idx = j
                break

    if chapter_idx is not None:
        # Prose: skip the chapter marker itself, then skip epigraphs / attributions
        # (short lines < _PROSE_THRESHOLD) until the first real paragraph.
        logger.debug("Prose detected; scanning past chapter marker on line %d for epigraphs", chapter_idx)
        content_start = len(lines)
        for j in range(chapter_idx + 1, len(lines)):
            s = lines[j].strip()
            if not s:
                continue
            if len(s) >= _PROSE_THRESHOLD:
                content_start = j
                break
            logger.debug("Stripping epigraph/attribution: %r", s)
        content = lines[content_start:]
    else:
        # Poetry: skip blank lines and metadata until first substantial line
        content_start = len(lines)
        for j in range(i, len(lines)):
            s = lines[j].strip()
            if not s:
                continue
            if _ISBN.match(s):
                logger.debug("Stripping ISBN: %r", s)
                continue
            if len(s) >= _CONTENT_THRESHOLD:
                content_start = j
                break
            logger.debug("Stripping preamble line: %r", s)
        content = lines[content_start:]

    # Remove trailing blank lines
    while content and not content[-1].strip():
        content.pop()

    return "\n".join(content)
