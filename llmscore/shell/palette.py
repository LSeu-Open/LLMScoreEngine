"""Action palette scaffolding for the interactive shell."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, List, Sequence

from rapidfuzz import fuzz, process


@dataclass(slots=True)
class PaletteEntry:
    """Represents an entry in the action palette."""

    title: str
    action: str
    description: str = ""
    tags: tuple[str, ...] = ()


class PaletteProvider:
    """Provider backed by an action registry with fuzzy discovery support."""

    def __init__(
        self,
        entries: Sequence[PaletteEntry],
        *,
        recent_limit: int = 20,
    ) -> None:
        self._entries: List[PaletteEntry] = list(entries)
        self._index = {entry.action: entry for entry in self._entries}
        self._usage = Counter[str]()
        self._recent: Deque[str] = deque(maxlen=recent_limit)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def add(self, entry: PaletteEntry) -> None:
        if entry.action in self._index:
            self.remove(entry.action)
        self._entries.append(entry)
        self._index[entry.action] = entry

    def remove(self, action: str) -> None:
        if action not in self._index:
            return
        self._entries = [
            item for item in self._entries if item.action != action
        ]
        self._index.pop(action, None)
        self._usage.pop(action, None)
        try:
            self._recent.remove(action)
        except ValueError:
            pass

    def record_usage(self, action: str) -> None:
        if action not in self._index:
            return
        self._usage[action] += 1
        self._recent.appendleft(action)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def search(self, query: str, *, limit: int = 10) -> Iterator[PaletteEntry]:
        """Return entries ranked by fuzzy match against palette metadata."""

        query = query.strip()
        if not query:
            yield from self.recommend(limit=limit)
            return

        matches = process.extract(  # type: ignore[arg-type]
            query,
            self._entries,
            scorer=fuzz.WRatio,
            processor=self._palette_key,
            limit=limit * 2,
        )
        seen: set[str] = set()
        for entry, score, _ in matches:
            if score < 40:
                continue
            if entry.action in seen:
                continue
            seen.add(entry.action)
            yield entry
            if len(seen) >= limit:
                break

    def recent(self, *, limit: int = 5) -> Iterable[PaletteEntry]:
        for action in list(self._recent)[:limit]:
            entry = self._index.get(action)
            if entry:
                yield entry

    def recommend(self, *, limit: int = 5) -> Iterable[PaletteEntry]:
        if not self._usage:
            yield from self._entries[:limit]
            return
        ranked = self._usage.most_common(limit)
        for action, _ in ranked:
            entry = self._index.get(action)
            if entry:
                yield entry
        if len(ranked) < limit:
            remaining = [
                entry
                for entry in self._entries
                if entry.action not in self._usage
            ]
            for entry in remaining[: limit - len(ranked)]:
                yield entry

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _palette_key(entry: PaletteEntry) -> str:
        parts = [
            entry.title,
            entry.action,
            entry.description,
            " ".join(entry.tags),
        ]
        return " ".join(part for part in parts if part)


__all__ = ["PaletteEntry", "PaletteProvider"]
