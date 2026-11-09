"""The `datasets` toolset: what the store holds and structured as-of reads (no free SQL)."""
from __future__ import annotations

import datetime as dt
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from ..auth import tier_check
from ..config import Settings
from ..paging import MAX_OFFSET, cap_result, clamp_limit, decode_cursor, encode_cursor
from .common import AS_OF_DESCRIPTION, READ_ONLY, parse_as_of, rows_to_utc, store_call

FilterScalar = str | int | float | bool
FilterValue = FilterScalar | list[FilterScalar]
FILTERS_DESCRIPTION = ('{column: value or list of values}, exact match (IN); columns from datasets_describe, '
                       'e.g. {"entity_value": "0000320193", "frequency": ["annual"]}. Filter on '
                       'entity_value for speed: an unfiltered read of a large dataset runs out of time')
MAX_NAME_CHARS = 100
MAX_CURSOR_CHARS = 200


def rows_to_csv(rows: list[dict], columns: list[str] | None) -> str:
    """CSV text, header + one line per row. Arrow writes Decimal exactly (decimal128, no float).
    No rows: the header alone when `columns` names them, else an empty string (nothing to name)."""
    import io

    import pyarrow as pa
    import pyarrow.csv as pcsv

    if rows:
        table = pa.Table.from_pylist(rows)
    elif columns:
        table = pa.table({c: pa.array([], pa.null()) for c in columns})
    else:
        return ""
    buf = io.BytesIO()
    pcsv.write_csv(table, buf)
    return buf.getvalue().decode()


def build_datasets_toolset(store, settings: Settings, *, auth_enabled: bool) -> FastMCP:
    ds = FastMCP("datasets", mask_error_details=True)
    read = tier_check("read", auth_enabled=auth_enabled)

    async def page_rows(dataset, filters, columns, as_of, vintages, limit, cursor):
        """One page: read offset+page+1 rows (sorted by natural key in the store), slice; the extra
        row only says whether a next page exists (no total count: it would scan)."""
        page, offset = clamp_limit(limit, settings), decode_cursor(cursor)
        when = parse_as_of(as_of)
        rows = await store_call(store.read, settings=settings, dataset=dataset, as_of=when,
                                filters=filters or None, columns=columns or None,
                                limit=offset + page + 1, vintages=vintages)
        chunk, more = rows[offset:offset + page], len(rows) > offset + page
        end = offset + len(chunk)
        return when, rows_to_utc(chunk), (encode_cursor(end) if more and end <= MAX_OFFSET else None)

    @ds.tool(name="list", auth=read, annotations=READ_ONLY)
    async def list_datasets() -> dict:
        """Published datasets: name, version, kind, licence, row estimate. Then datasets_describe."""
        datasets = await store_call(store.datasets, settings=settings)
        return cap_result({"datasets": rows_to_utc(datasets)}, settings)

    @ds.tool(auth=read, annotations=READ_ONLY)
    async def describe(
        dataset: Annotated[str, Field(description="Name from datasets_list", max_length=MAX_NAME_CHARS)],
    ) -> dict:
        """Columns (type, allowed values, doc), natural key and licence of one dataset."""
        return cap_result(await store_call(store.describe, settings=settings, dataset=dataset), settings)

    @ds.tool(auth=read, annotations=READ_ONLY)
    async def read_asof(
        dataset: Annotated[str, Field(description="Name from datasets_list", max_length=MAX_NAME_CHARS)],
        filters: Annotated[dict[str, FilterValue] | None, Field(description=FILTERS_DESCRIPTION)] = None,
        columns: Annotated[list[str] | None, Field(description="Columns to return; omitted = all")] = None,
        as_of: Annotated[dt.datetime | None, Field(description=AS_OF_DESCRIPTION)] = None,
        vintages: Annotated[bool, Field(
            description="Every version known at as_of, not only the latest per natural key")] = False,
        limit: Annotated[int | None, Field(
            description="Rows per page; omitted = server default; outside [1, server maximum] is refused")] = None,
        cursor: Annotated[str | None, Field(description="next_cursor of the previous page, unchanged",
                                            max_length=MAX_CURSOR_CHARS)] = None,
    ) -> dict:
        """Rows of a dataset as publicly known at `as_of` (latest version per natural key unless
        `vintages`). Filter on natural-key columns (entity_value...) for speed. Rows are ordered by
        natural key then known_at; page with next_cursor. Datetimes are UTC."""
        when, chunk, nxt = await page_rows(dataset, filters, columns, as_of, vintages, limit, cursor)
        return cap_result({"dataset": dataset, "as_of": when.isoformat(), "returned": len(chunk),
                           "next_cursor": nxt, "rows": chunk}, settings,
                          hint="ask for fewer columns or a smaller limit")

    @ds.tool(auth=read, annotations=READ_ONLY)
    async def export_csv(
        dataset: Annotated[str, Field(description="Name from datasets_list", max_length=MAX_NAME_CHARS)],
        filters: Annotated[dict[str, FilterValue] | None, Field(description=FILTERS_DESCRIPTION)] = None,
        columns: Annotated[list[str] | None, Field(description="Columns to return; omitted = all")] = None,
        as_of: Annotated[dt.datetime | None, Field(description=AS_OF_DESCRIPTION)] = None,
        vintages: Annotated[bool, Field(
            description="Every version known at as_of, not only the latest per natural key")] = False,
        limit: Annotated[int | None, Field(
            description="Rows per page; omitted = server default; outside [1, server maximum] is refused")] = None,
        cursor: Annotated[str | None, Field(description="next_cursor of the previous page, unchanged",
                                            max_length=MAX_CURSOR_CHARS)] = None,
    ) -> dict:
        """Same rows as datasets_read_asof, as CSV text (header + one line per row, datetimes UTC) to
        save or paste. Refused, never cut, when larger than the server's result cap: page with
        next_cursor. No rows: header only if `columns` is given, else an empty string."""
        when, chunk, nxt = await page_rows(dataset, filters, columns, as_of, vintages, limit, cursor)
        return cap_result({"dataset": dataset, "as_of": when.isoformat(), "returned": len(chunk),
                           "next_cursor": nxt, "csv": rows_to_csv(chunk, columns)}, settings,
                          hint="ask for fewer columns or a smaller limit")

    return ds
