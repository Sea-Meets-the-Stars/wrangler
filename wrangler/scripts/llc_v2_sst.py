""" SST-only alias of ``llc_v2_surface`` (kept so ``wr_llc_v2_sst`` keeps working).

Same arguments as ``wr_llc_v2_surface``; the default ``--fields`` is SST.
"""

from wrangler.scripts.llc_v2_surface import parser, main  # noqa: F401


if __name__ == '__main__':
    main(parser())
