# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..preprocess import FmriRealign4d


def test_FmriRealign4d_inputs():
    input_map = dict(between_loops=dict(usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(mandatory=True,
    ),
    loops=dict(usedefault=True,
    ),
    slice_order=dict(requires=[u'time_interp'],
    ),
    speedup=dict(usedefault=True,
    ),
    start=dict(usedefault=True,
    ),
    time_interp=dict(requires=[u'slice_order'],
    ),
    tr=dict(mandatory=True,
    ),
    tr_slices=dict(requires=[u'time_interp'],
    ),
    )
    inputs = FmriRealign4d.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_FmriRealign4d_outputs():
    output_map = dict(out_file=dict(),
    par_file=dict(),
    )
    outputs = FmriRealign4d.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
