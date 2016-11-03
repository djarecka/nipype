# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ..confounds import FramewiseDisplacement


def test_FramewiseDisplacement_inputs():
    input_map = dict(figdpi=dict(usedefault=True,
    ),
    figsize=dict(usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_plots=dict(mandatory=True,
    ),
    normalize=dict(usedefault=True,
    ),
    out_figure=dict(usedefault=True,
    ),
    out_file=dict(usedefault=True,
    ),
    radius=dict(usedefault=True,
    ),
    save_plot=dict(usedefault=True,
    ),
    series_tr=dict(),
    )
    inputs = FramewiseDisplacement.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_FramewiseDisplacement_outputs():
    output_map = dict(fd_average=dict(),
    out_figure=dict(),
    out_file=dict(),
    )
    outputs = FramewiseDisplacement.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
