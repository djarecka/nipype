# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..utils import MRIFill


def test_MRIFill_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='%s',
    mandatory=True,
    position=-2,
    ),
    log_file=dict(argstr='-a %s',
    ),
    out_file=dict(argstr='%s',
    mandatory=True,
    position=-1,
    ),
    segmentation=dict(argstr='-segmentation %s',
    ),
    subjects_dir=dict(),
    terminal_output=dict(nohash=True,
    ),
    transform=dict(argstr='-xform %s',
    ),
    )
    inputs = MRIFill.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_MRIFill_outputs():
    output_map = dict(log_file=dict(),
    out_file=dict(),
    )
    outputs = MRIFill.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
