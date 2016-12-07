# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..preprocess import Automask


def test_Automask_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    brain_file=dict(argstr='-apply_prefix %s',
    name_source='in_file',
    name_template='%s_masked',
    ),
    clfrac=dict(argstr='-clfrac %s',
    ),
    dilate=dict(argstr='-dilate %s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    erode=dict(argstr='-erode %s',
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='%s',
    copyfile=False,
    mandatory=True,
    position=-1,
    ),
    out_file=dict(argstr='-prefix %s',
    name_source='in_file',
    name_template='%s_mask',
    ),
    outputtype=dict(),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = Automask.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_Automask_outputs():
    output_map = dict(brain_file=dict(),
    out_file=dict(),
    )
    outputs = Automask.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
