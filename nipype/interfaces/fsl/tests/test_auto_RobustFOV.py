# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ..utils import RobustFOV


def test_RobustFOV_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='-i %s',
    mandatory=True,
    position=0,
    ),
    out_roi=dict(argstr='-r %s',
    hash_files=False,
    name_source=[u'in_file'],
    name_template='%s_ROI',
    ),
    output_type=dict(),
    terminal_output=dict(nohash=True,
    ),
    )
    inputs = RobustFOV.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value


def test_RobustFOV_outputs():
    output_map = dict(out_roi=dict(),
    )
    outputs = RobustFOV.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value
