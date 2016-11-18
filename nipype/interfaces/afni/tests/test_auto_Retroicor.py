# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from ....testing import assert_equal
from ..preprocess import Retroicor


def test_Retroicor_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    card=dict(argstr='-card %s',
    position=-2,
    ),
    cardphase=dict(argstr='-cardphase %s',
    hash_files=False,
    position=-6,
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    in_file=dict(argstr='%s',
    copyfile=False,
    mandatory=True,
    position=-1,
    ),
    order=dict(argstr='-order %s',
    position=-5,
    ),
    out_file=dict(argstr='-prefix %s',
    name_source=[u'in_file'],
    name_template='%s_retroicor',
    position=1,
    ),
    outputtype=dict(),
    resp=dict(argstr='-resp %s',
    position=-3,
    ),
    respphase=dict(argstr='-respphase %s',
    hash_files=False,
    position=-7,
    ),
    terminal_output=dict(nohash=True,
    ),
    threshold=dict(argstr='-threshold %d',
    position=-4,
    ),
    )
    inputs = Retroicor.input_spec()

    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(inputs.traits()[key], metakey), value


def test_Retroicor_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = Retroicor.output_spec()

    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
