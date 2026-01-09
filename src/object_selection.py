"""
    Object selection
"""
import awkward as ak
from coffea.lookup_tools import extractor

def veto_map_selection(root_file, histo_name, *args):
    """Apply veto map selection using a ROOT histogram."""
    ext = extractor()
    ext.add_weight_sets([f"veto_map {histo_name} {root_file}"])
    ext.finalize()
    veto_map = ext.make_evaluator()
    return veto_map["veto_map"](*args) == 0

def trailing_selection(leading_mask, subleading_mask, obj_var):
    """Apply leading and subleading masks to object variable."""
    leading_mask = ak.firsts(leading_mask)
    subleading_mask = subleading_mask[:,1:]
    leading_mask_broadcasted = ak.broadcast_arrays( # pylint: disable=unsubscriptable-object
                    obj_var,
                    leading_mask)[1]
    tot_mask = ak.concatenate(
                    [leading_mask_broadcasted[:, :1], subleading_mask],
                    axis=1)
    return tot_mask
