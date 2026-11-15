import unittest
import numpy as np
from run import edge_selection,classify
from src.grid import make_regular_grid
from src.mapping import make_ideal_mapping,make_moderate_mapping
from src.spv_forward import IdealGaussianUtilityProjector,GaussianPhospheneForwardModel
class Contracts(unittest.TestCase):
    def test_zero_edge_fixed_budget(self):
        p=IdealGaussianUtilityProjector(make_ideal_mapping(make_regular_grid()))
        s,d=edge_selection(np.zeros((96,96),dtype=np.float32),p)
        np.testing.assert_array_equal(s,np.arange(64));self.assertEqual(d['zero_selected'],64)
    def test_target_geometry_rejected(self):
        with self.assertRaises(ValueError):IdealGaussianUtilityProjector(make_moderate_mapping(make_regular_grid(),seed=5301))
    def test_nonfinite_rejected(self):
        p=IdealGaussianUtilityProjector(make_ideal_mapping(make_regular_grid()))
        with self.assertRaises(ValueError):edge_selection(np.full((96,96),np.nan,dtype=np.float32),p)
    def test_budget_energy_under_distortion(self):
        m=make_moderate_mapping(make_regular_grid(),seed=5301)
        im=GaussianPhospheneForwardModel(m).generate(np.arange(64,dtype=np.int64),np.ones(64,dtype=np.float32))
        self.assertAlmostEqual(float(im.sum(dtype=np.float64)),64,places=4)
    def test_invalid_split_rejected(self):
        with self.assertRaises(ValueError):classify({},np.arange(180),np.repeat('only_one_font',180))
if __name__=='__main__':unittest.main()
