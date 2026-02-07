import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from model.state import StateVector
from model.gasifier_system import GasifierSystem
from model.constants import PhysicalConstants

class TestCoreModels(unittest.TestCase):
    
    def test_state_vector_serialization(self):
        """Test StateVector to_array and from_array"""
        original_moles = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        state = StateVector(
            gas_moles=original_moles,
            solid_mass=10.0,
            carbon_fraction=0.8,
            T=1500.0,
            P=4e6,
            z=1.5
        )
        
        # Serialize
        arr = state.to_array()
        
        # Deserialize (P and z are passed externally usually, but state has them)
        # from_array signature: (cls, arr_flat, P, z=0.0)
        reconstructed = StateVector.from_array(arr, P=4e6, z=1.5)
        
        np.testing.assert_allclose(state.gas_moles, reconstructed.gas_moles, rtol=1e-7)
        self.assertAlmostEqual(state.solid_mass, reconstructed.solid_mass)
        self.assertAlmostEqual(state.carbon_fraction, reconstructed.carbon_fraction)
        self.assertAlmostEqual(state.T, reconstructed.T)
        self.assertAlmostEqual(state.P, reconstructed.P)
        self.assertAlmostEqual(state.z, reconstructed.z)

    def test_gasifier_input_validation(self):
        """Test GasifierSystem input validation logic"""
        valid_geom = {'L': 10.0, 'D': 2.0}
        valid_coal = {'Cd': 60.0, 'HHV_d': 25.0}
        valid_op = {'coal_flow': 10.0, 'o2_flow': 10.0, 'P': 4e6, 'T_in': 300.0}

        # 1. Valid Check
        try:
            GasifierSystem(valid_geom, valid_coal, valid_op)
        except ValueError as e:
            self.fail(f"Valid inputs raised ValueError: {e}")

        # 2. Missing Geometry
        with self.assertRaisesRegex(ValueError, "Missing geometry parameters"):
            GasifierSystem({'L': 10.0}, valid_coal, valid_op)

        # 3. Negative Dimensions
        with self.assertRaisesRegex(ValueError, "dimensions L and D must be positive"):
            GasifierSystem({'L': -1.0, 'D': 2.0}, valid_coal, valid_op)

        # 4. Missing Coal Property
        bad_coal = {'HHV_d': 25.0} # Missing Cd
        with self.assertRaisesRegex(ValueError, "Coal property 'Cd'"):
            GasifierSystem(valid_geom, bad_coal, valid_op)

    def test_constants_sanity(self):
        """Ensure PhysicalConstants are loaded and reasonable"""
        self.assertGreater(PhysicalConstants.PARTICLE_DENSITY, 0)
        self.assertGreater(PhysicalConstants.GRAVITY, 0)
        self.assertEqual(PhysicalConstants.TOLERANCE_SMALL, 1e-9)

if __name__ == '__main__':
    unittest.main()
