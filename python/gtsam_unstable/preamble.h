/* Please refer to:
 * https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
 *
 * `PYBIND11_MAKE_OPAQUE` will mark the type as "opaque" for the pybind11
 * automatic STL binding, such that the raw objects can be accessed in Python.
 */

// BetweenFactorEM and the TransformBtwRobotsUnaryFactor variants declare
// whitenedError with a gtsam::OptionalMatrixVecType H = nullptr default.
PYBIND11_MAKE_OPAQUE(std::vector<gtsam::Matrix>); // JacobianVector
