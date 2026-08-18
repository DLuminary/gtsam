/* Please refer to:
 * https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
 * These are required to save one copy operation on Python calls.
 *
 * NOTES
 * =================
 *
 * `PYBIND11_MAKE_OPAQUE` will mark the type as "opaque" for the pybind11
 * automatic STL binding, such that the raw objects can be accessed in Python.
 * Without this they will be automatically converted to a Python object, and all
 * mutations on Python side will not be reflected on C++.
 */

// NoiseModelFactor::unwhitenedError and the WNOA factors take
// gtsam::OptionalMatrixVecType, i.e. std::vector<gtsam::Matrix>*. This must be
// opaque here for the same reason it is in preamble/custom.h: every .i file is
// generated into its own translation unit including only its own preamble, so
// without this the type falls back to the pybind11 stl.h list caster in this
// unit while remaining opaque in custom.cpp. That inconsistency means the
// `H = nullptr` default cannot round-trip through None, so the argument cannot
// be omitted from Python.
PYBIND11_MAKE_OPAQUE(std::vector<gtsam::Matrix>); // JacobianVector
