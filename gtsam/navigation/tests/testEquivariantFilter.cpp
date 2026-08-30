/**
 * @file testEquivariantFilter.cpp
 * @brief Simple SO(3) equivariant filter example (attitude-only),
 *        exercising EquivariantFilter with a different M/G/Actions combo.
 *
 * This is inspired by the simple sphere / attitude example in Mahony's
 * equivariant filter tutorial, but here formulated for S^2 directions:
 * - Physical state M is Unit3 (a direction on S^2).
 * - Symmetry group G is Rot3 (attitude).
 * - The state estimate is recovered as \hat{η} = Q^T \bar{η}, matching Mahony's
 * notation.
 *
 * The goal is to ensure EquivariantFilter.h is generic and not tied to ABC.h.
 *
 * The innovation term follows Mahony's equivariant update on S², where the
 * error is defined via the right action φ_{η̄}(Q) = Qᵀη̄ and innovations are
 * formed from ρ_y(Q̂⁻¹).
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/GroupAction.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/MatrixConstants.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Unit3.h>
#include <gtsam/navigation/EquivariantFilter.h>

#include <random>

using namespace gtsam;

namespace attitude_example {

//---------------------------------------------------------------------------
// Types
//---------------------------------------------------------------------------

using M = Unit3;  // physical state: direction η on S^2
using G = Rot3;   // symmetry group: SO(3) attitude Q

//---------------------------------------------------------------------------
// Symmetry: group action on the state
//   φ_η(Q) = Q^T η.
//---------------------------------------------------------------------------

struct Symmetry : public GroupAction<Symmetry, G, M> {
  static constexpr ActionType type = ActionType::Right;

  /// Group action at state eta by group element Q.
  M operator()(const M& eta, const G& Q, OptionalJacobian<2, 2> H_eta = {},
               OptionalJacobian<2, 3> H_Q = {}) const {
    // Apply the right action Q^T * eta as in Mahony's example.
    return Q.unrotate(eta, H_Q, H_eta);
  }
};

//---------------------------------------------------------------------------
// Dynamics
//------------------------------------------- --------------------------------

// Compute the dynamics at a given point on the manifold
Vector2 dynamics(const Vector3& omega, const M& eta) {
  // eta.cross(omega_) is in 3D ambient space
  const Vector3 ambient_dynamics = eta.cross(omega);

  // Convert to 2D tangent space via local basis
  return eta.basis().transpose() * ambient_dynamics;
}

//---------------------------------------------------------------------------
// Lift: maps (state, input) -> group tangent
//
// For this simple example, the lift just returns the body angular velocity
// itself, independent of the state. This is enough to test the EqF plumbing.
//---------------------------------------------------------------------------

struct Lift {
  explicit Lift(const Vector3& omega) : omega_(omega) {}

  typename traits<G>::TangentVector operator()(
      const M& /*R*/, OptionalJacobian<3, 2> H = {}) const {
    if (H) *H = Matrix::Zero(3, 2);
    // Tangent space of SO(3) ~ R^3, so we simply return omega.
    return omega_;
  }

 private:
  Vector3 omega_;
};

//---------------------------------------------------------------------------
// Input action: right action on the input omega.
//---------------------------------------------------------------------------

struct InputAction : public GroupAction<InputAction, G, Vector3> {
  static constexpr ActionType type = ActionType::Right;

  /// Right group action on input: Q^{-1} * omega.
  Vector3 operator()(const Vector3& omega, const G& Q,
                     OptionalJacobian<3, 3> H_omega = {},
                     OptionalJacobian<3, 3> H_Q = {}) const {
    return Q.unrotate(omega, H_Q, H_omega);
  }
};

// Use the generated Orbit type directly for the pure group action.
using InputOrbit = InputAction::Orbit;

/// Embed process noise covariance into the lifted coordinates (identity for
/// this simple example).
inline Matrix3 processNoise(const Matrix3& Sigma) { return Sigma; }

/// Derivative of the lifted dynamics wrt. local coordinates.
inline Matrix2 stateMatrixA(const G& /*Q_hat*/) { return Matrix2::Zero(); }

/// Input matrix B that maps process noise to manifold coordinates.
inline Matrix23 inputMatrixB(const G& /*Q_hat*/) {
  // TODO(Frank): suspect, should depend on eta? Or make sure it works for
  // chosen reference direction.
  return Matrix23::Identity();
}

struct MeasurementFunctor {
  double c_m_;
  explicit MeasurementFunctor(double c_m) : c_m_(c_m) {}

  /// Measurement function h(η̂) = c_m * η̂.
  Vector3 operator()(const Unit3& eta_hat,
                     OptionalJacobian<3, 2> H = {}) const {
    // point3 writes its own Jacobian, so scale afterwards.
    const Point3 direction = eta_hat.point3(H);
    if (H) *H *= c_m_;
    return c_m_ * direction;
  }
};

//---------------------------------------------------------------------------
// example values
//---------------------------------------------------------------------------

const G Q0;  // Rot3() == identity (observer state \hat{Q})

const M eta_ref = Unit3(0, 0, 1);  // \bar{η}
const typename Symmetry::Orbit phi_ref(
    eta_ref);  // state action on reference direction
const double c_m = 1.2;

// Test dynamics on manifold: eta x omega
Rot3 Q1 = Rot3::Ypr(0.1, 0.2, 0.3);   // non-identity rotation
Unit3 eta1 = Q1.inverse() * eta_ref;  // eta1 = Q1 * eta_ref

const Vector3 omega(0.1, -0.2, 0.3);
Lift lift_omega(omega);

const InputOrbit psi_u(omega);
}  // namespace attitude_example

//==============================================================================
// Symmetry Tests
//==============================================================================
TEST(EquivariantFilter_Attitude, Symmetry) {
  using namespace attitude_example;

  // Check the state action
  Unit3 expected_eta1 = phi_ref(Q1);
  EXPECT(assert_equal(expected_eta1, eta1));

  // Check jacobian at identity
  Matrix Dphi0;
  phi_ref(Q0, Dphi0);

  // Expected Jacobian
  Matrix expected_Dphi0;
  Q0.unrotate(eta_ref, expected_Dphi0);

  EXPECT(assert_equal(expected_Dphi0, Dphi0));
}

//==============================================================================
// Test the induced group action
TEST(EquivariantFilter_Attitude, InducedGroupAction) {
  using namespace attitude_example;

  // Purposefully a vector field which is not the dynamics
  const auto f = [](const Unit3& eta) -> Vector2 {
    auto p = eta.point3();
    return Vector2(p.z() * p.x() * 3, p.y() * p.x());
  };

  using InducedField = Symmetry::InducedVectorField<decltype(f)>;
  const InducedField fInduced(Q1, f);

  // A bit of a self-fulfilling prophecy:
  Matrix H;
  Symmetry::Diffeomorphism{Q1}(Symmetry::Orbit{eta1}(Q1.inverse()), H);
  const Vector2 expected = H * f(Symmetry::Orbit{eta1}(Q1.inverse()));
  const Vector2 actual = fInduced(eta1);
  EXPECT(assert_equal(expected, actual));
}

//==============================================================================
TEST(EquivariantFilter_Attitude, DiffeomorphismPushForward) {
  using namespace attitude_example;

  const Vector2 tangent{0.1, -0.2};
  Matrix H;
  Symmetry::Diffeomorphism phi_Q1(Q1);
  phi_Q1(eta1, H);
  const Vector2 expected = H * tangent;
  const Vector2 actual = phi_Q1.pushforward(eta1, tangent);
  EXPECT(assert_equal(expected, actual));
}

//==============================================================================
// Dynamics Tests
//==============================================================================
// Manifold dynamics at reference direction (north pole)
TEST(EquivariantFilter_Attitude, Dynamics0) {
  using namespace attitude_example;

  Vector3 omega_y(0, 0.1, 0);

  // The tangent space basis for Unit3(0,0,1) is the Y and -X axis.
  // std::cout << eta_ref.basis() << std::endl; // comment out to verify

  // Manually calculate expected dynamics at eta_ref (Unit3(0,0,1))
  // eta_ref.cross(omega_y) = (0,0,1) x (0, 0.1, 0) = (-0.1, 0, 0)
  // The projection of (-0.1, 0, 0) onto tangent space will be (0, 0.1).
  Vector2 expected_dynamics = Vector2(0, 0.1);

  Vector2 actual_dynamics = dynamics(omega_y, eta_ref);
  EXPECT(assert_equal(expected_dynamics, actual_dynamics));
}

//==============================================================================
// Regression test and affine map construction
TEST(EquivariantFilter_Attitude, Dynamics) {
  using namespace attitude_example;

  // Manually calculate expected dynamics: eta1.cross(omega)
  Matrix23 B1t = eta1.basis().transpose();
  Vector2 expected = B1t * eta1.cross(omega);
  Vector2 actual = dynamics(omega, eta1);
  EXPECT(assert_equal(expected, actual));

  // Regression test
  EXPECT(assert_equal(Vector2(0.143203, -0.279723), actual, 1e-5));

  // Define vector fields f_i(eta) = eta.cross(e_i)
  Vector2 f1 = B1t * eta1.cross(Point3(1, 0, 0));
  Vector2 f2 = B1t * eta1.cross(Point3(0, 1, 0));
  Vector2 f3 = B1t * eta1.cross(Point3(0, 0, 1));

  // Dynamics as an affine map: omega_x * f1 + omega_y * f2 + omega_z * f3
  Vector2 actual_affine_map = omega.x() * f1 + omega.y() * f2 + omega.z() * f3;
  EXPECT(assert_equal(actual, actual_affine_map));
}

//==============================================================================
TEST(EquivariantFilter_Attitude, InputEquivariance) {
  using namespace attitude_example;

  // Use knowledge that induced dynamics is eta .cross(Q.inverse() * omega)
  const Vector2 expected =
      eta1.basis().transpose() * eta1.cross(Q1.inverse() * omega);
  const Vector2 equivariant =
      dynamics(psi_u(Q1), eta1);  // dynamics given input transformed omega
  EXPECT(assert_equal(expected, equivariant));

  // f_u is dynamics given input omega
  auto f_u = [&](const M& eta) -> Vector2 { return dynamics(omega, eta); };
  using InducedField = Symmetry::InducedVectorField<decltype(f_u)>;
  const InducedField fInduced(Q1, f_u);

  const Vector2 induced = fInduced(eta1);
  EXPECT(assert_equal(expected, induced));
  // Equation (3.5) in Fornasier thesis
  EXPECT(assert_equal(induced, equivariant));
}

//==============================================================================
// Lift shadow property: Dφ_{η}(I) Λ(η, u) = ξ̇ on the manifold.
TEST(EquivariantFilter_Attitude, LiftShadowManifoldDynamics) {
  using namespace attitude_example;

  Symmetry::Orbit phi_eta(eta1);
  Matrix H;
  phi_eta(G::Identity(), H);  // derivative w.r.t. group at identity

  const Vector3 lifted = Lift(omega)(eta1);
  const Vector2 shadow = H * lifted;

  const Vector2 manifold = dynamics(omega, eta1);
  EXPECT(assert_equal(shadow, manifold));
}

//==============================================================================
TEST(EquivariantFilter_Attitude, LiftEquivariance) {
  using namespace attitude_example;

  // Λ should satisfy Ad_{Q^{-1}} Λ(η, u) = Λ(φ_Q(η), ψ_Q(u)).
  Lift lift_omega(omega);
  InputOrbit psi_u(omega);

  const Rot3 g = Q1;
  const Vector3 lifted = lift_omega(eta1);
  const Vector3 lifted_ad = g.inverse().AdjointMap() * lifted;

  const Vector3 psi_u_transformed = psi_u(g);
  const Vector3 lifted_equivariant =
      Lift(psi_u_transformed)(Symmetry::Orbit{eta1}(g));

  EXPECT(assert_equal(lifted_ad, lifted_equivariant));
}

//==============================================================================
// Prediction Tests
//==============================================================================
TEST(EquivariantFilter_Attitude, Predict) {
  using namespace attitude_example;

  // Initial group and reference state: both identity.x
  Matrix2 Sigma0 = 0.01 * I_2x2;
  EquivariantFilter<M, Symmetry> filter(eta_ref, Sigma0);

  // --- Perform prediction through EqF ---
  InputOrbit psi_u(omega);
  Matrix3 Sigma_u = 0.1 * I_3x3;
  Matrix3 Q = processNoise(Sigma_u);
  Matrix23 B = inputMatrixB(Q0);
  Matrix2 Qc = B * Q * B.transpose();  // manifold continuous-time covariance
  const double dt = 0.01;
  filter.predict(lift_omega, psi_u, Qc, dt);

  // --- Expected result ---
  // X_new = X_old * Exp(omega * dt) (Right action predict or left?)
  // Filter impl: X_ = Compose(X_, Exp(Lambda*dt)) -> X_new = X_old * Exp.
  const G X_expected = Q0 * Rot3::Expmap(omega * dt);
  EXPECT(assert_equal(X_expected, filter.groupEstimate()));

  // --- Expected covariance update ---
  Matrix2 Phi = I_2x2;

  // Qc is already on manifold, continuous-time.
  Matrix2 Q_process = Qc * dt;
  Matrix2 P_expected = Phi * Sigma0 * Phi.transpose() + Q_process;
  EXPECT(assert_equal(P_expected, filter.errorCovariance()));

  // state() should be the rotated reference direction on S^2
  const Unit3 state_expected(X_expected.unrotate(eta_ref.point3()));
  EXPECT(assert_equal(state_expected, filter.state()));
}

/* ************************************************************************* */
namespace covariance_transport {
using namespace attitude_example;

// Anisotropic and correlated, so that J * P * J^T and J^T * P * J differ.
const Matrix2 kSigma0{{4e-4, 1.5e-4},  //
                      {1.5e-4, 1e-4}};

/// Map an error perturbation at the reference state to the corresponding
/// perturbation of the current state, using only the state relation
/// η = φ_g(Retract(η_ref, ε)). This deliberately avoids the filter's own
/// Jacobian, so the tests below do not restate the formula they check.
Vector2 statePerturbation(const G& g, const Vector2& epsilon) {
  const typename Symmetry::Diffeomorphism phi_g(g);
  const M eta_hat = phi_g(eta_ref);
  const M eta = phi_g(traits<M>::Retract(eta_ref, epsilon));
  return traits<M>::Local(eta_hat, eta);
}

// With the group estimate at identity the error coordinates already live in the
// tangent space at the current state, so covariance() returns P unchanged.
TEST(EquivariantFilter_Attitude, CovarianceAtIdentity) {
  EquivariantFilter<M, Symmetry> filter(eta_ref, kSigma0);
  EXPECT(assert_equal(kSigma0, filter.covariance(), 1e-9));
}

// covariance() pushes the error covariance forward through the differential of
// the group action, that is J * P * J^T and not J^T * P * J.
TEST(EquivariantFilter_Attitude, CovariancePushforward) {
  EquivariantFilter<M, Symmetry> filter(eta_ref, kSigma0, Q1);

  // Differential of the state relation at zero error, obtained numerically.
  const Matrix2 D = numericalDerivative11<Vector2, Vector2>(
      [](const Vector2& epsilon) { return statePerturbation(Q1, epsilon); },
      Vector2::Zero());

  // The test only has teeth if the two congruence directions disagree here.
  EXPECT((D * kSigma0 * D.transpose() - D.transpose() * kSigma0 * D).norm() >
         1e-5);

  EXPECT(assert_equal(Matrix2(D * kSigma0 * D.transpose()), filter.covariance(),
                      1e-7));
}

// Sampled errors at the reference state, mapped to the current state, have a
// sample covariance matching covariance().
TEST(EquivariantFilter_Attitude, CovarianceMonteCarlo) {
  EquivariantFilter<M, Symmetry> filter(eta_ref, kSigma0, Q1);

  const Matrix2 L = Eigen::LLT<Matrix2>(kSigma0).matrixL();
  std::mt19937 rng(42);
  std::normal_distribution<double> gauss(0.0, 1.0);

  constexpr size_t numSamples = 200000;
  Matrix2 sampleCovariance = Matrix2::Zero();
  for (size_t i = 0; i < numSamples; i++) {
    const Vector2 epsilon = L * Vector2(gauss(rng), gauss(rng));
    const Vector2 delta = statePerturbation(Q1, epsilon);
    sampleCovariance += delta * delta.transpose();
  }
  sampleCovariance /= numSamples;

  EXPECT(assert_equal(sampleCovariance, filter.covariance(), 1e-5));
}

}  // namespace covariance_transport
/* ************************************************************************* */

//==============================================================================
TEST(EquivariantFilter_Attitude, Update) {
  using namespace attitude_example;

  // 1. Setup Filter
  Matrix2 Sigma0 = 0.01 * I_2x2;
  EquivariantFilter<M, Symmetry> filter(eta_ref, Sigma0);

  // 2. Predict to move away from identity
  const double dt = 0.01;
  Matrix3 Sigma_u = 0.1 * I_3x3;
  Matrix3 Q = processNoise(Sigma_u);
  Matrix23 B = inputMatrixB(Q0);
  Matrix2 Qc = B * Q * B.transpose();  // manifold continuous-time covariance
  filter.predict(lift_omega, psi_u, Qc, dt);

  const G Q_before = filter.groupEstimate();
  const Matrix2 P_before = filter.errorCovariance();

  // 3. Setup Measurement
  const Vector3 z = c_m * eta_ref.point3();
  const Matrix3 R_meas = 0.01 * I_3x3;
  MeasurementFunctor h(c_m);

  // 4. Run Filter Update
  filter.update(h, z, R_meas);

  const G Q_after = filter.groupEstimate();
  const Matrix2 P_after = filter.errorCovariance();

  // 5. Run Manual Update (Mirroring EquivariantFilter implementation)

  // Re-calculate InnovationLift (PseudoInverse of Dphi at identity)
  Matrix23 Dphi0;
  phi_ref(G::Identity(), Dphi0);
  Matrix32 InnovationLift =
      Dphi0.completeOrthogonalDecomposition().pseudoInverse();

  // Re-calculate Measurement Matrix H
  const M eta_hat = phi_ref(Q_before);
  Matrix H;
  const Vector3 z_hat = h(eta_hat, H);

  // Calculate Gain K
  Matrix S = H * P_before * H.transpose() + R_meas;
  Matrix K = P_before * H.transpose() * S.inverse();

  // Calculate Innovation
  const Vector3 innovation = z_hat - z;

  // Calculate Correction
  Vector2 delta_xi = -K * innovation;
  Vector3 delta_x = InnovationLift * delta_xi;

  // Update State: X_new = Exp(delta_x) * X_old (Left Update)
  const G X_expected = Rot3::Expmap(delta_x) * Q_before;

  // Update Covariance: Joseph Form
  Matrix2 I_KC = Matrix2::Identity() - K * H;
  Matrix2 P_expected =
      I_KC * P_before * I_KC.transpose() + K * R_meas * K.transpose();

  // 6. Assertions
  EXPECT(assert_equal(P_expected, P_after, 1e-9));
  EXPECT(assert_equal(X_expected, Q_after, 1e-9));

  const Unit3 state_expected(Q_after.unrotate(eta_ref.point3()));
  EXPECT(assert_equal(state_expected, filter.state(), 1e-9));
}

//==============================================================================
TEST(EquivariantFilter_Attitude, CheckMatrices) {
  using namespace attitude_example;

  // Initial group and reference state: both identity.
  const G Q0 = Rot3::Ypr(0.1, 0.2, 0.3);
  const M eta_ref(1, 2, 3);
  Matrix2 Sigma0 = 0.01 * I_2x2;

  EquivariantFilter<M, Symmetry> filter(eta_ref, Sigma0);

  // Check A matrix
  InputOrbit psi_u(omega);
  Matrix2 A_computed =
      filter.computeErrorDynamicsMatrix<Lift, InputOrbit>(psi_u);
  Matrix2 A_provided = stateMatrixA(Q0);
  EXPECT(assert_equal(A_provided, A_computed));

  // Check C matrix
  const M eta_hat = phi_ref(Q0);
  Matrix C_computed;
  MeasurementFunctor h(c_m);
  h(eta_hat, C_computed);
  EXPECT(C_computed.rows() == 3 && C_computed.cols() == 2);
}

/* ************************************************************************* */
namespace left_action_guard {

using M = Rot3;
using G = Rot3;

struct Symmetry : public GroupAction<Symmetry, G, M> {
  static constexpr ActionType type = ActionType::Left;

  M operator()(const G& group, const M& state,
               OptionalJacobian<3, 3> H_group = {},
               OptionalJacobian<3, 3> H_state = {}) const {
    return group.compose(state, H_group, H_state);
  }
};

struct Lift {
  explicit Lift(const Vector3& omega) : omega_(omega) {}

  Vector3 operator()(const M&, OptionalJacobian<3, 3> H = {}) const {
    if (H) *H = Z_3x3;
    return omega_;
  }

 private:
  Vector3 omega_;
};

// A constant lift is only equivariant if the input action carries the state
// dependence instead: Lambda(phi_X(xi), psi_X(u)) = psi_X(u) must equal
// Ad_X Lambda(xi,u) = Ad_X u, so psi_X(u) = Ad_X u. With a trivial input
// action the lift is not equivariant, and the error is stationary only at
// X0 = Identity.
struct InputAction : public GroupAction<InputAction, G, Vector3> {
  static constexpr ActionType type = ActionType::Left;

  Vector3 operator()(const G& group, const Vector3& input) const {
    return group.rotate(input);  // Ad_X u
  }
};

using InputOrbit = InputAction::Orbit;

// A state-independent lift, made equivariant by the adjoint input action,
// describes spatial dynamics xi_dot = omega^ xi. The error is then exactly
// stationary and A = 0 -- for any initial group estimate, not just Identity.
TEST(EquivariantFilter_LeftAction, ConstantLiftGivesZeroDynamics) {
  const Vector3 omega{0.1, -0.2, 0.3};
  const Lift lift(omega);
  const InputOrbit inputOrbit(omega);
  const double dt = 0.01;

  const G X0 = Rot3::Expmap(Vector3(0.4, 0.1, -0.7));
  EquivariantFilter<M, Symmetry> filter(M::Identity(), I_3x3, X0);

  const Matrix3 A =
      filter.computeErrorDynamicsMatrix<Lift, InputOrbit>(inputOrbit);
  EXPECT(assert_equal(Matrix3(Z_3x3), A, 1e-9));

  // Mean follows the spatial dynamics: xi_hat+ = Exp(omega dt) xi_hat.
  filter.predict(lift, inputOrbit, Z_3x3, dt);
  EXPECT(assert_equal(Rot3::Expmap(omega * dt) * X0, filter.state(), 1e-9));
}

}  // namespace left_action_guard

/* ************************************************************************* */
// Left-regular action of a noncommutative group on itself, with the genuine
// equivariant lift Lambda(xi,u) = Ad_xi u. Its differential at the origin is
// -ad_u, which is exactly the transport the automatic path must reproduce.
namespace left_regular {

using M = Rot3;
using G = Rot3;

struct Symmetry : public GroupAction<Symmetry, G, M> {
  static constexpr ActionType type = ActionType::Left;
  M operator()(const G& group, const M& state,
               OptionalJacobian<3, 3> H_group = {},
               OptionalJacobian<3, 3> H_state = {}) const {
    return group.compose(state, H_group, H_state);
  }
};

/// Equivariant lift Lambda(xi, u) = Ad_xi u, with its true Jacobian.
///
/// The Ad is here because the frames do not match, not because the action is a
/// left one. A left action's fundamental vector field is spatial, while this
/// system's velocity is a body velocity (xi+ = xi Exp(u h)), so the lift
/// condition X_Lambda(xi) = f_u(xi) forces Ad_xi u. The mirror case exists for
/// right actions: their fundamental field is a body velocity, so a spatially
/// specified system needs Ad_{xi^-1} u there. A constant lift is correct only
/// when the two frames agree, which is why every right-action fixture above
/// has one -- IMU inputs are body quantities and those actions are body-framed.
struct Lift {
  explicit Lift(const Vector3& u) : u_(u) {}
  Vector3 operator()(const M& xi, OptionalJacobian<3, 3> H = {}) const {
    if (H) *H = -Rot3::adjointMap(u_);  // d/dxi Ad_xi u at xi = Identity
    return xi.rotate(u_);
  }
 private:
  Vector3 u_;
};

/// Equivariance then forces the input action: Lambda(phi_X(xi), psi_X(u)) =
/// Ad_X Lambda(xi,u) reduces to psi_X(u) = u here. The lift and the input
/// action are not independent choices.
struct InputAction : public GroupAction<InputAction, G, Vector3> {
  static constexpr ActionType type = ActionType::Left;
  Vector3 operator()(const G&, const Vector3& input) const { return input; }
};
using InputOrbit = InputAction::Orbit;

TEST(EquivariantFilter_LeftRegular, ErrorDynamicsIsMinusAdU) {
  const Vector3 u{0.3, -0.2, 0.5};
  // A does not depend on X0 here: the input action is trivial, so u_origin = u,
  // and D_lift is taken at the origin. A non-identity estimate is used only to
  // keep the fixture from being accidentally special.
  const G X0 = Rot3::Expmap(Vector3(0.4, 0.1, -0.7));
  EquivariantFilter<M, Symmetry> filter(M::Identity(), I_3x3, X0);

  const InputOrbit inputOrbit(u);
  const Matrix3 A =
      filter.computeErrorDynamicsMatrix<Lift, InputOrbit>(inputOrbit);
  // -ad_u: the same term LieGroupEKF::transitionMatrix() obtains as Df - ad_xi
  // for a left-invariant error with state-independent dynamics (Df = 0).
  EXPECT(assert_equal(Matrix3(-Rot3::adjointMap(u)), A, 1e-9));
}

// Compare the automatic transition against the exact nonlinear error flow.
// True system: xi+ = xi Exp(u h); estimate right-composes; E = Xhat^-1 xi.
TEST(EquivariantFilter_LeftRegular, MatchesNumericalErrorFlow) {
  const Vector3 u{0.3, -0.2, 0.5};
  const G X0 = Rot3::Expmap(Vector3(0.4, 0.1, -0.7));
  const double h = 1e-3;

  EquivariantFilter<M, Symmetry> filter(M::Identity(), I_3x3, X0);
  const InputOrbit inputOrbit(u);
  const Matrix3 A =
      filter.computeErrorDynamicsMatrix<Lift, InputOrbit>(inputOrbit);
  const Matrix3 Phi = I_3x3 + A * h;

  const Vector3 eps0{1e-4, -2e-4, 1.5e-4};
  const Rot3 xi0 = X0.compose(Rot3::Expmap(eps0));
  const Rot3 xi1 = xi0.compose(Rot3::Expmap(u * h));
  const Rot3 X1 = X0.compose(Rot3::Expmap(u * h));
  const Vector3 eps1 = Rot3::Logmap(X1.inverse().compose(xi1));

  EXPECT(assert_equal(Vector3(Phi * eps0), eps1, 1e-9));
  // A = 0, what a zero-Jacobian lift would give, does not reproduce the flow.
  EXPECT(!assert_equal(Vector3(eps0), Vector3(eps1), 1e-9));
}

// The mean must also be right: the prediction right-composes, so the lifted
// increment has to be the origin-frame Lambda(xi_ref, u_origin).
TEST(EquivariantFilter_LeftRegular, MeanPropagation) {
  const Vector3 u{0.3, -0.2, 0.5};
  const G X0 = Rot3::Expmap(Vector3(0.4, 0.1, -0.7));
  const double h = 0.01;

  EquivariantFilter<M, Symmetry> filter(M::Identity(), I_3x3, X0);
  filter.predict(Lift(u), InputOrbit(u), Z_3x3, h);
  EXPECT(assert_equal(X0.compose(Rot3::Expmap(u * h)), filter.state(), 1e-9));
}

// predictWithTransition() must use the same frame as predict(); with a
// state-dependent lift and a left action, taking the lift at the estimate
// would give a silently wrong mean.
TEST(EquivariantFilter_LeftRegular, PredictWithTransitionMatchesPredict) {
  const Vector3 u{0.3, -0.2, 0.5};
  const G X0 = Rot3::Expmap(Vector3(0.4, 0.1, -0.7));
  const double h = 0.01;

  EquivariantFilter<M, Symmetry> a(M::Identity(), I_3x3, X0);
  a.predict(Lift(u), InputOrbit(u), Z_3x3, h);

  EquivariantFilter<M, Symmetry> b(M::Identity(), I_3x3, X0);
  const Matrix3 A =
      b.computeErrorDynamicsMatrix<Lift, InputOrbit>(InputOrbit(u));
  b.predictWithTransition(Lift(u), InputOrbit(u), Matrix3(I_3x3 + A * h),
                          Matrix3(Z_3x3), h);

  EXPECT(assert_equal(a.state(), b.state(), 1e-12));
  EXPECT(assert_equal(X0.compose(Rot3::Expmap(u * h)), b.state(), 1e-9));
}

// A measurement correction lives in error coordinates at the reference state,
// so it must be composed on the side that keeps it there -- the opposite side
// from a right action. Composing on the wrong side rotates the correction by
// Ad_{X^-1}, which is invisible only when X is the identity.
TEST(EquivariantFilter_LeftRegular, UpdateAppliesCorrectionAtTheOrigin) {
  const G X0 = Rot3::Expmap(Vector3(0.4, 0.1, -0.7));
  EquivariantFilter<M, Symmetry> filter(M::Identity(), I_3x3, X0);

  const Vector3 eps{2e-3, -3e-3, 1.5e-3};
  const Rot3 xi_true = X0.compose(Rot3::Expmap(eps));  // error is eps

  // Measure the error coordinates directly: H = I and a tiny R give K ~ I,
  // so the correction delta_xi is the measured vector.
  const Vector3 v{1e-3, -1.5e-3, 0.8e-3};
  filter.update<Vector3>(Vector3::Zero(), Matrix3(I_3x3), v,
                         Matrix3(1e-9 * I_3x3));

  const Vector3 epsAfter =
      Rot3::Logmap(filter.groupEstimate().inverse().compose(xi_true));
  EXPECT(assert_equal(Vector3(eps - v), epsAfter, 1e-5));
}

// gtsam#2753's counterexample, in the terms the issue states it: the
// left-regular action on a noncommutative group, prediction Yhat+ = Yhat
// Exp(u h), and error flow E+ = Exp(-u h) E Exp(u h). The issue observes
// A = 0 for a state-independent lift; that lift does not satisfy the lift
// condition for this system, and the one that does returns exactly the
// -ad_u the issue expects.
TEST(EquivariantFilter_LeftRegular, Issue2753Counterexample) {
  const Vector3 u{0.3, -0.2, 0.5};
  const G X0 = Rot3::Expmap(Vector3(0.4, 0.1, -0.7));
  const double h = 1e-3;

  EquivariantFilter<M, Symmetry> filter(M::Identity(), I_3x3, X0);
  const InputOrbit inputOrbit(u);
  const Matrix3 A =
      filter.computeErrorDynamicsMatrix<Lift, InputOrbit>(inputOrbit);

  // the continuous generator the issue states
  EXPECT(assert_equal(Matrix3(-Rot3::adjointMap(u)), A, 1e-9));

  // the discrete transition the issue states, Ad_Exp(-u h)
  EXPECT(assert_equal(Matrix3(Rot3::Expmap(-u * h).matrix()),
                      Matrix3(filter.transitionMatrix<8>(A, h)), 1e-9));

  // the prediction the issue states, Yhat+ = Yhat Exp(u h)
  filter.predict(Lift(u), inputOrbit, Z_3x3, h);
  EXPECT(assert_equal(X0.compose(Rot3::Expmap(u * h)), filter.groupEstimate(),
                      1e-9));

  // the error flow the issue states, E+ = Exp(-u h) E Exp(u h)
  const Rot3 E0 = Rot3::Expmap(Vector3(1e-4, -2e-4, 1.5e-4));
  const Rot3 xi1 = X0.compose(E0).compose(Rot3::Expmap(u * h));
  const Rot3 E1 = filter.groupEstimate().inverse().compose(xi1);
  EXPECT(assert_equal(
      Rot3::Expmap(-u * h).compose(E0).compose(Rot3::Expmap(u * h)), E1, 1e-9));
}

}  // namespace left_regular
/* ************************************************************************* */

/* ************************************************************************* */
// Left action with DimM != DimG and a non-constant equivariant lift.
//   G = SO(3) (dim 3) acting on M = S^2 (dim 2) by phi_Q(eta) = Q eta.
// Transitive but NOT free: the stabiliser of eta is span{eta}. The minimal
// equivariant lift Lambda(eta,w) = w - (w.eta) eta removes that freedom and,
// unlike every other fixture in this file, has a non-zero Jacobian.
namespace left_sphere {

using M = Unit3;
using G = Rot3;

struct Symmetry : public GroupAction<Symmetry, G, M> {
  static constexpr ActionType type = ActionType::Left;
  M operator()(const G& Q, const M& eta, OptionalJacobian<2, 3> H_Q = {},
               OptionalJacobian<2, 2> H_eta = {}) const {
    return Q.rotate(eta, H_Q, H_eta);
  }
};

/// Lambda(eta, w) = w - (w . eta) eta, with its Jacobian in local coordinates:
///   d/d(eta) = -eta (w^T B) - (w . eta) B,  B = eta.basis()
struct Lift {
  explicit Lift(const Vector3& w) : w_(w) {}
  Vector3 operator()(const M& eta, OptionalJacobian<3, 2> H = {}) const {
    const Vector3 n = eta.unitVector();
    const double wn = w_.dot(n);
    if (H) {
      const Matrix32 B = eta.basis();
      *H = -n * (w_.transpose() * B) - wn * B;
    }
    return w_ - wn * n;
  }
 private:
  Vector3 w_;
};

/// Left input action psi_Q(w) = Q w makes the lift above equivariant.
struct InputAction : public GroupAction<InputAction, G, Vector3> {
  static constexpr ActionType type = ActionType::Left;
  Vector3 operator()(const G& Q, const Vector3& w) const { return Q.rotate(w); }
};
using InputOrbit = InputAction::Orbit;

TEST(EquivariantFilter_LeftSphere, ErrorDynamicsMatchesNumericalGenerator) {
  const Unit3 eta_ref(0.2, -0.5, 0.84);
  const G X0 = Rot3::Expmap(Vector3(0.4, 0.1, -0.7));
  const Vector3 w{0.3, -0.2, 0.5};

  EquivariantFilter<M, Symmetry> filter(eta_ref, I_2x2, X0);
  const InputOrbit inputOrbit(w);
  const Matrix2 A =
      filter.computeErrorDynamicsMatrix<Lift, InputOrbit>(inputOrbit);

  // Exact error velocity in local coordinates:
  //   E = Q^T eta,  Edot = (u_origin - Lambda(eta_ref, u_origin)) x E
  const Vector3 u_origin = X0.inverse().rotate(w);
  const Vector3 lambda = Lift(u_origin)(eta_ref);
  const Matrix32 B = eta_ref.basis();
  auto errorVelocity = [&](const Vector2& eps) -> Vector2 {
    const Vector3 E = eta_ref.retract(eps).unitVector();
    return B.transpose() * (u_origin - lambda).cross(E);
  };
  const Matrix2 A_numerical =
      numericalDerivative11<Vector2, Vector2>(errorVelocity, Vector2::Zero());

  EXPECT(assert_equal(A_numerical, A, 1e-7));
  // The origin is an equilibrium of the error.
  EXPECT(assert_equal(Vector2::Zero().eval(), errorVelocity(Vector2::Zero()),
                      1e-9));
  // A constant lift would report zero here, which is wrong by ~0.38.
  EXPECT(!assert_equal(Matrix2(Z_2x2), Matrix2(A), 1e-3));
}

}  // namespace left_sphere

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
