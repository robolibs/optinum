#include <doctest/doctest.h>
#include <optinum/simd/pack/quaternion.hpp>

#include <cmath>

using optinum::simd::pack;
namespace dp = datapod;

TEST_CASE("Quaternion pack construction") {
    using quat_t = dp::mat::Quaternion<float>;
    using qpack = pack<quat_t, 4>;

    // Zero initialization
    auto z = qpack::zero();
    CHECK(z.w()[0] == doctest::Approx(0.0f));
    CHECK(z.x()[0] == doctest::Approx(0.0f));
    CHECK(z.y()[0] == doctest::Approx(0.0f));
    CHECK(z.z()[0] == doctest::Approx(0.0f));

    // Identity quaternion
    auto id = qpack::identity();
    CHECK(id.w()[0] == doctest::Approx(1.0f));
    CHECK(id.x()[0] == doctest::Approx(0.0f));
    CHECK(id.y()[0] == doctest::Approx(0.0f));
    CHECK(id.z()[0] == doctest::Approx(0.0f));

    // Broadcast single value
    qpack p(quat_t{0.707f, 0.0f, 0.707f, 0.0f}); // 90 degree rotation about Y
    CHECK(p.w()[0] == doctest::Approx(0.707f));
    CHECK(p.x()[0] == doctest::Approx(0.0f));
    CHECK(p.y()[0] == doctest::Approx(0.707f));
    CHECK(p.z()[0] == doctest::Approx(0.0f));

    // From component packs
    pack<float, 4> pw(1.0f), px(0.0f), py(0.0f), pz(0.0f);
    qpack p2(pw, px, py, pz);
    CHECK(p2.w()[0] == doctest::Approx(1.0f));
    CHECK(p2.x()[0] == doctest::Approx(0.0f));
}

TEST_CASE("Quaternion pack arithmetic - addition and subtraction") {
    using quat_t = dp::mat::Quaternion<double>;
    using qpack = pack<quat_t, 2>;

    // Create two quaternion packs
    pack<double, 2> w1, x1, y1, z1, w2, x2, y2, z2;
    w1.data_[0] = 1.0;
    w1.data_[1] = 2.0;
    x1.data_[0] = 0.0;
    x1.data_[1] = 0.1;
    y1.data_[0] = 0.0;
    y1.data_[1] = 0.2;
    z1.data_[0] = 0.0;
    z1.data_[1] = 0.3;

    w2.data_[0] = 0.5;
    w2.data_[1] = 1.0;
    x2.data_[0] = 0.1;
    x2.data_[1] = 0.2;
    y2.data_[0] = 0.2;
    y2.data_[1] = 0.3;
    z2.data_[0] = 0.3;
    z2.data_[1] = 0.4;

    qpack a(w1, x1, y1, z1);
    qpack b(w2, x2, y2, z2);

    // Addition
    auto sum = a + b;
    CHECK(sum.w()[0] == doctest::Approx(1.5));
    CHECK(sum.x()[0] == doctest::Approx(0.1));
    CHECK(sum.y()[0] == doctest::Approx(0.2));
    CHECK(sum.z()[0] == doctest::Approx(0.3));

    // Subtraction
    auto diff = a - b;
    CHECK(diff.w()[0] == doctest::Approx(0.5));
    CHECK(diff.x()[0] == doctest::Approx(-0.1));
}

TEST_CASE("Quaternion pack Hamilton product") {
    using quat_t = dp::mat::Quaternion<double>;
    using qpack = pack<quat_t, 2>;

    // Test quaternion multiplication (Hamilton product)
    // i * j = k
    pack<double, 2> w1(0.0), x1(1.0), y1(0.0), z1(0.0); // pure i
    pack<double, 2> w2(0.0), x2(0.0), y2(1.0), z2(0.0); // pure j

    qpack qi(w1, x1, y1, z1);
    qpack qj(w2, x2, y2, z2);

    auto qk = qi * qj;
    CHECK(qk.w()[0] == doctest::Approx(0.0));
    CHECK(qk.x()[0] == doctest::Approx(0.0));
    CHECK(qk.y()[0] == doctest::Approx(0.0));
    CHECK(qk.z()[0] == doctest::Approx(1.0)); // k

    // Test j * i = -k (non-commutative!)
    auto neg_k = qj * qi;
    CHECK(neg_k.w()[0] == doctest::Approx(0.0));
    CHECK(neg_k.x()[0] == doctest::Approx(0.0));
    CHECK(neg_k.y()[0] == doctest::Approx(0.0));
    CHECK(neg_k.z()[0] == doctest::Approx(-1.0)); // -k

    // Test identity multiplication
    auto id = qpack::identity();
    auto result = id * qi;
    CHECK(result.w()[0] == doctest::Approx(0.0));
    CHECK(result.x()[0] == doctest::Approx(1.0));
    CHECK(result.y()[0] == doctest::Approx(0.0));
    CHECK(result.z()[0] == doctest::Approx(0.0));
}

TEST_CASE("Quaternion pack conjugate and norm") {
    using quat_t = dp::mat::Quaternion<float>;
    using qpack = pack<quat_t, 4>;

    pack<float, 4> w, x, y, z;
    // Unit quaternion: (0.5, 0.5, 0.5, 0.5) - norm = 1
    w.data_[0] = 0.5f;
    x.data_[0] = 0.5f;
    y.data_[0] = 0.5f;
    z.data_[0] = 0.5f;
    // (1, 0, 0, 0) - identity
    w.data_[1] = 1.0f;
    x.data_[1] = 0.0f;
    y.data_[1] = 0.0f;
    z.data_[1] = 0.0f;
    // (3, 4, 0, 0) - norm = 5
    w.data_[2] = 3.0f;
    x.data_[2] = 4.0f;
    y.data_[2] = 0.0f;
    z.data_[2] = 0.0f;
    // (0, 1, 0, 0) - pure i
    w.data_[3] = 0.0f;
    x.data_[3] = 1.0f;
    y.data_[3] = 0.0f;
    z.data_[3] = 0.0f;

    qpack q(w, x, y, z);

    // Conjugate
    auto conj = q.conjugate();
    CHECK(conj.w()[0] == doctest::Approx(0.5f));
    CHECK(conj.x()[0] == doctest::Approx(-0.5f));
    CHECK(conj.y()[0] == doctest::Approx(-0.5f));
    CHECK(conj.z()[0] == doctest::Approx(-0.5f));

    CHECK(conj.w()[3] == doctest::Approx(0.0f));
    CHECK(conj.x()[3] == doctest::Approx(-1.0f));

    // Norm
    auto n = q.norm();
    CHECK(n[0] == doctest::Approx(1.0f));
    CHECK(n[1] == doctest::Approx(1.0f));
    CHECK(n[2] == doctest::Approx(5.0f));
    CHECK(n[3] == doctest::Approx(1.0f));

    // Norm squared
    auto n2 = q.norm_squared();
    CHECK(n2[0] == doctest::Approx(1.0f));
    CHECK(n2[2] == doctest::Approx(25.0f));
}

TEST_CASE("Quaternion pack inverse and normalized") {
    using quat_t = dp::mat::Quaternion<double>;
    using qpack = pack<quat_t, 2>;

    // (3, 4, 0, 0) - norm = 5
    pack<double, 2> w, x, y, z;
    w.data_[0] = 3.0;
    x.data_[0] = 4.0;
    y.data_[0] = 0.0;
    z.data_[0] = 0.0;
    w.data_[1] = 1.0;
    x.data_[1] = 1.0;
    y.data_[1] = 1.0;
    z.data_[1] = 1.0; // norm = 2

    qpack q(w, x, y, z);

    // Inverse: q^-1 = conj(q) / |q|^2
    auto inv = q.inverse();
    // For (3, 4, 0, 0): inverse = (3, -4, 0, 0) / 25 = (0.12, -0.16, 0, 0)
    CHECK(inv.w()[0] == doctest::Approx(0.12));
    CHECK(inv.x()[0] == doctest::Approx(-0.16));

    // Verify q * q^-1 = identity
    auto product = q * inv;
    CHECK(product.w()[0] == doctest::Approx(1.0));
    CHECK(product.x()[0] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(product.y()[0] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(product.z()[0] == doctest::Approx(0.0).epsilon(1e-10));

    // Normalized
    auto norm = q.normalized();
    CHECK(norm.w()[0] == doctest::Approx(0.6)); // 3/5
    CHECK(norm.x()[0] == doctest::Approx(0.8)); // 4/5
    CHECK(norm.w()[1] == doctest::Approx(0.5)); // 1/2
    CHECK(norm.x()[1] == doctest::Approx(0.5)); // 1/2

    // Verify normalized quaternion has unit norm
    auto norm_check = norm.norm();
    CHECK(norm_check[0] == doctest::Approx(1.0));
    CHECK(norm_check[1] == doctest::Approx(1.0));
}

TEST_CASE("Quaternion pack interleaved load/store") {
    using quat_t = dp::mat::Quaternion<double>;
    using qpack = pack<quat_t, 4>;

    quat_t data[4] = {{1.0, 0.0, 0.0, 0.0},     // identity
                      {0.707, 0.0, 0.707, 0.0}, // 90 deg Y
                      {0.5, 0.5, 0.5, 0.5},     // some rotation
                      {0.0, 1.0, 0.0, 0.0}};    // 180 deg X

    // Load interleaved
    auto p = qpack::loadu_interleaved(data);
    CHECK(p.w()[0] == doctest::Approx(1.0));
    CHECK(p.x()[0] == doctest::Approx(0.0));
    CHECK(p.w()[1] == doctest::Approx(0.707));
    CHECK(p.y()[1] == doctest::Approx(0.707));
    CHECK(p.w()[2] == doctest::Approx(0.5));
    CHECK(p.w()[3] == doctest::Approx(0.0));
    CHECK(p.x()[3] == doctest::Approx(1.0));

    // Store interleaved
    quat_t output[4];
    p.storeu_interleaved(output);
    CHECK(output[0].w == doctest::Approx(1.0));
    CHECK(output[1].w == doctest::Approx(0.707));
    CHECK(output[1].y == doctest::Approx(0.707));
    CHECK(output[3].x == doctest::Approx(1.0));
}

TEST_CASE("Quaternion pack split load/store") {
    using quat_t = dp::mat::Quaternion<float>;
    using qpack = pack<quat_t, 4>;

    float ws[4] = {1.0f, 0.707f, 0.5f, 0.0f};
    float xs[4] = {0.0f, 0.0f, 0.5f, 1.0f};
    float ys[4] = {0.0f, 0.707f, 0.5f, 0.0f};
    float zs[4] = {0.0f, 0.0f, 0.5f, 0.0f};

    // Load split
    auto p = qpack::loadu_split(ws, xs, ys, zs);
    CHECK(p.w()[0] == doctest::Approx(1.0f));
    CHECK(p.y()[1] == doctest::Approx(0.707f));
    CHECK(p.x()[3] == doctest::Approx(1.0f));

    // Store split
    float out_w[4], out_x[4], out_y[4], out_z[4];
    p.storeu_split(out_w, out_x, out_y, out_z);
    CHECK(out_w[0] == doctest::Approx(1.0f));
    CHECK(out_y[1] == doctest::Approx(0.707f));
    CHECK(out_x[3] == doctest::Approx(1.0f));
}

TEST_CASE("Quaternion pack rotation") {
    using quat_t = dp::mat::Quaternion<double>;
    using qpack = pack<quat_t, 2>;

    // Create 90 degree rotation about Z axis
    // q = cos(45°) + sin(45°)k = (0.707, 0, 0, 0.707)
    const double angle = M_PI / 2.0;
    const double c = std::cos(angle / 2.0);
    const double s = std::sin(angle / 2.0);

    pack<double, 2> w(c), x(0.0), y(0.0), z(s);
    qpack q(w, x, y, z);

    // Rotate vector (1, 0, 0) by 90 degrees about Z
    // Result should be (0, 1, 0)
    pack<double, 2> vx(1.0), vy(0.0), vz(0.0);
    q.rotate_vector(vx, vy, vz);

    CHECK(vx[0] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(vy[0] == doctest::Approx(1.0));
    CHECK(vz[0] == doctest::Approx(0.0).epsilon(1e-10));
}

TEST_CASE("Quaternion pack dot product") {
    using quat_t = dp::mat::Quaternion<double>;
    using qpack = pack<quat_t, 2>;

    // Two identical quaternions have dot product = norm^2
    pack<double, 2> w1(0.5), x1(0.5), y1(0.5), z1(0.5);
    qpack a(w1, x1, y1, z1);

    auto d = a.dot(a);
    CHECK(d[0] == doctest::Approx(1.0)); // 0.25*4 = 1

    // Orthogonal quaternions have dot product = 0
    // i and j are orthogonal
    pack<double, 2> wi(0.0), xi(1.0), yi(0.0), zi(0.0);
    pack<double, 2> wj(0.0), xj(0.0), yj(1.0), zj(0.0);
    qpack qi(wi, xi, yi, zi);
    qpack qj(wj, xj, yj, zj);

    auto d2 = qi.dot(qj);
    CHECK(d2[0] == doctest::Approx(0.0));
}

TEST_CASE("Quaternion pack interpolation") {
    using quat_t = dp::mat::Quaternion<double>;
    using qpack = pack<quat_t, 2>;

    // Interpolate between identity and 90-deg Z rotation
    qpack id = qpack::identity();

    const double angle = M_PI / 2.0;
    pack<double, 2> w(std::cos(angle / 2.0)), x(0.0), y(0.0), z(std::sin(angle / 2.0));
    qpack rot90(w, x, y, z);

    // Lerp at t=0.5
    auto mid = id.lerp(rot90, 0.5);
    // Should be roughly (0.854, 0, 0, 0.354) before normalization
    CHECK(mid.w()[0] > 0.8);
    CHECK(mid.z()[0] > 0.3);

    // Nlerp gives normalized result
    auto mid_norm = id.nlerp(rot90, 0.5);
    auto n = mid_norm.norm();
    CHECK(n[0] == doctest::Approx(1.0));
}

TEST_CASE("Quaternion pack scalar multiplication") {
    using quat_t = dp::mat::Quaternion<float>;
    using qpack = pack<quat_t, 4>;

    pack<float, 4> w(1.0f), x(2.0f), y(3.0f), z(4.0f);
    qpack q(w, x, y, z);

    auto scaled = q * 2.0f;
    CHECK(scaled.w()[0] == doctest::Approx(2.0f));
    CHECK(scaled.x()[0] == doctest::Approx(4.0f));
    CHECK(scaled.y()[0] == doctest::Approx(6.0f));
    CHECK(scaled.z()[0] == doctest::Approx(8.0f));

    // Commutative scalar multiplication
    auto scaled2 = 0.5f * q;
    CHECK(scaled2.w()[0] == doctest::Approx(0.5f));
    CHECK(scaled2.x()[0] == doctest::Approx(1.0f));
}

TEST_CASE("Quaternion pack horizontal sum") {
    using quat_t = dp::mat::Quaternion<double>;
    using qpack = pack<quat_t, 4>;

    pack<double, 4> w, x, y, z;
    w.data_[0] = 1.0;
    w.data_[1] = 2.0;
    w.data_[2] = 3.0;
    w.data_[3] = 4.0;
    x.data_[0] = 0.1;
    x.data_[1] = 0.2;
    x.data_[2] = 0.3;
    x.data_[3] = 0.4;
    y.data_[0] = 0.0;
    y.data_[1] = 0.0;
    y.data_[2] = 0.0;
    y.data_[3] = 0.0;
    z.data_[0] = 0.0;
    z.data_[1] = 0.0;
    z.data_[2] = 0.0;
    z.data_[3] = 0.0;

    qpack q(w, x, y, z);
    auto sum = q.hsum();

    CHECK(sum.w == doctest::Approx(10.0)); // 1+2+3+4
    CHECK(sum.x == doctest::Approx(1.0));  // 0.1+0.2+0.3+0.4
    CHECK(sum.y == doctest::Approx(0.0));
    CHECK(sum.z == doctest::Approx(0.0));
}

// ===== NEW OPERATIONS TESTS =====

TEST_CASE("Quaternion pack from_axis_angle") {
    using quat_t = dp::mat::Quaternion<double>;
    using qpack = pack<quat_t, 4>;

    // Create 90 degree rotations about different axes
    pack<double, 4> ax, ay, az, angle;
    // Rotation about X
    ax.data_[0] = 1.0;
    ay.data_[0] = 0.0;
    az.data_[0] = 0.0;
    angle.data_[0] = M_PI / 2.0;
    // Rotation about Y
    ax.data_[1] = 0.0;
    ay.data_[1] = 1.0;
    az.data_[1] = 0.0;
    angle.data_[1] = M_PI / 2.0;
    // Rotation about Z
    ax.data_[2] = 0.0;
    ay.data_[2] = 0.0;
    az.data_[2] = 1.0;
    angle.data_[2] = M_PI / 2.0;
    // 180 deg about X
    ax.data_[3] = 1.0;
    ay.data_[3] = 0.0;
    az.data_[3] = 0.0;
    angle.data_[3] = M_PI;

    auto q = qpack::from_axis_angle(ax, ay, az, angle);

    // cos(45°) ≈ 0.707
    CHECK(q.w()[0] == doctest::Approx(std::cos(M_PI / 4.0)));
    CHECK(q.x()[0] == doctest::Approx(std::sin(M_PI / 4.0)));
    CHECK(q.y()[0] == doctest::Approx(0.0).epsilon(1e-10));

    CHECK(q.w()[1] == doctest::Approx(std::cos(M_PI / 4.0)));
    CHECK(q.y()[1] == doctest::Approx(std::sin(M_PI / 4.0)));

    CHECK(q.w()[2] == doctest::Approx(std::cos(M_PI / 4.0)));
    CHECK(q.z()[2] == doctest::Approx(std::sin(M_PI / 4.0)));

    // 180 deg about X: w = cos(90) = 0, x = sin(90) = 1
    CHECK(q.w()[3] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(q.x()[3] == doctest::Approx(1.0));
}

TEST_CASE("Quaternion pack from_euler and to_euler") {
    using quat_t = dp::mat::Quaternion<double>;
    using qpack = pack<quat_t, 4>;

    // Create quaternions from Euler angles
    pack<double, 4> roll, pitch, yaw;
    roll.data_[0] = 0.0;
    pitch.data_[0] = 0.0;
    yaw.data_[0] = 0.0; // Identity
    roll.data_[1] = M_PI / 4.0;
    pitch.data_[1] = 0.0;
    yaw.data_[1] = 0.0; // 45 deg roll
    roll.data_[2] = 0.0;
    pitch.data_[2] = M_PI / 4.0;
    yaw.data_[2] = 0.0; // 45 deg pitch
    roll.data_[3] = 0.0;
    pitch.data_[3] = 0.0;
    yaw.data_[3] = M_PI / 4.0; // 45 deg yaw

    auto q = qpack::from_euler(roll, pitch, yaw);

    // Identity quaternion
    CHECK(q.w()[0] == doctest::Approx(1.0));
    CHECK(q.x()[0] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(q.y()[0] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(q.z()[0] == doctest::Approx(0.0).epsilon(1e-10));

    // Convert back to Euler angles
    pack<double, 4> out_roll, out_pitch, out_yaw;
    q.to_euler(out_roll, out_pitch, out_yaw);

    CHECK(out_roll[0] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(out_pitch[0] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(out_yaw[0] == doctest::Approx(0.0).epsilon(1e-10));

    CHECK(out_roll[1] == doctest::Approx(M_PI / 4.0).epsilon(1e-3));
    CHECK(out_pitch[2] == doctest::Approx(M_PI / 4.0).epsilon(1e-3));
    CHECK(out_yaw[3] == doctest::Approx(M_PI / 4.0).epsilon(1e-3));
}

TEST_CASE("Quaternion pack to_rotation_matrix") {
    using quat_t = dp::mat::Quaternion<double>;
    using qpack = pack<quat_t, 2>;

    // Identity quaternion -> identity matrix
    auto id = qpack::identity();
    pack<double, 2> m00, m01, m02, m10, m11, m12, m20, m21, m22;
    id.to_rotation_matrix(m00, m01, m02, m10, m11, m12, m20, m21, m22);

    CHECK(m00[0] == doctest::Approx(1.0));
    CHECK(m11[0] == doctest::Approx(1.0));
    CHECK(m22[0] == doctest::Approx(1.0));
    CHECK(m01[0] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(m10[0] == doctest::Approx(0.0).epsilon(1e-10));

    // 90 deg rotation about Z
    pack<double, 2> w(std::cos(M_PI / 4.0)), x(0.0), y(0.0), z(std::sin(M_PI / 4.0));
    qpack rot_z(w, x, y, z);
    rot_z.to_rotation_matrix(m00, m01, m02, m10, m11, m12, m20, m21, m22);

    // Rotation about Z by 90 deg:
    // [0 -1 0]
    // [1  0 0]
    // [0  0 1]
    CHECK(m00[0] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(m01[0] == doctest::Approx(-1.0));
    CHECK(m10[0] == doctest::Approx(1.0));
    CHECK(m11[0] == doctest::Approx(0.0).epsilon(1e-10));
    CHECK(m22[0] == doctest::Approx(1.0));
}

TEST_CASE("Quaternion pack slerp") {
    using quat_t = dp::mat::Quaternion<double>;
    using qpack = pack<quat_t, 2>;

    // SLERP between identity and 90-deg Z rotation
    auto id = qpack::identity();

    const double angle = M_PI / 2.0;
    pack<double, 2> w(std::cos(angle / 2.0)), x(0.0), y(0.0), z(std::sin(angle / 2.0));
    qpack rot90(w, x, y, z);

    // At t=0, should be identity
    auto s0 = id.slerp(rot90, 0.0);
    CHECK(s0.w()[0] == doctest::Approx(1.0));
    CHECK(s0.z()[0] == doctest::Approx(0.0).epsilon(1e-6));

    // At t=1, should be rot90
    auto s1 = id.slerp(rot90, 1.0);
    CHECK(s1.w()[0] == doctest::Approx(std::cos(angle / 2.0)));
    CHECK(s1.z()[0] == doctest::Approx(std::sin(angle / 2.0)));

    // At t=0.5, should be 45-deg rotation
    auto s05 = id.slerp(rot90, 0.5);
    CHECK(s05.w()[0] == doctest::Approx(std::cos(M_PI / 8.0)).epsilon(1e-3));
    CHECK(s05.z()[0] == doctest::Approx(std::sin(M_PI / 8.0)).epsilon(1e-3));
}

TEST_CASE("Quaternion pack exp and log") {
    using quat_t = dp::mat::Quaternion<double>;
    using qpack = pack<quat_t, 2>;

    // exp(log(q)) should equal q for unit quaternions
    const double angle = M_PI / 3.0;
    pack<double, 2> w(std::cos(angle / 2.0)), x(0.0), y(std::sin(angle / 2.0)), z(0.0);
    qpack q(w, x, y, z);

    auto log_q = q.log();
    // log of unit quaternion should have w ≈ 0
    CHECK(log_q.w()[0] == doctest::Approx(0.0).epsilon(1e-6));

    auto exp_log_q = log_q.qexp();
    CHECK(exp_log_q.w()[0] == doctest::Approx(q.w()[0]).epsilon(1e-6));
    CHECK(exp_log_q.x()[0] == doctest::Approx(q.x()[0]).epsilon(1e-6));
    CHECK(exp_log_q.y()[0] == doctest::Approx(q.y()[0]).epsilon(1e-6));
    CHECK(exp_log_q.z()[0] == doctest::Approx(q.z()[0]).epsilon(1e-6));
}

TEST_CASE("Quaternion pack power") {
    using quat_t = dp::mat::Quaternion<double>;
    using qpack = pack<quat_t, 2>;

    // q^0 = identity
    const double angle = M_PI / 2.0;
    pack<double, 2> w(std::cos(angle / 2.0)), x(0.0), y(0.0), z(std::sin(angle / 2.0));
    qpack q(w, x, y, z);

    auto q0 = q.pow(0.0);
    CHECK(q0.w()[0] == doctest::Approx(1.0).epsilon(1e-6));
    CHECK(q0.z()[0] == doctest::Approx(0.0).epsilon(1e-6));

    // q^1 = q
    auto q1 = q.pow(1.0);
    CHECK(q1.w()[0] == doctest::Approx(q.w()[0]).epsilon(1e-3));
    CHECK(q1.z()[0] == doctest::Approx(q.z()[0]).epsilon(1e-3));

    // q^0.5 should be half the rotation
    auto q05 = q.pow(0.5);
    CHECK(q05.w()[0] == doctest::Approx(std::cos(angle / 4.0)).epsilon(1e-3));
    CHECK(q05.z()[0] == doctest::Approx(std::sin(angle / 4.0)).epsilon(1e-3));
}

TEST_CASE("Quaternion pack difference and angular_difference") {
    using quat_t = dp::mat::Quaternion<double>;
    using qpack = pack<quat_t, 2>;

    auto id = qpack::identity();

    const double angle = M_PI / 2.0;
    pack<double, 2> w(std::cos(angle / 2.0)), x(0.0), y(0.0), z(std::sin(angle / 2.0));
    qpack rot90(w, x, y, z);

    // difference: id^-1 * rot90 = rot90
    auto diff = id.difference(rot90);
    CHECK(diff.w()[0] == doctest::Approx(rot90.w()[0]));
    CHECK(diff.z()[0] == doctest::Approx(rot90.z()[0]));

    // Angular difference (should be the rotation vector)
    pack<double, 2> wx, wy, wz;
    id.angular_difference(rot90, wx, wy, wz);
    // For 90 deg about Z: angular velocity should be (0, 0, pi/2)
    CHECK(wx[0] == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(wy[0] == doctest::Approx(0.0).epsilon(1e-6));
    CHECK(wz[0] == doctest::Approx(angle).epsilon(1e-3));
}

TEST_CASE("Quaternion pack geodesic_distance") {
    using quat_t = dp::mat::Quaternion<double>;
    using qpack = pack<quat_t, 2>;

    auto id = qpack::identity();

    // Distance from identity to itself = 0
    auto d0 = id.geodesic_distance(id);
    CHECK(d0[0] == doctest::Approx(0.0).epsilon(1e-10));

    // Distance from identity to 90 deg rotation
    const double angle = M_PI / 2.0;
    pack<double, 2> w(std::cos(angle / 2.0)), x(0.0), y(0.0), z(std::sin(angle / 2.0));
    qpack rot90(w, x, y, z);

    auto d90 = id.geodesic_distance(rot90);
    // geodesic distance = arccos(|q1·q2|) = arccos(cos(45°)) = 45° = pi/4
    CHECK(d90[0] == doctest::Approx(M_PI / 4.0).epsilon(1e-6));
}
