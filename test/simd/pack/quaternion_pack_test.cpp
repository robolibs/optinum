#include <cmath>
#include <doctest/doctest.h>
#include <optinum/simd/pack/quaternion.hpp>

using optinum::simd::pack;
namespace dp = datapod;

TEST_CASE("Quaternion pack construction") {
    using quat_t = dp::mat::quaternion<float>;
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
    using quat_t = dp::mat::quaternion<double>;
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
    using quat_t = dp::mat::quaternion<double>;
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
    using quat_t = dp::mat::quaternion<float>;
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
    using quat_t = dp::mat::quaternion<double>;
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
    using quat_t = dp::mat::quaternion<double>;
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
    using quat_t = dp::mat::quaternion<float>;
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
    using quat_t = dp::mat::quaternion<double>;
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
    using quat_t = dp::mat::quaternion<double>;
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
    using quat_t = dp::mat::quaternion<double>;
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
    using quat_t = dp::mat::quaternion<float>;
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
    using quat_t = dp::mat::quaternion<double>;
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
