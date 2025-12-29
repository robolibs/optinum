#pragma once

// =============================================================================
// optinum/simd/view/quaternion_view.hpp
// quaternion_view<T,W> - Non-owning view over quaternion array with SIMD access
// =============================================================================

#include <datapod/matrix/math/quaternion.hpp>
#include <datapod/spatial/quaternion.hpp>
#include <optinum/simd/pack/quaternion.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

namespace optinum::simd {

    namespace dp = ::datapod;

    // =============================================================================
    // quaternion_view<T, W> - SIMD view over quaternion arrays
    //
    // Provides transparent SIMD access to dp::mat::quaternion<T> arrays.
    // User works with quaternions directly; SIMD is handled internally.
    //
    // Usage:
    //   dp::mat::quaternion<double> quats[8];
    //   auto qv = quaternion_view<double, 4>(quats, 8);
    //   auto result = qv.normalized();  // SIMD under the hood
    //   result.store(quats);            // back to quaternion array
    //
    // Or via bridge:
    //   auto qv = view(quats);  // auto-detect width
    // =============================================================================

    template <typename T, std::size_t W> class quaternion_view {
        static_assert(std::is_floating_point_v<T>, "quaternion_view requires floating-point type");
        static_assert(W > 0, "quaternion_view requires W > 0");

      public:
        using value_type = dp::mat::quaternion<T>;
        using real_type = T;
        using pack_type = pack<value_type, W>;
        using real_pack = pack<T, W>;

        static constexpr std::size_t width = W;

      private:
        value_type *ptr_ = nullptr;
        std::size_t size_ = 0;

      public:
        // ===== CONSTRUCTORS =====

        OPTINUM_INLINE constexpr quaternion_view() noexcept = default;

        OPTINUM_INLINE constexpr quaternion_view(value_type *ptr, std::size_t n) noexcept : ptr_(ptr), size_(n) {}

        // From const pointer (creates const view)
        OPTINUM_INLINE constexpr quaternion_view(const value_type *ptr, std::size_t n) noexcept
            : ptr_(const_cast<value_type *>(ptr)), size_(n) {}

        // ===== SIZE QUERIES =====

        [[nodiscard]] OPTINUM_INLINE constexpr std::size_t size() const noexcept { return size_; }

        [[nodiscard]] OPTINUM_INLINE constexpr std::size_t num_packs() const noexcept { return (size_ + W - 1) / W; }

        [[nodiscard]] OPTINUM_INLINE constexpr std::size_t tail_size() const noexcept {
            const std::size_t rem = size_ % W;
            return (rem == 0) ? W : rem;
        }

        [[nodiscard]] OPTINUM_INLINE constexpr bool empty() const noexcept { return size_ == 0; }

        // ===== ELEMENT ACCESS (scalar) =====

        [[nodiscard]] OPTINUM_INLINE value_type &operator[](std::size_t i) noexcept { return ptr_[i]; }

        [[nodiscard]] OPTINUM_INLINE const value_type &operator[](std::size_t i) const noexcept { return ptr_[i]; }

        [[nodiscard]] OPTINUM_INLINE value_type *data() noexcept { return ptr_; }
        [[nodiscard]] OPTINUM_INLINE const value_type *data() const noexcept { return ptr_; }

        // ===== PACK ACCESS (SIMD) =====

        // Load pack at index (loads W quaternions)
        [[nodiscard]] OPTINUM_INLINE pack_type load_pack(std::size_t pack_idx) const noexcept {
            return pack_type::loadu_interleaved(ptr_ + pack_idx * W);
        }

        // Store pack at index
        OPTINUM_INLINE void store_pack(std::size_t pack_idx, const pack_type &p) noexcept {
            p.storeu_interleaved(ptr_ + pack_idx * W);
        }

        // Tail-safe load (handles partial packs at end)
        [[nodiscard]] OPTINUM_INLINE pack_type load_pack_safe(std::size_t pack_idx) const noexcept {
            const std::size_t start = pack_idx * W;
            const std::size_t valid = (pack_idx == num_packs() - 1) ? tail_size() : W;

            if (valid == W) {
                return pack_type::loadu_interleaved(ptr_ + start);
            }

            // Partial load - load valid elements, zero the rest
            alignas(32) T w_vals[W] = {};
            alignas(32) T x_vals[W] = {};
            alignas(32) T y_vals[W] = {};
            alignas(32) T z_vals[W] = {};

            for (std::size_t i = 0; i < valid; ++i) {
                w_vals[i] = ptr_[start + i].w;
                x_vals[i] = ptr_[start + i].x;
                y_vals[i] = ptr_[start + i].y;
                z_vals[i] = ptr_[start + i].z;
            }

            return pack_type::loadu_split(w_vals, x_vals, y_vals, z_vals);
        }

        // Tail-safe store
        OPTINUM_INLINE void store_pack_safe(std::size_t pack_idx, const pack_type &p) noexcept {
            const std::size_t start = pack_idx * W;
            const std::size_t valid = (pack_idx == num_packs() - 1) ? tail_size() : W;

            if (valid == W) {
                p.storeu_interleaved(ptr_ + start);
                return;
            }

            // Partial store
            alignas(32) T w_vals[W];
            alignas(32) T x_vals[W];
            alignas(32) T y_vals[W];
            alignas(32) T z_vals[W];

            p.storeu_split(w_vals, x_vals, y_vals, z_vals);

            for (std::size_t i = 0; i < valid; ++i) {
                ptr_[start + i].w = w_vals[i];
                ptr_[start + i].x = x_vals[i];
                ptr_[start + i].y = y_vals[i];
                ptr_[start + i].z = z_vals[i];
            }
        }

        // ===== FILL OPERATIONS =====

        // Fill all elements with a value
        void fill(const value_type &val) noexcept {
            pack_type p(val);
            for (std::size_t i = 0; i < num_packs(); ++i) {
                store_pack_safe(i, p);
            }
        }

        // Fill with identity quaternion
        void fill_identity() noexcept { fill(value_type::identity()); }

        // ===== BULK OPERATIONS (return new view with results) =====
        // These operate on the entire array using SIMD

        // Conjugate all quaternions
        [[nodiscard]] quaternion_view conjugate_to(value_type *out) const noexcept {
            quaternion_view result(out, size_);
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto p = load_pack_safe(i);
                result.store_pack_safe(i, p.conjugate());
            }
            return result;
        }

        // Normalize all quaternions
        [[nodiscard]] quaternion_view normalized_to(value_type *out) const noexcept {
            quaternion_view result(out, size_);
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto p = load_pack_safe(i);
                result.store_pack_safe(i, p.normalized());
            }
            return result;
        }

        // Inverse all quaternions
        [[nodiscard]] quaternion_view inverse_to(value_type *out) const noexcept {
            quaternion_view result(out, size_);
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto p = load_pack_safe(i);
                result.store_pack_safe(i, p.inverse());
            }
            return result;
        }

        // ===== IN-PLACE OPERATIONS =====

        // Conjugate in place
        void conjugate_inplace() noexcept {
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto p = load_pack_safe(i);
                store_pack_safe(i, p.conjugate());
            }
        }

        // Normalize in place
        void normalize_inplace() noexcept {
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto p = load_pack_safe(i);
                store_pack_safe(i, p.normalized());
            }
        }

        // Inverse in place
        void inverse_inplace() noexcept {
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto p = load_pack_safe(i);
                store_pack_safe(i, p.inverse());
            }
        }

        // ===== BINARY OPERATIONS =====

        // Hamilton product: this * other -> out
        [[nodiscard]] quaternion_view multiply_to(const quaternion_view &other, value_type *out) const noexcept {
            quaternion_view result(out, size_);
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto a = load_pack_safe(i);
                auto b = other.load_pack_safe(i);
                result.store_pack_safe(i, a * b);
            }
            return result;
        }

        // SLERP interpolation: slerp(this, other, t) -> out
        [[nodiscard]] quaternion_view slerp_to(const quaternion_view &other, T t, value_type *out) const noexcept {
            quaternion_view result(out, size_);
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto a = load_pack_safe(i);
                auto b = other.load_pack_safe(i);
                result.store_pack_safe(i, a.slerp(b, t));
            }
            return result;
        }

        // NLERP interpolation: nlerp(this, other, t) -> out
        [[nodiscard]] quaternion_view nlerp_to(const quaternion_view &other, T t, value_type *out) const noexcept {
            quaternion_view result(out, size_);
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto a = load_pack_safe(i);
                auto b = other.load_pack_safe(i);
                result.store_pack_safe(i, a.nlerp(b, t));
            }
            return result;
        }

        // ===== REDUCTION OPERATIONS =====

        // Compute norms of all quaternions -> output array
        void norms_to(T *out) const noexcept {
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto p = load_pack_safe(i);
                auto n = p.norm();

                alignas(32) T vals[W];
                n.storeu(vals);

                const std::size_t start = i * W;
                const std::size_t valid = (i == num_packs() - 1) ? tail_size() : W;
                for (std::size_t j = 0; j < valid; ++j) {
                    out[start + j] = vals[j];
                }
            }
        }

        // Compute dot products with another view -> output array
        void dot_to(const quaternion_view &other, T *out) const noexcept {
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto a = load_pack_safe(i);
                auto b = other.load_pack_safe(i);
                auto d = a.dot(b);

                alignas(32) T vals[W];
                d.storeu(vals);

                const std::size_t start = i * W;
                const std::size_t valid = (i == num_packs() - 1) ? tail_size() : W;
                for (std::size_t j = 0; j < valid; ++j) {
                    out[start + j] = vals[j];
                }
            }
        }

        // ===== ROTATION OPERATIONS =====

        // Rotate vectors (vx, vy, vz arrays) by these quaternions
        void rotate_vectors(T *vx, T *vy, T *vz) const noexcept {
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto q = load_pack_safe(i);

                const std::size_t start = i * W;
                const std::size_t valid = (i == num_packs() - 1) ? tail_size() : W;

                // Load vectors
                alignas(32) T vx_vals[W] = {};
                alignas(32) T vy_vals[W] = {};
                alignas(32) T vz_vals[W] = {};

                for (std::size_t j = 0; j < valid; ++j) {
                    vx_vals[j] = vx[start + j];
                    vy_vals[j] = vy[start + j];
                    vz_vals[j] = vz[start + j];
                }

                real_pack px = real_pack::loadu(vx_vals);
                real_pack py = real_pack::loadu(vy_vals);
                real_pack pz = real_pack::loadu(vz_vals);

                q.rotate_vector(px, py, pz);

                px.storeu(vx_vals);
                py.storeu(vy_vals);
                pz.storeu(vz_vals);

                for (std::size_t j = 0; j < valid; ++j) {
                    vx[start + j] = vx_vals[j];
                    vy[start + j] = vy_vals[j];
                    vz[start + j] = vz_vals[j];
                }
            }
        }

        // ===== CONVERSION OPERATIONS =====

        // Convert all to Euler angles
        void to_euler(T *roll, T *pitch, T *yaw) const noexcept {
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto q = load_pack_safe(i);

                real_pack r, p, y;
                q.to_euler(r, p, y);

                alignas(32) T r_vals[W], p_vals[W], y_vals[W];
                r.storeu(r_vals);
                p.storeu(p_vals);
                y.storeu(y_vals);

                const std::size_t start = i * W;
                const std::size_t valid = (i == num_packs() - 1) ? tail_size() : W;
                for (std::size_t j = 0; j < valid; ++j) {
                    roll[start + j] = r_vals[j];
                    pitch[start + j] = p_vals[j];
                    yaw[start + j] = y_vals[j];
                }
            }
        }

        // Create from Euler angles
        static quaternion_view from_euler(const T *roll, const T *pitch, const T *yaw, value_type *out,
                                          std::size_t n) noexcept {
            quaternion_view result(out, n);
            const std::size_t num_packs = (n + W - 1) / W;

            for (std::size_t i = 0; i < num_packs; ++i) {
                const std::size_t start = i * W;
                const std::size_t valid = (i == num_packs - 1) ? ((n % W == 0) ? W : n % W) : W;

                alignas(32) T r_vals[W] = {};
                alignas(32) T p_vals[W] = {};
                alignas(32) T y_vals[W] = {};

                for (std::size_t j = 0; j < valid; ++j) {
                    r_vals[j] = roll[start + j];
                    p_vals[j] = pitch[start + j];
                    y_vals[j] = yaw[start + j];
                }

                real_pack r = real_pack::loadu(r_vals);
                real_pack p = real_pack::loadu(p_vals);
                real_pack y = real_pack::loadu(y_vals);

                auto q = pack_type::from_euler(r, p, y);
                result.store_pack_safe(i, q);
            }

            return result;
        }

        // Convert all to axis-angle
        void to_axis_angle(T *ax, T *ay, T *az, T *angle) const noexcept {
            for (std::size_t i = 0; i < num_packs(); ++i) {
                auto q = load_pack_safe(i);

                real_pack pax, pay, paz, pangle;
                q.to_axis_angle(pax, pay, paz, pangle);

                alignas(32) T ax_vals[W], ay_vals[W], az_vals[W], angle_vals[W];
                pax.storeu(ax_vals);
                pay.storeu(ay_vals);
                paz.storeu(az_vals);
                pangle.storeu(angle_vals);

                const std::size_t start = i * W;
                const std::size_t valid = (i == num_packs() - 1) ? tail_size() : W;
                for (std::size_t j = 0; j < valid; ++j) {
                    ax[start + j] = ax_vals[j];
                    ay[start + j] = ay_vals[j];
                    az[start + j] = az_vals[j];
                    angle[start + j] = angle_vals[j];
                }
            }
        }

        // ===== SUBVIEW =====

        [[nodiscard]] quaternion_view subview(std::size_t offset, std::size_t count) const noexcept {
            return quaternion_view(ptr_ + offset, count);
        }

        // ===== ITERATORS (for range-based for) =====

        value_type *begin() noexcept { return ptr_; }
        value_type *end() noexcept { return ptr_ + size_; }
        const value_type *begin() const noexcept { return ptr_; }
        const value_type *end() const noexcept { return ptr_ + size_; }
    };

    // ===== TYPE ALIASES =====

    template <std::size_t W> using quaternion_viewf = quaternion_view<float, W>;
    template <std::size_t W> using quaternion_viewd = quaternion_view<double, W>;

} // namespace optinum::simd
