#pragma once

#include <datapod/matrix/vector.hpp>

namespace optinum::opti {

    namespace dp = ::datapod;

    /**
     * Function wrapper interface for optimization
     *
     * This defines the expected interface for objective functions.
     * User functions should provide:
     *   - evaluate(x) -> T
     *   - gradient(x, g) -> void
     *   - evaluate_with_gradient(x, g) -> T (optional, more efficient)
     *
     * Example:
     * @code
     * template <typename T, std::size_t N>
     * struct MyFunction {
     *     using tensor_type = dp::mat::Vector<T, N>;
     *
     *     T evaluate(const tensor_type& x) const {
     *         // Compute f(x)
     *     }
     *
     *     void gradient(const tensor_type& x, tensor_type& g) const {
     *         // Compute gradient into g
     *     }
     *
     *     T evaluate_with_gradient(const tensor_type& x, tensor_type& g) const {
     *         // Compute both f(x) and gradient (more efficient)
     *     }
     * };
     * @endcode
     */
    template <typename FunctionType, typename T, std::size_t N> class FunctionWrapper {
      public:
        using tensor_type = dp::mat::Vector<T, N>;

        /// Constructor
        explicit FunctionWrapper(FunctionType &func) : function(func) {}

        /// Evaluate objective function: f(x)
        T evaluate(const tensor_type &x) const { return function.evaluate(x); }

        /// Compute gradient: ∇f(x)
        void gradient(const tensor_type &x, tensor_type &g) const { function.gradient(x, g); }

        /// Evaluate both objective and gradient (more efficient)
        T evaluate_with_gradient(const tensor_type &x, tensor_type &g) const {
            // Check if function has evaluate_with_gradient method
            if constexpr (requires {
                              { function.evaluate_with_gradient(x, g) } -> std::convertible_to<T>;
                          }) {
                return function.evaluate_with_gradient(x, g);
            } else {
                // Fallback: compute separately
                function.gradient(x, g);
                return function.evaluate(x);
            }
        }

        /// Get reference to underlying function
        FunctionType &get_function() { return function; }
        const FunctionType &get_function() const { return function; }

      private:
        FunctionType &function;
    };

    /**
     * Helper to create function wrapper
     */
    template <typename FunctionType, typename T, std::size_t N> auto make_function_wrapper(FunctionType &func) {
        return FunctionWrapper<FunctionType, T, N>(func);
    }

} // namespace optinum::opti
