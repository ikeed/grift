import itertools
import logging
from typing import Dict, List

import mpmath as mp
import numpy as np
import pandas as pd
from pandas import Series
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value, PULP_CBC_CMD
from scipy.optimize import minimize, OptimizeResult

precision_digits = 300
mp.mp.dps = precision_digits  # Set decimal precision globally for mpmath
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
verbose = logger.isEnabledFor(logging.DEBUG)

def apply_row_scaling(M: mp.matrix, b: mp.matrix):
    """Apply row scaling to M and b based on 1 / sqrt(abs(b[i])) for each row i,
       using a scaling factor of 1 if b[i] is zero."""
    row_scaling_factors = [
        1 / mp.sqrt(abs(b[i])) if b[i] != 0 else 1
        for i in range(len(b))
    ]
    row_scaling_matrix = mp.diag(row_scaling_factors)
    scaled_M = row_scaling_matrix * M
    scaled_b = row_scaling_matrix * b
    return scaled_M, scaled_b


def geometric_mean_of_nonzero_elements(numbers: list[mp.mpf]):
    p = mp.mpf(1)
    for x in numbers:
        if x == 0:
            p *= mp.mpf(1)
        else:
            p *= x
    return mp.power(p, len(numbers))


def apply_column_scaling(scaled_M: mp.matrix):
    """Apply column scaling to scaled_M based on the maximum log-fab value in each column and return the scaled matrix and the inverse scaling matrix."""
    column_scaling_factors = []
    for j in range(scaled_M.cols):
        # Find the element in column j with the largest fabs(ln(fabs(scaled_M[i, j])))
        max_log_fabs = max((mp.fabs(mp.ln(mp.fabs(scaled_M[i, j]))) for i in range(scaled_M.rows)), default=1)

        # Find the entry with that max_log_fabs value
        max_entry = next(scaled_M[i, j] for i in range(scaled_M.rows)
                         if mp.fabs(mp.ln(mp.fabs(scaled_M[i, j]))) == max_log_fabs)

        # Calculate the scaling factor as 1 / sqrt(fabs(max_entry))
        scale_factor = 1 / mp.sqrt(mp.fabs(max_entry)) if max_entry != 0 else 1
        column_scaling_factors.append(scale_factor)

    # Construct the column scaling and inverse scaling matrices
    column_scaling_matrix = mp.diag(column_scaling_factors)
    inverse_column_scaling_matrix = mp.diag([1 / factor if factor != 0 else 1 for factor in column_scaling_factors])

    # Apply the scaling to the matrix
    return scaled_M * column_scaling_matrix, inverse_column_scaling_matrix


def rescale_system(M: mp.matrix, b: mp.matrix, use_row_scaling: bool = False, use_col_scaling: bool = False):
    # Step 1: Apply row scaling
    if use_row_scaling:
        row_scaled_M, row_scaled_b = apply_row_scaling(M, b)
    else:
        row_scaled_M, row_scaled_b = M, b

    if use_col_scaling:
        # Step 2: Apply column scaling
        fully_scaled_M, inverse_column_scaling_matrix = apply_column_scaling(row_scaled_M)
    else:
        fully_scaled_M, inverse_column_scaling_matrix = row_scaled_M, mp.eye(M.cols)

    # Return fully scaled matrix, scaled vector b, and inverse column scaling matrix
    return fully_scaled_M, row_scaled_b, inverse_column_scaling_matrix


class DecoupleService:
    def __init__(self, pair_columns: List[str], algo: str = "trust-krylov", use_row_scaling: bool = False,
                 use_col_scaling: bool = False, use_recursive_reduction: bool = False):
        """
        Initialize the decoupling service for paired forex data.

        The main goal of this service is to decouple exchange rates into individual currency
        strengths, which can then be modeled as a closed system, making their behavior easier
        to predict or analyze. This approach assumes that the individual currencies are more
        predictable than the currency pairs themselves.

        Parameters:
        - pair_columns: List of currency pairs (e.g., ['CADUSD', 'USDGBP']). Each pair represents
          an observed exchange rate between two currencies.
        """
        self._pair_columns = sorted(pair_columns)  # List of currency pairs (e.g., ['CADUSD', 'USDGBP', ...])
        self._currencies = sorted(self._derive_currencies())  # Derive the list of unique currencies
        self._algo = algo
        self._use_row_scaling = use_row_scaling
        self._use_col_scaling = use_col_scaling
        self._use_recursive_reduction = use_recursive_reduction
        matrix = self._build_matrix()  # Build matrix M once during instantiation

        balance_factors: mp.matrix = self._balance_matrix(matrix)
        matrix = balance_factors @ matrix
        self._positional_matrix = mp.matrix([row for i, row in enumerate(matrix.tolist())])
        self._constant_column = balance_factors @ mp.ones(balance_factors.cols, 1)
        self._initial_guess = np.ones(len(self._currencies), dtype=np.float64) / len(self._currencies)
        self._minimize_methods = {
            'bfgs': self.minimize_bfgs,
            'newton-cg': self.minimize_newton_cg,
            'l-bgfs-b': self.minimize_lbfgsb,
            'trust-ncg': self.minimize_trust_ncg,
            'trust-exact': self.minimize_trust_exact,
            'trust-krylov': self.minimize_trust_krylov,
            'slsqp': self.minimize_slsqp
        }

    @staticmethod
    def _balance_matrix(mat: mp.matrix) -> mp.matrix:
        m = mat.rows
        n = mat.cols

        # Create a linear programming problem
        problem = LpProblem("Maximize_Rows_Used", LpMaximize)

        # Decision variables (z_i^1, z_i^(-1), and z_i^0)
        z_pos = [LpVariable(f"z_pos_{i}", cat='Binary') for i in range(m)]
        z_neg = [LpVariable(f"z_neg_{i}", cat='Binary') for i in range(m)]

        # Constraint 1: z_pos + z_neg <= 1 for each row (this ensures that z^0 is automatically modeled)
        for i in range(m):
            problem += z_pos[i] + z_neg[i] <= 1

        # Objective function: Maximize the number of non-zero rows (sum of |x_i|)
        problem += lpSum([z_pos[i] + z_neg[i] for i in range(m)])

        # Constraint 2: Ensure the column sums are zero
        for j in range(n):
            problem += lpSum([mat[i, j] * (z_pos[i] - z_neg[i]) for i in range(m)]) == 0

        # Solve the problem
        problem.solve(PULP_CBC_CMD(msg=False))

        # Extract the solution: x = [-1, 0, 1] vector
        x = mp.matrix([mp.mpf(value(z_pos[i])) - mp.mpf(value(z_neg[i])) for i in range(m)])
        if sum(x) < 0:
            x = -x
        x = [1 if v == 0 else v for v in x]
        return mp.diag(x)

    @staticmethod
    def _parse_pair(pair_symbol: str) -> (str, str):
        """
        Extracts the base and quote currencies from a currency pair symbol.
        This is a helper method to assist in identifying which currencies are involved in a pair.
        """
        return pair_symbol[:3], pair_symbol[-3:]

    def _derive_currencies(self) -> List[str]:
        """
        Derives a unique list of currencies from the currency pairs provided.
        This ensures that the matrix M can be constructed with all necessary currencies.

        This method scans through all pair symbols (e.g., 'CADUSD') and extracts individual currencies
        (e.g., 'CAD' and 'USD'). The resulting list of currencies is used to build the decoupling matrix.
        """
        return sorted(
            set(itertools.chain.from_iterable(DecoupleService._parse_pair(pair) for pair in self._pair_columns)))

    def _build_matrix(self) -> mp.matrix:
        """
        Constructs the matrix decouple_matrix that represents the system of currency pairs.

        Each row in decouple_matrix corresponds to one of the currency pairs. The values in the row will be
        -1, 0, or 1 depending on how the base and quote currencies relate to the logarithms
        of the individual currencies.

        Example:
        ln(CADUSD) = 1 * ln(CAD) + -1 * ln(USD)
        decouple_matrix will then have a row [1, -1] corresponding to ln(CADUSD).
        """
        num_currencies = len(self._currencies)
        num_pairs = len(self._pair_columns)

        decouple_matrix = mp.zeros(num_pairs, num_currencies)
        for row_idx, pair in enumerate(self._pair_columns):
            base_currency, quote_currency = self._parse_pair(pair)
            base_idx = self._currencies.index(base_currency)
            quote_idx = self._currencies.index(quote_currency)
            decouple_matrix[row_idx, base_idx] = 1
            decouple_matrix[row_idx, quote_idx] = -1
        return decouple_matrix

    def decouple(self, row: Dict[str, mp.mpf]) -> Series:
        """
        Decouples a row of observed currency pair rates into individual currency strengths.

        This function solves for v in the equation Mv = b, where M is the matrix of currency pairs,
        and b is a vector containing the logarithms of the observed exchange rates. The solution v
        contains the logarithms of individual currency strengths.

        After solving the system, the results are re-centered to ensure numerical stability,
        meaning that the values in v will sum to zero while retaining the correct relationships.

        Parameters:
        - row: A dictionary representing observed currency pair rates (e.g., {'CADUSD': 123.4, 'USDGBP': 32.6}).

        Returns:
        - A dictionary where keys are individual currencies and values are their proportional strengths
          (in logarithmic form). The results are re-centered for stability.
        """

        # Scale the matrix
        matrix = self._positional_matrix

        # Scale the constant column (b)
        b = mp.diag([mp.ln(row[pair]) for pair in self._pair_columns]) @ self._constant_column

        try:
            if self._use_recursive_reduction:
                result_vector, arbitrage = self.recursive_minimize_linear_system(algo=self._algo, matrix_in=matrix,
                                                                                 b_in=b,
                                                                      use_row_scaling=self._use_row_scaling,
                                                                      use_col_scaling=self._use_col_scaling)
            else:
                result_vector, arbitrage = self.minimize_linear_system(self._algo, matrix, b, self._use_row_scaling,
                                                            self._use_col_scaling)

            result_vector = result_vector - mp.fdiv(mp.fsum(result_vector), len(result_vector))
        except Exception as e:
            logger.debug(f'exception during minimization: {e}')
            raise e

        self._initial_guess = np.array(result_vector).astype(np.float64)
        # Return the individual currency strengths, normalized and re-centered
        return pd.concat([pd.Series({self._currencies[i]: result_vector[i] for i in range(len(result_vector))}),
                          pd.Series({'arbitrage': arbitrage})])

    def minimize_linear_system(self, algo: str, matrix_in: mp.matrix, b_in: mp.matrix, use_row_scaling: bool = False,
                               use_col_scaling: bool = False) -> (mp.matrix, float):
        # rescale rows by 1/sqrt(abs(b[i]))
        matrix, b, denormalizer = rescale_system(matrix_in, b_in, use_row_scaling, use_col_scaling)
        matrix_transpose = matrix.transpose()
        matrix_transpose_matrix = matrix_transpose * matrix
        zeros_matrix = mp.zeros(matrix.cols, matrix.cols)
        zeros_column = mp.zeros(matrix.cols, 1)

        def objective(current_vec):
            """
            Objective function:
            ||M * current_vec - b|| (unsquared L2 norm)

            Parameters:
            - current_vec: The vector of parameters (numpy array), including v_extra as the last element.

            Returns:
            - The objective value as a float.
            """
            # Convert the numpy vector to mpmath for high-precision operations
            current_vec = mp.matrix([mp.mpf(x) for x in current_vec])  # Convert numpy array to mpmath matrix

            # Perform high-precision matrix multiplication using mpmath
            b_hat = matrix @ current_vec
            difference = (b_hat) - b

            # Compute the unsquared L2 norm using mpmath for the difference (residual)
            norm_l2 = mp.sqrt(mp.fsum([x ** 2 for x in difference]))

            return np.float64(norm_l2)

        def jacobian(current_vec):
            """
            Compute the Jacobian of the unsquared L2 objective function using mpmath.

            Parameters:
            - current_vec: The vector of parameters (numpy array).

            Returns:
            - The Jacobian as a numpy array.
            """
            # Convert current_vec to an mpmath matrix
            current_vec = mp.matrix([mp.mpf(x) for x in current_vec])

            # Calculate the residuals
            residual = matrix @ current_vec - b

            # Compute the gradient: M.T * residual / ||Mv - b||
            norm_l2 = mp.sqrt(mp.fsum([x ** 2 for x in residual]))
            if norm_l2 == 0:  # Avoid division by zero
                gradient = zeros_column
            else:
                gradient = matrix_transpose * residual / norm_l2

            # Convert the mpmath gradient back to a numpy array for further processing
            return np.array(gradient.tolist(), dtype=np.float64).flatten()

        def hess(current_vec):
            """
            Compute the simplified Hessian of the unsquared L2 objective function using mpmath.

            Parameters:
            - current_vec: The vector of parameters (numpy array).

            Returns:
            - The Hessian as a numpy array.
            """
            # Convert current_vec to an mpmath matrix
            current_vec = mp.matrix([mp.mpf(x) for x in current_vec])

            # Calculate the residuals
            residual = matrix @ current_vec - b

            # Compute ||Mv - b|| (L2 norm)
            norm_l2 = mp.sqrt(mp.fsum([x ** 2 for x in residual]))
            if norm_l2 == 0:  # Avoid division by zero in Hessian terms
                hessian = zeros_matrix
            else:
                # Compute only the primary term of the Hessian: (M.T * M) / ||Mv - b||
                hessian = matrix_transpose_matrix / norm_l2

            # Convert the resulting Hessian to a numpy array for further processing
            return np.array(hessian.tolist(), dtype=np.float64)

        # Minimize the objective function using scipy's BFGS method
        result = self.minimize(algo, hess, jacobian, objective)

        # Handle optimization result
        if result.success is False:
            test_code = generate_unit_test_code(matrix=matrix_in, b=b_in, algo=algo, initial_guess=self._initial_guess,
                                                pair_columns=self._pair_columns)
            # print(test_code)
            logger.warning(
                f'----------\n\n{algo}: {result.message or "Unsuccessful decoupling"}\n{result}\n\n----------\n')

        arbitrage = result.fun / b_in.rows
        # print(f"Result found in {result.nit} iterations, per-component MSE: {arbitrage}")
        return denormalizer @ mp.matrix(result.x), float(result.fun)

    def recursive_minimize_linear_system(self, algo: str, matrix_in: mp.matrix, b_in: mp.matrix,
                                         tolerance: float = 1e-15, max_iterations: int = 100,
                                         use_row_scaling: bool = False, use_col_scaling: bool = False) -> (
    mp.matrix, float):
        # First minimization step
        v_approx, residual_fun = self.minimize_linear_system(algo=algo, matrix_in=matrix_in, b_in=b_in,
                                                             use_row_scaling=use_row_scaling,
                                                             use_col_scaling=use_col_scaling)

        # Compute initial residual error
        b_hat = matrix_in @ v_approx
        epsilon = b_hat - b_in
        prev_epsilon = epsilon + epsilon
        iteration = 0
        while tolerance <= mp.norm(epsilon) <= mp.norm(prev_epsilon) and iteration < max_iterations:
            prev_epsilon = epsilon
            iteration += 1
            print(f"Iteration {iteration}: Residual norm = {mp.nstr(mp.norm(epsilon), 8, strip_zeros=False)}")

            # Solve Mw = epsilon
            w_correction, w_residual = self.minimize_linear_system(algo=algo, matrix_in=matrix_in, b_in=epsilon,
                                                       use_row_scaling=use_row_scaling, use_col_scaling=False)
            # Recompute epsilon for the updated solution
            epsilon = (matrix_in @ (v_approx - w_correction)) - b_in
            if mp.norm(epsilon) < mp.norm(prev_epsilon):
                # Update the solution
                v_approx -= w_correction
            else:
                # well we tried, but this iteration didn't improve our solution.
                epsilon = prev_epsilon
                break

        logger.debug(
            f"Converged in {iteration} iterations with residual norm = {mp.nstr(mp.norm(epsilon), 8, strip_zeros=False)}")

        return v_approx, float(mp.norm(epsilon))

    def minimize(self, method, hessian, jacobian, objective) -> OptimizeResult:
        # logger.debug(f'minimizing with {method}')
        method = method.lower()
        methods = self._minimize_methods
        # Call the appropriate function, raising an error if the method is not supported
        if method in methods:
            decoupled = methods[method](hessian, jacobian, objective)
            return decoupled
        else:
            raise ValueError(f"Unsupported minimization method: {method}")

    def minimize_bfgs(self, hessian, jacobian, objective):
        """Run optimization using the BFGS method."""
        # OOPS Should be xrtol
        # {'maxiter': 1000, 'gtol': 1e-7, 'disp': verbose} = FAIL
        # {'maxiter': 1000, 'gtol': 1e-6, 'disp': verbose} = FAIL
        # {'maxiter': 1000, 'gtol': 125e-8, 'xrtol': 1e-17, 'disp': verbose} = FAIL
        # {'maxiter': 1000, 'gtol': 1.5e-6, 'xrtol': 1e-17, 'disp': verbose} = {"BFGS": {"mean": 0.820414925371664, "stddev": 1.1861317270158815, "mean/stddev": 0.6916726925732753}}
        # {'maxiter': 1000, 'gtol': 1.5e-6, 'xrtol': 1e-16, 'disp': verbose} = {"BFGS": {"mean": 0.8204149253727349, "stddev": 1.186131727015467, "mean/stddev": 0.6916726925744199}}
        # {'maxiter': 1000, 'gtol': 1.5e-6, 'xrtol': 1e-10, 'disp': verbose} = {"BFGS": {"mean": 0.8204149254159213, "stddev": 1.186131727001019, "mean/stddev": 0.6916726926192545}}
        # {'maxiter': 1000, 'gtol': 2e-6, 'disp': verbose} = {"BFGS": {"mean": 0.8204149253948145, "stddev": 1.1861317268001612, "mean/stddev": 0.6916726927185867}}
        # {'maxiter': 1000, 'gtol': 3e-6, 'disp': verbose} = {"BFGS": {"mean": 0.8204149268202027, "stddev": 1.1861317271638752, "mean/stddev": 0.6916726937082045}}
        # {'maxiter': 1000, 'gtol': 5e-6, 'disp': verbose} = {"BFGS": {"mean": 0.820414925987836, "stddev": 1.1861317273697363, "mean/stddev": 0.6916726928864112}}
        # {'maxiter': 1000, 'gtol': 5e-6, 'disp': verbose} = {"BFGS": {"mean": 0.820414925987836, "stddev": 1.1861317273697363, "mean/stddev": 0.6916726928864112}}
        # {'maxiter': 1000, 'gtol': 5e-6, 'disp': verbose} = {"BFGS": {"mean": 0.8204149259878538, "stddev": 1.1861317273697105, "mean/stddev": 0.6916726928864412}}
        # {'maxiter': 1000, 'gtol': 1e-5, 'disp': verbose} = {"BFGS": {"mean": 0.8204149262324251, "stddev": 1.186131726970222, "mean/stddev": 0.6916726933255885}}
        options = {'maxiter': 1000, 'gtol': 1.5e-6, 'xrtol': 1e-17, 'disp': verbose}
        return self.do_minimize("BFGS", hessian=hessian, jacobian=jacobian, objective=objective, options=options)

    def minimize_trust_ncg(self, hessian, jacobian, objective):
        """Run optimization using the trust-ncg method."""
        # xrtol not supported!!
        # {'maxiter': 1000, 'gtol': 1e-8} = FAIL
        # {'maxiter': 1000, 'gtol': 3e-8} = FAIL
        # {'maxiter': 1000, 'gtol': 3.5e-8} = FAIL
        # {'maxiter': 1000, 'gtol': 4e-8} = {"BFGS": {"mean": 0.8204149252711201, "stddev": 1.1861317270885767, "mean/stddev": 0.6916726924461182}}
        # {'maxiter': 1000, 'gtol': 6e-8} = {"trust-ncg": {"mean": 0.820414925243935, "stddev": 1.186131726232778, "mean/stddev": 0.6916726929222436}}
        # {'maxiter': 1000, 'gtol': 1e-7} = {"trust-ncg": {"mean": 0.8204149252391741, "stddev": 1.1861317262266162, "mean/stddev": 0.691672692921823}}
        # {'maxiter': 1000, 'gtol': 1e-6} = {"trust-ncg": {"mean": 0.8204149252983961, "stddev": 1.1861317263955526, "mean/stddev": 0.6916726928732392}}
        # {'maxiter': 1000, 'gtol': 5e-5} = {"trust-ncg": {"mean": 0.8204149274855858, "stddev": 1.1861316992573896, "mean/stddev": 0.6916727105423699},
        options = {'maxiter': 1000, 'gtol': 5e-8}
        return self.do_minimize("trust-ncg", hessian=hessian, jacobian=jacobian, objective=objective, options=options)

    def minimize_newton_cg(self, hessian, jacobian, objective):
        """Run optimization using the Newton-CG method."""
        # xtol=1e-12: {"Newton-CG": {"mean": 0.4283522074984076, "stddev": 0.6102659335623968}}
        # xtol=1e-11: {"Newton-CG": {"mean": 0.4283522074982811, "stddev": 0.6102659335623732}}
        #   gtol=1e-3: {"Newton-CG": {"mean": 0.4283522074982811, "stddev": 0.6102659335623732}}
        #   gtol=1e-22: {"Newton-CG": {"mean": 0.4283522074982811, "stddev": 0.6102659335623732}}
        # xtol=1e-10: {"Newton-CG": {"mean": 0.4283522075052016, "stddev": 0.6102659335593438}}
        # xtol=1e-9: {"Newton-CG": {"mean": 0.4283522078599851, "stddev": 0.6102659335621407}}
        # xtol=1e-8: {"Newton-CG": {"mean": 0.42835220379435357, "stddev": 0.6102659401879389}}
        # xtol=1e-5: {"Newton-CG": {"mean": 0.4285592341304815, "stddev": 0.6096977867918844}}
        options = {
            'xtol': 1e-6,  # Tighter convergence on solution
            'maxiter': 1000,  # Increase max iterations
        }
        return self.do_minimize("Newton-CG", hessian=hessian, jacobian=jacobian, objective=objective, options=options)

    def minimize_lbfgsb(self, hessian, jacobian, objective):
        """Run optimization using the L-BFGS-B method."""
        # ftol: 1e-24: {"L-BFGS-B": {"mean": 0.8204148309496204, "stddev": 1.1861316417349619, "mean/stddev": 0.6916726626983787}}
        # ftol: 1e-20: {"L-BFGS-B": {"mean": 0.8204148309496204, "stddev": 1.1861316417349619, "mean/stddev": 0.6916726626983787}}
        # ftol: 1e-16: {"L-BFGS-B": {"mean": 0.8204148309496204, "stddev": 1.1861316417349619, "mean/stddev": 0.6916726626983787}}
        # ftol: 1e-14: {"L-BFGS-B": {"mean": 0.8204148311766897, "stddev": 1.1861316411703784, "mean/stddev": 0.6916726632190429}}
        # ftol: 1e-13: {"L-BFGS-B": {"mean": 0.8204149886621578, "stddev": 1.186131775727768, "mean/stddev": 0.6916727175264995}}
        # ftol: 1e-12: {"L-BFGS-B": {"mean": 0.8204153455136204, "stddev": 1.1861333637108833, "mean/stddev": 0.6916720923749299}}
        # ftol: 1e-11: {"L-BFGS-B": {"mean": 0.820416897662648, "stddev": 1.1861243448125087, "mean/stddev": 0.6916786602102258}}
        # ftol: 1e-10: {"L-BFGS-B": {"mean": 0.82039104844114, "stddev": 1.1860298404037917, "mean/stddev": 0.6917119793224026}}
        # ftol: 1e-9: {"L-BFGS-B": {"mean": 0.8202853368849248, "stddev": 1.1856220886978157, "mean/stddev": 0.6918607073067059}}
        # ftol: 5e-9: {"L-BFGS-B": {"mean": 0.8202560915895206, "stddev": 1.1855474312850867, "mean/stddev": 0.6918796076344202}}
        # ftol: 1e-8: {"L-BFGS-B": {"mean": 0.820239835399687, "stddev": 1.1855675704442605, "mean/stddev": 0.6918541429842953}}
        # ftol: 3e-8: {"L-BFGS-B": {"mean": 0.820242954498278, "stddev": 1.1855980705239975, "mean/stddev": 0.6918389755271415}}
        # ftol: 5e-8: {"L-BFGS-B": {"mean": 0.8202538042162698, "stddev": 1.1855489209798613, "mean/stddev": 0.6918768088779723}}
        # ftol: 1e-7: {"L-BFGS-B": {"mean": 0.8202574540599656, "stddev": 1.1855917992308163, "mean/stddev": 0.6918548648802473}}
        # ftol: 1e-3: {"L-BFGS-B": {"mean": 0.8277198052333458, "stddev": 1.1835364124884953, "mean/stddev": 0.6993615037943682}}
        options = {'maxiter': 1000, 'ftol': 1e-16, 'eps': 1e-8}
        return self.do_minimize("l-bfgs-b", hessian=None, jacobian=jacobian, objective=objective, options=options)

    def minimize_trust_exact(self, hessian, jacobian, objective):
        """Run optimization using the trust-exact method."""
        # gtol=1e-8: {"trust-ncg": {"mean": 0.8204149252450154, "stddev": 1.1861317262332736, "mean/stddev": 0.6916726929228655}}
        # gtol=1e-7: {"trust-ncg": {"mean": 0.8204149252450154, "stddev": 1.1861317262332736, "mean/stddev": 0.6916726929228655}}
        # gtol=5e-7: {"trust-ncg": {"mean": 0.8204149252450154, "stddev": 1.1861317262332736, "mean/stddev": 0.6916726929228655}}
        # gtol=5e-6: {"trust-exact": {"mean": 0.8204149259252502, "stddev": 1.1861317270880474, "mean/stddev": 0.6916726929979087}}
        # gtol=5e-5: {"trust-exact": {"mean": 0.8204149256912995, "stddev": 1.1861317267359337, "mean/stddev": 0.6916726930059994}}
        # gtol=1e-4: {"trust-exact": {"mean": 0.8204149256912995, "stddev": 1.1861317267359337, "mean/stddev": 0.6916726930059994}}
        # observed gtol=1e-6 hangs
        options = {'maxiter': 1000, 'gtol': 5e-5}
        return self.do_minimize("trust-exact", hessian=hessian, jacobian=jacobian, objective=objective, options=options)

    def minimize_trust_krylov(self, hessian, jacobian, objective):
        """Run optimization using the trust-krylov method."""
        options = {'maxiter': 1000, 'gtol': 5e-5}
        # gtol: 1e-8: [FAILED]
        # gtol: 1e-7:
        """************
        trust-krylov: RMSE in pips
         [1.71229032e-03 2.32948446e+00 6.35762914e-03 4.88400327e-02
         2.60182693e-01 9.23216872e-02 4.45588339e-02 1.97117109e-01
         2.05994174e-01 3.27281032e+00 2.56518495e+00]
        ************
        RMSE pips per method:
         u/s = 0.6916726929231727
         {"trust-krylov": {"mean": 0.8204149252438392, "stddev": 1.1861317262310462}}
         """

        # gtol: 1e-6:
        """************
        trust-krylov: RMSE in pips
         [1.71229033e-03 2.32948445e+00 6.35762914e-03 4.88400327e-02
         2.60182693e-01 9.23216873e-02 4.45588339e-02 1.97117109e-01
         2.05994174e-01 3.27281032e+00 2.56518495e+00]
        ************
        RMSE pips per method:
        u/s = 0.6916726929233246
        {"trust-krylov": {"mean": 0.8204149253361106, "stddev": 1.186131726364189}}"""

        # gtol: 1e-5:
        """************
        trust-krylov: RMSE in pips
         [1.71229033e-03 2.32948445e+00 6.35762915e-03 4.88400328e-02
         2.60182694e-01 9.23216859e-02 4.45588339e-02 1.97117110e-01
         2.05994177e-01 3.27281033e+00 2.56518495e+00]
        ************
        RMSE pips per method:
        u/s = 0.6916726929434377
        {"trust-krylov": {"mean": 0.8204149257500504, "stddev": 1.1861317269281595}}"""

        # gtol: 1e-4:
        """************
        trust-krylov: RMSE in pips
         [1.71229034e-03 2.32948445e+00 6.35762900e-03 4.88400327e-02
         2.60182691e-01 9.23216768e-02 4.45588341e-02 1.97117119e-01
         2.05994200e-01 3.27281034e+00 2.56518494e+00]
        ************
        RMSE pips per method:
        u/s = 0.6916726948390248
        {"trust-krylov": {"mean": 0.8204149272255961, "stddev": 1.1861317258107664}}"""

        # gtol: 1e-3:
        """************
        trust-krylov: RMSE in pips
         [1.71229216e-03 2.32948288e+00 6.35762227e-03 4.88401016e-02
         2.60181140e-01 9.23199192e-02 4.45588095e-02 1.97117728e-01
         2.05996045e-01 3.27281373e+00 2.56518110e+00]
        ************
        RMSE pips per method:
        u/s = 0.6916724843142598
        {"trust-krylov": {"mean": 0.8204146695938633, "stddev": 1.186131714357904}}"""

        # gtol: 1e-2:
        """************
        trust-krylov: RMSE in pips
         [1.71024250e-03 2.33081512e+00 6.35303816e-03 4.88144658e-02
         2.60032152e-01 9.22100505e-02 4.45840980e-02 1.97218499e-01
         2.05807434e-01 3.27052464e+00 2.56469025e+00]
        ************
        RMSE pips per method:
        u/s=0.6917236965454238
        {"trust-krylov": {"mean": 0.820250908216494, "stddev": 1.1858071543781936}}"""

        # gtol: 1e-1:
        """************
        trust-krylov: RMSE in pips
         [1.69090315e-03 2.34311587e+00 6.29848707e-03 4.85414927e-02
         2.58782318e-01 9.15741436e-02 4.48920301e-02 1.99721310e-01
         2.04387078e-01 3.24866564e+00 2.55952465e+00]
        ************
        RMSE pips per method:
        u/s=0.6924652642693532
        {"trust-krylov": {"mean": 0.8188358116414434, "stddev": 1.182493698807302}}"""

        return self.do_minimize("trust-krylov", hessian=hessian, jacobian=jacobian, objective=objective,
                                options=options)

    def minimize_slsqp(self, hessian, jacobian, objective):
        """Run optimization using the SLSQP method."""
        # gtol: 1e-24: {"SLSQP": {"mean": 0.8204149252462497, "stddev": 1.1861317262280586, "mean/stddev": 0.6916726929269471}}
        # gtol: 1e-20: {"SLSQP": {"mean": 0.8204149252461385, "stddev": 1.1861317262280602, "mean/stddev": 0.6916726929268524}}
        # gtol: 1e-18: {"SLSQP": {"mean": 0.8204149252089913, "stddev": 1.1861317262701438, "mean/stddev": 0.6916726928709941}}
        # gtol: 1e-16: {"SLSQP": {"mean": 0.8204149251994085, "stddev": 1.1861317266206104, "mean/stddev": 0.6916726926585465}}
        # gtol: 1e-14: {"SLSQP": {"mean": 0.8204139848074979, "stddev": 1.1861337893177848, "mean/stddev": 0.6916706970124898}}
        # gtol: 1e-13: {"SLSQP": {"mean": 0.8204139509992281, "stddev": 1.1861341906179041, "mean/stddev": 0.6916704344993563}}
        # gtol: 1e-12: {"SLSQP": {"mean": 0.8204146871661098, "stddev": 1.1861321485743537, "mean/stddev": 0.6916722459231797}}
        # gtol: 1e-11: {"SLSQP": {"mean": 0.8204138685765442, "stddev": 1.1861179684693401, "mean/stddev": 0.6916798247608295}}
        # gtol: 1e-10: {"SLSQP": {"mean": 0.8203955123703259, "stddev": 1.1860254811275552, "mean/stddev": 0.6917182855046043}}
        # gtol: 1e-9:  {"SLSQP": {"mean": 0.8203323870319171, "stddev": 1.1857850324905062, "mean/stddev": 0.6918053142473655}}
        # gtol: 1e-8: {"SLSQP": {"mean": 0.8203275545509996, "stddev": 1.1857280394192342, "mean/stddev": 0.6918344909451525}}
        # gtol: 1e-7: {"SLSQP": {"mean": 0.8203257717283221, "stddev": 1.185859294976963, "mean/stddev": 0.6917564125887785}}
        # gtol: 1e-6: {"SLSQP": {"mean": 0.8206725577731102, "stddev": 1.185941367156532, "mean/stddev": 0.6920009542636941}}
        # gtol: 1e-5: {"SLSQP": {"mean": 0.8207716117779055, "stddev": 1.1858447707096997, "mean/stddev": 0.692140853550919}}
        # gtol: 1e-4: {"SLSQP": {"mean": 0.8213239368495987, "stddev": 1.1853547688292188, "mean/stddev": 0.6928929283009717}}
        options = {'maxiter': 1000, 'ftol': 1e-14}
        return self.do_minimize('slsqp', hessian=None, jacobian=jacobian, objective=objective, options=options)

    def do_minimize(self, method, hessian, jacobian, objective, options):
        # Running the optimizer
        return minimize(fun=objective,
                        x0=self._initial_guess,
                        method=method,
                        jac=jacobian,
                        hess=hessian,
                        options=options | {'disp': False})

    def recouple(self, currency_map: pd.Series) -> pd.Series:
        """
        Computes the exchange rate for each pair as exp(log(base_currency) - log(quote_currency)),
        and then scales it by the pip size to reverse the scaling applied during decoupling.

        Parameters:
        - currency_map: A dictionary where keys are individual currencies and values are their strengths.

        Returns:
        - A dictionary where keys are currency pairs and values are their calculated exchange rates.
        """
        return pd.Series({
            pair: self.quotient(currency_map, pair)
            for pair in self._pair_columns
        })

    def quotient(self, currency_map, pair):
        lnA = currency_map[pair[:3]]
        lnB = currency_map[pair[-3:]]
        # c = max(lnA, lnB)
        # Exploiting e^(x +/-c) - (y +/- c)  == e^(x-y) to hopefully enhance stability
        return mp.exp(lnA - lnB)


class ContinuousDecoupleService(DecoupleService):
    def __init__(self, pair_columns: List[str], algo: str = "trust-krylov", use_row_scaling: bool = False,
                 use_col_scaling: bool = False, use_recursive_reduction: bool = False):
        super().__init__(pair_columns)
        self.set_algo(algo)
        self._use_row_scaling = use_row_scaling
        self._use_col_scaling = use_col_scaling
        self._use_recursive_reduction = use_recursive_reduction
        self._current_values = {}

    def set_algo(self, algo):
        if algo not in self._minimize_methods.keys():
            raise ValueError(f"Unknown algorithm: {algo}")
        self._algo = algo

    def update(self, tick: dict):
        instrument = tick['instrument']
        mid = tick['scaled_mid']
        self._current_values[instrument] = mid
        if len(self._current_values.keys()) < len(self._pair_columns):
            return None

        return super().decouple(self._current_values)


__all_ = [DecoupleService]
