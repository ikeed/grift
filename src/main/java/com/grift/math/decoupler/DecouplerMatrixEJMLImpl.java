package com.grift.math.decoupler;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.IntStream;
import com.google.common.collect.Sets;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.forex.symbol.SymbolPair;
import com.grift.math.real.Real;
import com.grift.math.stats.ProbabilityVector;
import org.ejml.data.Complex64F;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.interfaces.decomposition.EigenDecomposition;
import org.jetbrains.annotations.NotNull;

import static com.grift.math.real.Real.ZERO;
import static com.google.common.base.Preconditions.checkNotNull;
import static org.apache.commons.math.util.MathUtils.EPSILON;

/**
 * Implementation of decoupling algorithm.  Takes paired-currency ticks as input and produces
 * an individual value for each currency in the system.
 */
public class DecouplerMatrixEJMLImpl implements DecouplerMatrix {

    @NotNull
    private final EigenDecomposition<DenseMatrix64F> eigenDecomposition;

    @NotNull
    private final SymbolIndexMap symbolIndexMap;

    @NotNull
    private final ProbabilityVector.Factory vectorFactory;

    @NotNull
    private final Map<SymbolPair, Real> mostRecentValue;

    private DecouplerMatrixEJMLImpl(@NotNull SymbolIndexMap symbolIndexMap) {
        this.symbolIndexMap = symbolIndexMap;
        this.mostRecentValue = new HashMap<>();
        this.eigenDecomposition = DecompositionFactory.eig(symbolIndexMap.size(), true);
        this.vectorFactory = new ProbabilityVector.Factory(symbolIndexMap.getImmutableCopy());
    }

    @NotNull
    private static DenseMatrix64F getSteadyStateVector(@NotNull EigenDecomposition<DenseMatrix64F> eigenDecomposition, @NotNull DenseMatrix64F coefficientMatrix) {
        eigenDecomposition.decompose(coefficientMatrix);
        int bestIx = findIndexOfEigenvalue1(eigenDecomposition);
        return eigenDecomposition.getEigenVector(bestIx);
    }

    private static int findIndexOfEigenvalue1(@NotNull EigenDecomposition<DenseMatrix64F> eigenDecomposition) {
        double bestDiff = 10000;
        int bestIx = -1;
        for (int i = 0; i < eigenDecomposition.getNumberOfEigenvalues(); i++) {
            Complex64F eigenvalue = eigenDecomposition.getEigenvalue(i);

            double diff = Math.sqrt(Math.pow((eigenvalue.real - 1), 2) + Math.pow(eigenvalue.imaginary, 2));
            if (bestIx < 0 || diff < bestDiff) {
                bestDiff = diff;
                bestIx = i;
            }
        }
        return bestIx;
    }

    private static void scaleRow(@NotNull DenseMatrix64F newMat, int i, double scalar) {
        IntStream.range(0, newMat.numCols).forEach(j -> newMat.set(i, j, newMat.get(i, j) * scalar));
    }

    private static long getNonZeroCountForRow(@NotNull DenseMatrix64F newMat, int i) {
        return IntStream.range(0, newMat.numCols)
                .filter(j -> Math.abs(newMat.get(i, j)) > EPSILON)
                .count();
    }

    @NotNull
    private static DenseMatrix64F getIdentity(int n) {
        DenseMatrix64F mat = new DenseMatrix64F(n, n);
        IntStream.range(0, mat.numCols).forEach(i -> mat.set(i, i, 1));
        return mat;
    }

    @NotNull
    @Override
    public ProbabilityVector decouple() {
        DenseMatrix64F solution = findSolutionVector();
        return vectorFactory.create(solution.data);
    }

    @Override
    public int rows() {
        return symbolIndexMap.size();
    }

    @Override
    public int columns() {
        return symbolIndexMap.size();
    }

    @Override
    public void put(@NotNull SymbolPair symbolPair, Real val) {
        if (val.isNegative()) {
            throw new IllegalArgumentException("No negative values allowed");
        }
        mostRecentValue.put(symbolPair, val);
        symbolIndexMap.addSymbolPair(symbolPair);
    }

    @Override
    public Real get(@NotNull SymbolPair symbolPair) {
        if (mostRecentValue.containsKey(symbolPair)) {
            return mostRecentValue.get(symbolPair);
        }
        return ZERO;
    }

    @Override
    public boolean isReplete() {
        Set<String> unique = Sets.newHashSet();
        mostRecentValue.keySet().forEach(p -> {
            unique.add(p.getFirst());
            unique.add(p.getSecond());
        });
        return unique.size() == symbolIndexMap.size();
    }

    @NotNull
    private DenseMatrix64F findSolutionVector() {
        final DenseMatrix64F coefficientMatrix = prepareForDecoupling();
        return getSteadyStateVector(eigenDecomposition, coefficientMatrix);
    }

    @NotNull
    private DenseMatrix64F prepareForDecoupling() {
        DenseMatrix64F stochasticMatrix = createStartingMatrix();
        IntStream.range(0, stochasticMatrix.numRows).forEach(i -> {
            long nonZeroCount = getNonZeroCountForRow(stochasticMatrix, i);
            if (nonZeroCount != 0) {
                scaleRow(stochasticMatrix, i, 1d / nonZeroCount);
            }
        });
        return stochasticMatrix;
    }

    @NotNull
    private DenseMatrix64F createStartingMatrix() {
        DenseMatrix64F mat = getIdentity(rows());
        mostRecentValue.forEach((key, value) -> setMatrixCellFromPair(mat, key, value));
        return mat;
    }

    private void setMatrixCellFromPair(@NotNull DenseMatrix64F mat, @NotNull SymbolPair key, Real value) {
        Integer row = symbolIndexMap.get(key.getFirst());
        Integer col = symbolIndexMap.get(key.getSecond());
        mat.set(row, col, value.toDouble());
    }

    public static class EJMLFactory implements Factory {

        @NotNull
        private final SymbolIndexMap symbolIndexMap;

        public EJMLFactory(@NotNull SymbolIndexMap symbolIndexMap) {
            this.symbolIndexMap = checkNotNull(symbolIndexMap, "symbolIndexMap");
        }

        @NotNull
        @Override
        public DecouplerMatrix make() {
            return new DecouplerMatrixEJMLImpl(symbolIndexMap);
        }
    }
}
