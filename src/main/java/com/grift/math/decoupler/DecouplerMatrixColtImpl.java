package com.grift.math.decoupler;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.stream.IntStream;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.linalg.EigenvalueDecomposition;
import com.google.common.collect.Sets;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.forex.symbol.SymbolPair;
import com.grift.math.real.Real;
import com.grift.math.stats.ProbabilityVector;
import org.jetbrains.annotations.NotNull;

import static com.grift.math.real.Real.ZERO;
import static com.google.common.base.Preconditions.checkNotNull;
import static org.apache.commons.math.util.MathUtils.EPSILON;

/**
 * Implementation of decoupling algorithm.  Takes paired-currency ticks as input and produces
 * an individual value for each currency in the system.
 */
public class DecouplerMatrixColtImpl implements DecouplerMatrix {

    @NotNull
    private final SymbolIndexMap symbolIndexMap;

    @NotNull
    private final Map<SymbolPair, Real> mostRecentValue;

    private DecouplerMatrixColtImpl(@NotNull SymbolIndexMap symbolIndexMap) {
        this.symbolIndexMap = symbolIndexMap;
        this.mostRecentValue = new HashMap<>();
    }

    @NotNull
    private static DoubleMatrix1D getSteadyStateVector(@NotNull EigenvalueDecomposition eigenDecomposition) {
        int bestIx = getIndexOfEigenvalue1(eigenDecomposition);
        return eigenDecomposition.getV().viewColumn(bestIx);
    }

    private static int getIndexOfEigenvalue1(@NotNull EigenvalueDecomposition eigenDecomposition) {
        int bestIx = -1;
        double bestDiff = 1000;
        for (int i = 0; i < eigenDecomposition.getRealEigenvalues().size(); i++) {
            double diff = Math.abs(1 - eigenDecomposition.getRealEigenvalues().get(i));
            if (bestIx < 0 || diff < bestDiff) {
                bestDiff = diff;
                bestIx = i;
            }
        }
        return bestIx;
    }

    private static void scaleRow(@NotNull DoubleMatrix2D newMat, int i, double scalar) {
        IntStream.range(0, newMat.columns()).forEach(j -> newMat.set(i, j, newMat.get(i, j) * scalar));
    }

    private static long getNonZeroCountForRow(@NotNull DoubleMatrix2D newMat, int i) {
        return IntStream.range(0, newMat.columns())
                .filter(j -> Math.abs(newMat.get(i, j)) > EPSILON)
                .count();
    }

    @NotNull
    private static DoubleMatrix2D getIdentity(int n) {
        return DoubleFactory2D.dense.identity(n);
    }

    @NotNull
    @Override
    public ProbabilityVector decouple() {
        DoubleMatrix1D solution = findSolutionVector();
        return new ProbabilityVector.Factory(symbolIndexMap.getImmutableCopy()).create(solution.toArray());
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
    private DoubleMatrix1D findSolutionVector() {
        final DoubleMatrix2D coefficientMatrix = prepareForDecoupling();
        EigenvalueDecomposition eigenDecomposition = new EigenvalueDecomposition(coefficientMatrix);
        return getSteadyStateVector(eigenDecomposition);
    }

    @NotNull
    private DoubleMatrix2D prepareForDecoupling() {
        DoubleMatrix2D stochasticMatrix = createStartingMatrix();
        IntStream.range(0, stochasticMatrix.rows()).forEach(i -> {
            long nonZeroCount = getNonZeroCountForRow(stochasticMatrix, i);
            if (nonZeroCount != 0) {
                scaleRow(stochasticMatrix, i, 1d / nonZeroCount);
            }
        });
        return stochasticMatrix;
    }

    @NotNull
    private DoubleMatrix2D createStartingMatrix() {
        DoubleMatrix2D mat = getIdentity(rows());
        mostRecentValue.forEach((key, value) -> setMatrixCellFromPair(mat, key, value));
        return mat;
    }

    private void setMatrixCellFromPair(@NotNull DoubleMatrix2D mat, @NotNull SymbolPair key, Real value) {
        Integer row = symbolIndexMap.get(key.getFirst());
        Integer col = symbolIndexMap.get(key.getSecond());
        mat.set(row, col, value.toDouble());
    }

    public static class ColtFactory implements Factory {

        @NotNull
        private final SymbolIndexMap symbolIndexMap;

        public ColtFactory(@NotNull SymbolIndexMap symbolIndexMap) {
            this.symbolIndexMap = checkNotNull(symbolIndexMap, "symbolIndexMap");
        }

        @NotNull
        @Override
        public DecouplerMatrix make() {
            return new DecouplerMatrixColtImpl(symbolIndexMap);
        }
    }
}
