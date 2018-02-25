package com.grift.math.decoupler;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.forex.symbol.SymbolPair;
import com.grift.math.ProbabilityVector;
import lombok.NonNull;
import org.ejml.data.Complex64F;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.interfaces.decomposition.EigenDecomposition;
import org.jetbrains.annotations.NotNull;

import static org.apache.commons.math.util.MathUtils.EPSILON;

/**
 * Implementation of decoupling algorithm.  Takes paired-currency ticks as input and produces
 * an individual value for each currency in the system.
 */
public class DecouplerMatrixColtImpl implements DecouplerMatrix {

    private final EigenDecomposition<DenseMatrix64F> eigenDecomposition;

    @NonNull
    private final SymbolIndexMap symbolIndexMap;

    @NotNull @NonNull
    private final ProbabilityVector.Factory vectorFactory;

    @NonNull
    private final Map<SymbolPair, Double> mostRecentValue;

    private DecouplerMatrixColtImpl(@NonNull SymbolIndexMap symbolIndexMap) {
        this.symbolIndexMap = symbolIndexMap;
        this.mostRecentValue = new HashMap<>();
        this.eigenDecomposition = DecompositionFactory.eig(symbolIndexMap.size(), true);
        this.vectorFactory = new ProbabilityVector.Factory(symbolIndexMap.getImmutableCopy());
    }

    @NotNull
    @Override
    public ProbabilityVector decouple() {
        DenseMatrix64F solution = findSolutionVector();
        return vectorFactory.create(normalize(solution.data));
    }

    private double[] normalize(final double[] data) {
        return Arrays.stream(data).map(d -> d / Arrays.stream(data).sum()).toArray();
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
    public void put(@NonNull SymbolPair symbolPair, double val) {
        mostRecentValue.put(symbolPair, val);
        symbolIndexMap.addSymbolPair(symbolPair);
    }

    @Override
    public double get(@NonNull SymbolPair symbolPair) {
        if (mostRecentValue.containsKey(symbolPair)) {
            return mostRecentValue.get(symbolPair);
        }
        return 0;
    }

    @NotNull
    @NonNull
    private DenseMatrix64F findSolutionVector() {
        final DenseMatrix64F prepared = prepareForDecoupling();
        return getSteadyStateVector(prepared);
    }

    private DenseMatrix64F getSteadyStateVector(DenseMatrix64F prepared) {
        eigenDecomposition.decompose(prepared);
        int bestIx = findIndexOfEigenvalue1(eigenDecomposition);
        return eigenDecomposition.getEigenVector(bestIx);
    }

    private int findIndexOfEigenvalue1(EigenDecomposition<DenseMatrix64F> eigenDecomposition) {
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

    @NotNull
    @NonNull
    private DenseMatrix64F prepareForDecoupling() {
        DenseMatrix64F newMat = createStartingMatrix();
        for (int i = 0; i < newMat.numRows; i++) {
            int nonZeroCount = getNonZeroCountForRow(newMat, i);
            if (nonZeroCount != 0) {
                scaleRow(newMat, i, 1d / nonZeroCount);
            }
        }
        return newMat;
    }

    private void scaleRow(@NotNull @NonNull DenseMatrix64F newMat, int i, double scalar) {
        for (int j = 0; j < newMat.numCols; j++) {
            newMat.set(i, j, newMat.get(i, j) * scalar);
        }
    }

    private int getNonZeroCountForRow(@NotNull @NonNull DenseMatrix64F newMat, int i) {
        int nonZeroCount = 0;
        for (int j = 0; j < newMat.numCols; j++) {
            nonZeroCount += Math.abs(newMat.get(i, j)) < EPSILON ? 0 : 1;
        }
        return nonZeroCount;
    }

    @NonNull
    @NotNull
    private DenseMatrix64F createStartingMatrix() {
        DenseMatrix64F mat = getIdentity(rows());

        for (Map.Entry<SymbolPair, Double> entry : mostRecentValue.entrySet()) {
            setMatrixCellFromPair(mat, entry.getKey(), entry.getValue());
        }
        return mat;
    }

    @NonNull
    @NotNull
    private DenseMatrix64F getIdentity(int n) {
        DenseMatrix64F mat = new DenseMatrix64F(n, n);

        for (int i = 0; i < mat.numCols; i++) {
            mat.set(i, i, 1);
        }
        return mat;
    }

    private void setMatrixCellFromPair(@NonNull @NotNull DenseMatrix64F mat, @NonNull @NotNull SymbolPair key, double value) {
        Integer row = symbolIndexMap.get(key.getFirst());
        Integer col = symbolIndexMap.get(key.getSecond());
        mat.set(row, col, value);
    }

    public static class ColtFactory implements Factory {

        @NonNull
        private final SymbolIndexMap symbolIndexMap;

        public ColtFactory(SymbolIndexMap symbolIndexMap) {
            this.symbolIndexMap = symbolIndexMap;
        }

        @Override
        public DecouplerMatrix make() {
            return new DecouplerMatrixColtImpl(symbolIndexMap);
        }
    }
}
