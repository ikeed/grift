package com.grift.math.decoupler;

import cern.colt.list.DoubleArrayList;
import cern.colt.list.IntArrayList;
import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.SparseDoubleMatrix1D;
import cern.colt.matrix.impl.SparseDoubleMatrix2D;
import cern.colt.matrix.linalg.EigenvalueDecomposition;
import com.grift.forex.symbol.ImmutableSymbolIndexMap;
import com.grift.math.ProbabilityVector;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.forex.symbol.SymbolPair;
import com.google.common.annotations.VisibleForTesting;
import lombok.NonNull;
import org.jetbrains.annotations.NotNull;

import static lombok.Lombok.checkNotNull;

/**
 * Implementation of decoupling algorithm.  Takes paired-currency ticks as input and produces
 * an individual value for each currency in the system.
 */
public class DecouplerMatrixColtImpl implements DecouplerMatrix {
    @NonNull
    private final ImmutableSymbolIndexMap symbolIndexMap;
    @NotNull
    private final SparseDoubleMatrix2D mat;


    private DecouplerMatrixColtImpl(@NonNull ImmutableSymbolIndexMap symbolIndexMap, @NotNull double[][] doubles) {
        this.mat = new SparseDoubleMatrix2D(doubles);
        this.symbolIndexMap = symbolIndexMap;
    }

    private DecouplerMatrixColtImpl(@NonNull ImmutableSymbolIndexMap symbolIndexMap) {
        this.mat = new SparseDoubleMatrix2D(symbolIndexMap.size(), symbolIndexMap.size());
        this.symbolIndexMap = symbolIndexMap;
    }

    @NotNull
    @Override
    public ProbabilityVector decouple() {
        SparseDoubleMatrix1D solution = findSolutionVector();
        return new ProbabilityVector(symbolIndexMap, solution.toArray());
    }

    @Override
    public int rows() {
        return mat.rows();
    }

    @Override
    public int columns() {
        return mat.columns();
    }

    @Override
    public void put(@NonNull SymbolPair symbolPair, double val) {
        Integer[] indeces = symbolIndexMap.getIndecesForSymbolPair(symbolPair);
        set(indeces[0], indeces[1], val);
    }

    @Override
    public double get(@NonNull SymbolPair symbolPair) {
        Integer[] indeces = symbolIndexMap.getIndecesForSymbolPair(symbolPair);
        return get(indeces[0], indeces[1]);
    }

    @NotNull
    private SparseDoubleMatrix1D findSolutionVector() {
        final EigenvalueDecomposition decomposition = getEigenvalueDecomposition();
        final DoubleMatrix1D solutionVector = getEigenvectorWithEigenvalue1(decomposition);
        return new SparseDoubleMatrix1D(solutionVector.toArray());
    }

    private DoubleMatrix1D getEigenvectorWithEigenvalue1(EigenvalueDecomposition decomposition) {
        final DoubleMatrix1D eigenValues = DoubleFactory2D.sparse.diagonal(decomposition.getD());
        final DoubleMatrix2D eigenVectors = decomposition.getV();
        int indexOf1 = findIndexOfEigenvalue1(eigenValues);
        return eigenVectors.viewColumn(indexOf1);
    }

    @NotNull
    private EigenvalueDecomposition getEigenvalueDecomposition() {
        final SparseDoubleMatrix2D prepared = prepareForDecoupling();
        return new EigenvalueDecomposition(prepared);
    }

    private double get(int row, int column) {
        return mat.get(row, column);
    }

    @VisibleForTesting
    private void set(int row, int column, double val) {
        mat.setQuick(row, column, val);
    }

    private int findIndexOfEigenvalue1(DoubleMatrix1D values) {
        double diff = 9999;
        int bestIx = -1;

        for (int i = 0; i < values.size(); i++) {
            final double d = Math.abs(values.get(i) - 1);
            if (d < diff) {
                diff = d;
                bestIx = i;
            }
        }
        return bestIx;
    }

    @NotNull
    private SparseDoubleMatrix2D prepareForDecoupling() {
        final SparseDoubleMatrix2D newMat = new SparseDoubleMatrix2D(mat.toArray());
        for (int i = 0; i < newMat.rows(); i++) {
            newMat.setQuick(i, i, 1);
            DoubleArrayList values = new DoubleArrayList();
            IntArrayList indeces = new IntArrayList();
            newMat.viewRow(i).getNonZeros(indeces, values);
            newMat.setQuick(i, i, 1 - indeces.size());
        }
        return newMat;
    }

    public static class Factory implements com.grift.math.decoupler.Factory {
        @NotNull
        @NonNull
        private final ImmutableSymbolIndexMap symbolIndexMap;

        public Factory(@NotNull @NonNull SymbolIndexMap symbolIndexMap) {
            this.symbolIndexMap = new ImmutableSymbolIndexMap(checkNotNull(symbolIndexMap, "map"));
        }

        @NotNull
        @Override
        public DecouplerMatrix make(@NotNull @NonNull double[][] doubles) {
            if (doubles.length != symbolIndexMap.size() || doubles[0].length != symbolIndexMap.size()) {
                throw new IllegalArgumentException("The array of values doesn't match the number of symbols in the map");
            }
            return new DecouplerMatrixColtImpl(symbolIndexMap, doubles);
        }

        @NotNull
        @Override
        public DecouplerMatrix make() {
            return new DecouplerMatrixColtImpl(symbolIndexMap);
        }
    }
}
