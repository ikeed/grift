package com.grift.math;

import java.util.Arrays;
import java.util.stream.IntStream;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Strings;
import com.grift.forex.symbol.ImmutableSymbolIndexMap;
import com.grift.forex.symbol.SymbolIndexMap;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import static lombok.Lombok.checkNotNull;
import static org.apache.commons.math.util.MathUtils.EPSILON;

public class ProbabilityVector {
    @NotNull
    private final ImmutableSymbolIndexMap symbolIndexMap;
    @NotNull
    private final double[] values;

    private boolean normalizationRequired = false;
    private int nonZeroElements = 0;
    private double elementSum = 0;

    private ProbabilityVector(@NotNull SymbolIndexMap symbolIndexMap) {
        this.symbolIndexMap = checkNotNull(symbolIndexMap, "map").getImmutableCopy();
        this.values = new double[symbolIndexMap.keySet().size()];
        this.nonZeroElements = 0;
        this.normalizationRequired = false;
        this.elementSum = 0;
        Arrays.fill(values, 0d);
    }

    private ProbabilityVector(@NotNull SymbolIndexMap symbolIndexMap, @NotNull double[] values) {
        this.symbolIndexMap = checkNotNull(symbolIndexMap, "map").getImmutableCopy();
        this.values = Arrays.copyOf(values, values.length);
        if (symbolIndexMap.size() != values.length) {
            throw new IllegalArgumentException(String.format("Wrong number of values for symbol table.  Expected: %d, Actual: %d", symbolIndexMap.size(), values.length));
        }
        setProperties(values);
    }

    @NotNull
    public static double[] normalize(@NotNull final double[] data) {
        return Arrays.stream(checkNotNull(data, "data")).map(d -> d / Arrays.stream(data).sum()).toArray();
    }

    public void put(@NotNull String symbol, double v) {
        assertSymbol(symbol);
        put(symbolIndexMap.get(symbol), v);
    }

    public void put(int index, double v) {
        assertIndex(index);
        if (v < 0) {
            throw new IllegalArgumentException("No negative values allowed");
        }
        if (isZero(values[index]) && !isZero(v)) {
            nonZeroElements++;
        } else if (isZero(v) && !isZero(values[index])) {
            nonZeroElements--;
        }
        final double delta = v - values[index];
        normalizationRequired = normalizationRequired || !isZero(delta);
        elementSum += delta;
        values[index] = v;
    }

    public double get(@NotNull String symbol) {
        assertSymbol(symbol);
        return get(symbolIndexMap.get(symbol));
    }

    public double get(int index) {
        assertIndex(index);
        if (normalizationRequired) {
            normalize();
        }
        return values[index];
    }

    public int getDimension() {
        return symbolIndexMap.size();
    }

    @NotNull
    @VisibleForTesting
    ImmutableSymbolIndexMap getSymbolIndexMap() {
        return symbolIndexMap;
    }

    @NotNull
    public double[] getValues() {
        if (normalizationRequired) {
            normalize();
        }
        return Arrays.copyOf(values, getDimension());
    }

    @NotNull
    @Override
    public String toString() {
        if (normalizationRequired) {
            normalize();
        }
        return Arrays.toString(values);
    }

    @Override
    public int hashCode() {
        return Arrays.toString(values).hashCode();
    }

    @Override
    public boolean equals(@Nullable Object obj) {
        ProbabilityVector that;
        if (obj == null || !(obj instanceof ProbabilityVector)) {
            return false;
        }
        that = (ProbabilityVector) obj;
        return Arrays.equals(this.getValues(), that.getValues());
    }

    private void normalize() {
        if (elementSum != 0) {
            IntStream.range(0, values.length).forEach(i -> values[i] = values[i] / elementSum);
        }
        elementSum = 1;
        normalizationRequired = false;
    }

    private void assertIndex(int index) {
        if (!isLegalIndex(index)) {
            throw new IndexOutOfBoundsException(index + " not in array");
        }
    }

    private void assertSymbol(@NotNull String symbol) {
        if (!isLegalSymbol(symbol)) {
            throw new IllegalArgumentException("Unknown symbol: " + symbol);
        }
    }

    private void setProperties(@NotNull double[] values) {
        this.nonZeroElements = 0;
        this.elementSum = 0;
        Arrays.stream(values).forEach(d -> {
            this.nonZeroElements += (d == 0 ? 0 : 1);
            this.elementSum += d;
        });
        this.normalizationRequired = calculateIsNormalizationRequired();
    }

    private boolean calculateIsNormalizationRequired() {
        return !(isZero(elementSum - 1) && nonZeroElements == symbolIndexMap.size());
    }

    private boolean isZero(double v) {
        return Math.abs(v) < EPSILON;
    }

    private boolean isLegalSymbol(@NotNull String symbol) {
        return !Strings.isNullOrEmpty(symbol) && symbolIndexMap.containsKey(symbol);
    }

    private boolean isLegalIndex(int index) {
        return symbolIndexMap.containsValue(index) && index >= 0 && index < values.length;
    }

    public static class Factory {
        private final ImmutableSymbolIndexMap immutableSymbolIndexMap;

        public Factory(String... symbols) {
            this(new ImmutableSymbolIndexMap(symbols));
        }

        public Factory(ImmutableSymbolIndexMap immutableSymbolIndexMap) {
            this.immutableSymbolIndexMap = immutableSymbolIndexMap;
        }

        @NotNull
        public ProbabilityVector create() {
            return new ProbabilityVector(immutableSymbolIndexMap);
        }

        @NotNull
        public ProbabilityVector create(@NotNull double... values) {
            return new ProbabilityVector(immutableSymbolIndexMap, values);
        }

        @NotNull
        public ProbabilityVector copy(@NotNull ProbabilityVector probabilityVector) {
            if (probabilityVector.getSymbolIndexMap().equals(immutableSymbolIndexMap)) {
                return new ProbabilityVector(immutableSymbolIndexMap, probabilityVector.values);
            }
            throw new IllegalStateException("Symbol tables don't match");
        }
    }
}
