package com.grift.math;

import java.util.Arrays;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Strings;
import com.grift.forex.symbol.ImmutableSymbolIndexMap;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.math.real.Real;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import static com.grift.math.real.Real.ONE;
import static com.grift.math.real.Real.ZERO;
import static lombok.Lombok.checkNotNull;

public class ProbabilityVector {
    @NotNull
    private final ImmutableSymbolIndexMap symbolIndexMap;
    @NotNull
    private final Real[] values;

    private boolean normalizationRequired = false;
    private int nonZeroElements = 0;
    private Real elementSum = ZERO;

    private ProbabilityVector(@NotNull SymbolIndexMap symbolIndexMap) {
        this.symbolIndexMap = checkNotNull(symbolIndexMap, "map").getImmutableCopy();
        this.values = new Real[symbolIndexMap.keySet().size()];
        this.nonZeroElements = 0;
        this.normalizationRequired = false;
        this.elementSum = ZERO;
        Arrays.fill(values, ZERO);
    }

    private ProbabilityVector(@NotNull SymbolIndexMap symbolIndexMap, @NotNull Real[] values) {
        this.symbolIndexMap = checkNotNull(symbolIndexMap, "map").getImmutableCopy();
        this.values = normalize(Arrays.copyOf(values, values.length));
        if (symbolIndexMap.size() != values.length) {
            throw new IllegalArgumentException(String.format("Wrong number of values for symbol table.  Expected: %d, Actual: %d", symbolIndexMap.size(), values.length));
        }
        setProperties(values);
    }

    @NotNull
    public static Real[] normalize(@NotNull final Real[] data) {
        Real sum = ZERO;

        for (Real d : data) {
            sum = sum.plus(d);
        }
        if (!sum.isZero()) {
            for (int i = 0; i < data.length; i++) {
                Real normalized = data[i].divide(sum);
                if (normalized.isNegative()) {
                    throw new IllegalArgumentException("Negative elements are not allowed");
                }
                data[i] = normalized;
            }
        }
        return data;
    }

    public void put(@NotNull String symbol, Real v) {
        assertSymbol(symbol);
        put(symbolIndexMap.get(symbol), v);
    }

    public void put(int index, Real val) {
        assertIndex(index);
        if (val.isNegative()) {
            throw new IllegalArgumentException("No negative values allowed");
        }
        Real v = new Real(val);
        if (values[index].isZero() && !v.isZero()) {
            nonZeroElements++;
        } else if (v.isZero() && !values[index].isZero()) {
            nonZeroElements--;
        }
        final Real delta = v.subtract(values[index]);
        normalizationRequired = normalizationRequired || !delta.isZero();
        elementSum = elementSum.plus(delta);
        values[index] = v;
    }

    public Real get(@NotNull String symbol) {
        assertSymbol(symbol);
        return get(symbolIndexMap.get(symbol));
    }

    public Real get(int index) {
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
    public Real[] getValues() {
        if (normalizationRequired) {
            normalize();
        }
        Real[] arr = new Real[getDimension()];
        for (int i = 0; i < getDimension(); i++) {
            arr[i] = new Real(values[i]);
        }
        return arr;
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
        for (int i = 0; i < getDimension(); i++) {
            if (!values[i].equals(that.values[i])) {
                return false;
            }
        }
        return true;
    }

    private void normalize() {
        if (!elementSum.isZero()) {
            normalize(values);
            elementSum = ONE;
            normalizationRequired = false;
        }
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

    private void setProperties(@NotNull Real[] values) {
        this.nonZeroElements = 0;
        this.elementSum = ZERO;
        Arrays.stream(values).forEach(d -> {
            this.nonZeroElements += (d.isZero() ? 0 : 1);
            this.elementSum = this.elementSum.plus(d);
        });
        this.normalizationRequired = calculateIsNormalizationRequired();
    }

    private boolean calculateIsNormalizationRequired() {
        return elementSum.isZero() || (elementSum.isNotEqual(ONE) && nonZeroElements == symbolIndexMap.size());
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
            Real[] newValues = new Real[values.length];
            for (int i = 0; i < values.length; i++) {
                newValues[i] = new Real(values[i]);
            }
            return create(newValues);
        }

        @NotNull
        public ProbabilityVector create(@NotNull Real... values) {
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
