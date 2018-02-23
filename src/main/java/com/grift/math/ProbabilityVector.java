package com.grift.math;

import java.util.Arrays;
import com.grift.forex.symbol.ImmutableSymbolIndexMap;
import lombok.NonNull;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import static lombok.Lombok.checkNotNull;

public class ProbabilityVector {
    private static final double EPSILON = 0.00000001;
    @NonNull
    private final ImmutableSymbolIndexMap symbolIndexMap;
    @NotNull
    @NonNull
    private final double[] values;

    private boolean normalizationRequired = false;
    private int nonZeroElements = 0;
    private double elementSum = 0;

    public ProbabilityVector(@NotNull @NonNull ImmutableSymbolIndexMap symbolIndexMap) {
        this.symbolIndexMap = checkNotNull(symbolIndexMap, "map");
        this.values = new double[symbolIndexMap.keySet().size()];
        this.nonZeroElements = 0;
        this.normalizationRequired = false;
        this.elementSum = 0;
        Arrays.fill(values, 0d);
    }

    public ProbabilityVector(@NotNull @NonNull ImmutableSymbolIndexMap symbolIndexMap, @NotNull @NonNull double[] vals) {
        this.symbolIndexMap = checkNotNull(symbolIndexMap, "map");
        this.values = Arrays.copyOf(vals, vals.length);
        if (symbolIndexMap.size() != vals.length) {
            throw new IllegalArgumentException("Wrong number of values for symbol table");
        }
        setProperties(vals);
    }

    @NotNull
    public ProbabilityVector put(String symb, double v) {
        assertSymbol(symb);
        return put(symbolIndexMap.get(symb), v);
    }

    @NotNull
    public ProbabilityVector put(int index, double v) {
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
        return this;
    }

    public double get(String symb) {
        assertSymbol(symb);
        return get(symbolIndexMap.get(symb));
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

    public ImmutableSymbolIndexMap getSymbolIndexMap() {
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
        StringBuilder sb = new StringBuilder();
        for (double d : values) {
            if (sb.length() > 0) {
                sb.append(", ");
            }
            sb.append(d);
        }
        return "<" + sb.toString() + ">";
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
        if (elementSum > 0) {
            for (int i = 0; i < values.length; i++) {
                values[i] /= elementSum;
            }
        }
        elementSum = 1;
        normalizationRequired = false;
    }

    private void assertIndex(int index) {
        if (!isLegalIndex(index)) {
            throw new IndexOutOfBoundsException(index + " not in array");
        }
    }

    private void assertSymbol(String symb) {
        if (!isLegalSymbol(symb)) {
            throw new IllegalArgumentException("Unknown symbol: " + symb);
        }
    }

    private void setProperties(double[] vals) {
        this.nonZeroElements = 0;
        this.elementSum = 0;
        for (double d : vals) {
            this.nonZeroElements += (d == 0 ? 0 : 1);
            this.elementSum += d;
        }
        this.normalizationRequired = calculateIsNormalizationRequired();
    }

    private boolean calculateIsNormalizationRequired() {
        return !(isZero(elementSum - 1) && nonZeroElements == symbolIndexMap.size());
    }

    private boolean isZero(double v) {
        return Math.abs(v) < EPSILON;
    }

    private boolean isLegalSymbol(String symb) {
        return symbolIndexMap.containsKey(symb);
    }

    private boolean isLegalIndex(int index) {
        return symbolIndexMap.containsValue(index) && index >= 0 && index < values.length;
    }
}