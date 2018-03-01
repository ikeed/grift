package com.grift.math.stats;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import com.grift.forex.symbol.SymbolPair;
import com.grift.math.real.Real;
import com.grift.spring.service.DecoupleService;
import org.jetbrains.annotations.NotNull;

class MarketSimulator implements Iterator<Map<SymbolPair, Real>> {
    @NotNull
    private final ProbabilityVector vec;
    @NotNull
    private final double[] wigglitude;

    private final DecoupleService decoupleService;
    private final List<SymbolPair> symbolPairList;
    private final int wiggleDigit = 4;

    private int iterations;

    public MarketSimulator(@NotNull List<SymbolPair> symbolPairList, @NotNull DecoupleService decoupleService, @NotNull ProbabilityVector startingVector) {
        this(symbolPairList, decoupleService, startingVector, -1);
    }

    public MarketSimulator(List<SymbolPair> symbolPairList, @NotNull DecoupleService decoupleService, @NotNull ProbabilityVector startingVector, int iterations) {
        this.symbolPairList = symbolPairList;
        this.vec = startingVector;
        this.iterations = iterations;
        this.wigglitude = new double[startingVector.getDimension()];
        this.decoupleService = decoupleService;
        Arrays.fill(wigglitude, 1d);
        mutateTrends(wigglitude, wiggleDigit - 1);
    }

    /**
     * Returns {@code true} if the iteration has more elements.
     * (In other words, returns {@code true} if {@link #next} would
     * return an element rather than throwing an exception.)
     *
     * @return {@code true} if the iteration has more elements
     */
    @Override
    public boolean hasNext() {
        return iterations != 0;
    }

    /**
     * Returns the next element in the iteration.
     *
     * @return the next element in the iteration
     * @throws NoSuchElementException if the iteration has no more elements
     */
    @Override
    public Map<SymbolPair, Real> next() {
        if (!hasNext()) {
            throw new NoSuchElementException("Check hasNext fool!");
        }
        if (iterations > 0) {
            iterations--;
        }
        return decoupleService.recouple(symbolPairList, doPermute(vec));
    }

    private ProbabilityVector doPermute(ProbabilityVector vec) {
        mutateTrends(wigglitude, wiggleDigit);
        for (int i = 0; i < vec.getDimension(); i++) {
            vec.put(i, vec.get(i).times(Real.valueOf(wigglitude[i])));
        }
        return vec;
    }

    private void mutateTrends(double[] ratios, int decimalPlaces) {
        for (int i = 0; i < ratios.length; i++) {
            double delta = Math.random() * Math.pow(10, -decimalPlaces);
            int polarity = 1;

            if (ratios[i] < 1) {
                polarity = -1;
            }
            if (Math.round(Math.rint(100)) % (ratios[i] == 0 ? 2 : 4) == 0) {
                polarity *= -1;
            }
            double wigglage = 1 + delta * polarity;
            ratios[i] *= wigglage;
        }
    }
}
