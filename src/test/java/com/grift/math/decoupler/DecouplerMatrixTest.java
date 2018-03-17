package com.grift.math.decoupler;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.forex.symbol.SymbolPair;
import com.grift.math.real.Real;
import com.grift.math.stats.ProbabilityVector;
import org.jetbrains.annotations.NotNull;
import org.junit.Before;
import org.junit.Test;

import static com.google.common.collect.Sets.cartesianProduct;
import static com.grift.math.real.Real.ONE;
import static com.grift.math.real.Real.ZERO;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

@SuppressWarnings("SpellCheckingInspection")
public abstract class DecouplerMatrixTest {

    private static final int DIGITS_PRECISION = 13;
    private SymbolIndexMap symbolIndexMap;
    private Factory factory;

    @Before
    public void setup() {
        symbolIndexMap = new SymbolIndexMap();
        Lists.newArrayList("CAD", "USD", "EUR").forEach(s -> symbolIndexMap.addSymbol(s));
        factory = getFactory(symbolIndexMap);
    }

    @NotNull
    protected abstract Factory getFactory(SymbolIndexMap symbolIndexMap);

    @Test
    public void decouple() {
        final List<String> allSymbols = symbolIndexMap.getAllSymbols();
        final Map<String, Real> trueValues = createRandomValues(allSymbols);
        final DecouplerMatrix mat = setInitialConditions(trueValues);
        ProbabilityVector result = mat.decouple();

        allSymbols.forEach(sym1 -> {
            Real expected = trueValues.get(sym1);
            Real actual = result.get(sym1);
            Real percentDifference = (ONE.subtract(expected.divide(actual))).abs().times(Real.valueOf(100d));
            String msg = String.format("mismatch (%f%% difference)", percentDifference.toDouble());
            assertEquals(msg, expected.setDigitsPrecision(DIGITS_PRECISION), actual.setDigitsPrecision(DIGITS_PRECISION));
        });
    }

    @Test
    public void rowsAndColumns() {
        final Map<String, Real> trueValues = createRandomValues(symbolIndexMap.getAllSymbols());
        final DecouplerMatrix mat = setInitialConditions(trueValues);

        assertEquals("rows", symbolIndexMap.size(), mat.rows());
        assertEquals("columns", symbolIndexMap.size(), mat.columns());
    }

    @Test
    public void getValueInvalidSymbol() {
        final Map<String, Real> trueValues = createRandomValues(symbolIndexMap.getAllSymbols());
        final DecouplerMatrix mat = setInitialConditions(trueValues);

        assertEquals(ZERO, mat.get(new SymbolPair("OOPSEZ")));
    }

    @Test(expected = IllegalArgumentException.class)
    public void putValueNegative() {
        final Map<String, Real> trueValues = createRandomValues(symbolIndexMap.getAllSymbols());
        final DecouplerMatrix mat = setInitialConditions(trueValues);
        mat.put(new SymbolPair("USDCAD"), Real.valueOf(-10));
    }

    @Test
    public void newTick() {
        final List<String> allSymbols = symbolIndexMap.getAllSymbols();
        final Map<String, Real> trueValues = createRandomValues(allSymbols);
        final DecouplerMatrix mat = setInitialConditions(trueValues);
        SymbolPair symbolPair = getRandomPair(allSymbols);

        mat.put(symbolPair, Real.valueOf(1000));

        assertEquals("Should match", Real.valueOf(1000), mat.get(symbolPair));
    }

    @Test
    public void isReplete() {
        final List<String> allSymbols = symbolIndexMap.getAllSymbols();
        String cur = allSymbols.remove(0);
        final Map<String, Real> trueValues = createRandomValues(allSymbols);
        final DecouplerMatrix mat = setInitialConditions(trueValues);
        assertFalse(mat.isReplete());
        mat.put(new SymbolPair(cur, allSymbols.get(0)), Real.valueOf(0.2));
        assertTrue(mat.isReplete());
    }

    private int randomInt(int tooLarge) {
        return ((int) (Math.random() * tooLarge * 31)) % tooLarge;
    }

    @SuppressWarnings("StatementWithEmptyBody")
    private SymbolPair getRandomPair(@NotNull List<String> allSymbols) {
        int i = randomInt(allSymbols.size());
        int j;
        while ((j = randomInt(allSymbols.size())) == i) {
        }
        return new SymbolPair(allSymbols.get(i) + allSymbols.get(j));
    }

    @SuppressWarnings("unchecked")
    @NotNull
    private DecouplerMatrix setInitialConditions(Map<String, Real> trueValues) {
        final DecouplerMatrix mat = factory.make();
        final List<String> currencyList = Lists.newArrayList(trueValues.keySet());

        HashSet<String> currencies = new HashSet<>(currencyList);
        cartesianProduct(currencies, currencies).forEach(pair -> {
            String sym1 = pair.get(0);
            String sym2 = pair.get(1);
            if (sym1.equals(sym2)) return;
            SymbolPair symbolPair = new SymbolPair(sym1 + sym2);
            Real val1 = trueValues.get(sym1);
            Real val2 = trueValues.get(sym2);
            Real val = val1.divide(val2);
            mat.put(symbolPair, val);
        });
        return mat;
    }

    @NotNull
    private Map<String, Real> createRandomValues(List<String> currencyList) {
        AtomicReference<Real> sum = new AtomicReference<>(ZERO);
        Map<String, Real> trueValues = Maps.newHashMap();
        currencyList.forEach(key -> {
            Real value = new Real(100 * Math.random());
            sum.updateAndGet(v -> v.plus(value));
            trueValues.put(key, value);
        });
        trueValues.forEach((key, value) -> trueValues.put(key, value.divide(sum.get())));
        return trueValues;
    }
}