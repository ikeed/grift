package com.grift.math.decoupler;

import java.util.List;
import java.util.Map;
import com.grift.math.ProbabilityVector;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.forex.symbol.SymbolPair;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import lombok.NonNull;
import org.jetbrains.annotations.NotNull;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static java.lang.Math.abs;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

@RunWith(MockitoJUnitRunner.class)
public abstract class DecouplerMatrixTest {

    private final double epsilon = 0.000001;
    SymbolIndexMap symbolIndexMap;
    private Factory factory;

    @Before
    public void setup() {
        symbolIndexMap = new SymbolIndexMap();
        for (String s : Lists.newArrayList("CAD", "USD", "EUR")) {
            symbolIndexMap.addSymbol(s);
        }
        factory = getFactory();
    }

    @NotNull
    protected abstract Factory getFactory();

    @Test
    public void decouple() {
        final List<String> allSymbols = symbolIndexMap.getAllSymbols();
        final Map<String, Double> trueValues = createRandomValues(allSymbols);
        final DecouplerMatrix mat = setInitialConditions(trueValues);

        ProbabilityVector result = mat.decouple();

        for (String sym1 : allSymbols) {
            for (String sym2 : allSymbols) {
                if (sym1.equals(sym2)) continue;
                double expected = trueValues.get(sym1) / trueValues.get(sym2);
                double actual = result.get(sym1) / result.get(sym2);
                assertTrue("Mismatch", abs(expected - actual) < epsilon);
            }
        }
    }

    @Test
    public void rowsAndColumns() {
        final List<String> allSymbols = symbolIndexMap.getAllSymbols();
        final Map<String, Double> trueValues = createRandomValues(allSymbols);
        final DecouplerMatrix mat = setInitialConditions(trueValues);

        assertEquals("rows", symbolIndexMap.size(), mat.rows());
        assertEquals("columns", symbolIndexMap.size(), mat.columns());
    }

    @Test
    public void newTick() {
        final List<String> allSymbols = symbolIndexMap.getAllSymbols();
        final Map<String, Double> trueValues = createRandomValues(allSymbols);
        final DecouplerMatrix mat = setInitialConditions(trueValues);
        SymbolPair symbolPair = getRandomPair(allSymbols);

        mat.put(symbolPair, 1000);

        assertEquals("Should match", 1000, mat.get(symbolPair), epsilon);
    }

    private int randomInt(int tooLarge) {
        return ((int) (Math.random() * tooLarge * 31)) % tooLarge;
    }

    @SuppressWarnings("StatementWithEmptyBody")
    private SymbolPair getRandomPair(@NonNull List<String> allSymbols) {
        int i = randomInt(allSymbols.size());
        int j;
        while ((j = randomInt(allSymbols.size())) == i) {
        }
        return new SymbolPair(allSymbols.get(i) + allSymbols.get(j));
    }

    private DecouplerMatrix setInitialConditions(Map<String, Double> trueValues) {
        DecouplerMatrix mat = factory.make();
        List<String> currencyList = Lists.newArrayList(trueValues.keySet());
        for (int i = 0; i < currencyList.size(); i++) {
            for (int j = i; j < currencyList.size(); j++) {
                final String sym1 = currencyList.get(i);
                final String sym2 = currencyList.get(j);
                SymbolPair pair = new SymbolPair(sym1 + sym2);
                mat.put(pair, trueValues.get(sym1) / trueValues.get(sym2));
            }
        }
        return mat;
    }

    @NotNull
    private Map<String, Double> createRandomValues(List<String> currencyList) {
        Map<String, Double> trueValues = Maps.newHashMap();
        for (String key : currencyList) {
            trueValues.put(key, 1000 * Math.random());
        }
        return trueValues;
    }
}