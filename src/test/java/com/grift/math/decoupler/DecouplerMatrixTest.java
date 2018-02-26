package com.grift.math.decoupler;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.grift.GriftApplication;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.forex.symbol.SymbolPair;
import com.grift.math.ProbabilityVector;
import org.jetbrains.annotations.NotNull;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.context.junit4.SpringRunner;

import static com.google.common.collect.Sets.cartesianProduct;
import static org.apache.commons.math.util.MathUtils.EPSILON;
import static org.junit.Assert.assertEquals;

@RunWith(SpringRunner.class)
@SpringBootTest(classes = {DecouplerMatrix.class})
@ContextConfiguration(name = "fixture", classes = {GriftApplication.class})
public class DecouplerMatrixTest {

    private SymbolIndexMap symbolIndexMap;
    private Factory factory;

    @Before
    public void setup() {
        symbolIndexMap = new SymbolIndexMap();
        Lists.newArrayList("CAD", "USD", "EUR").forEach(s -> symbolIndexMap.addSymbol(s));
        factory = new DecouplerMatrixColtImpl.ColtFactory(symbolIndexMap);
    }

    @Test
    public void decouple() {
        final List<String> allSymbols = symbolIndexMap.getAllSymbols();
        final Map<String, Double> trueValues = createRandomValues(allSymbols);
        final DecouplerMatrix mat = setInitialConditions(trueValues);
        ProbabilityVector result = mat.decouple();

        allSymbols.forEach(sym1 -> {
            Double expected = trueValues.get(sym1);
            double actual = result.get(sym1);
            assertEquals("mismatch", expected, actual, 10 * EPSILON);
        });
    }

    @Test
    public void rowsAndColumns() {
        final Map<String, Double> trueValues = createRandomValues(symbolIndexMap.getAllSymbols());
        final DecouplerMatrix mat = setInitialConditions(trueValues);

        assertEquals("rows", symbolIndexMap.size(), mat.rows());
        assertEquals("columns", symbolIndexMap.size(), mat.columns());
    }

    @Test
    public void getValueInvalidSymbol() {
        final Map<String, Double> trueValues = createRandomValues(symbolIndexMap.getAllSymbols());
        final DecouplerMatrix mat = setInitialConditions(trueValues);

        assertEquals(0, mat.get(new SymbolPair("OOPSEZ")), EPSILON);
    }

    @Test(expected = IllegalArgumentException.class)
    public void putValueNegative() {
        final Map<String, Double> trueValues = createRandomValues(symbolIndexMap.getAllSymbols());
        final DecouplerMatrix mat = setInitialConditions(trueValues);
        mat.put(new SymbolPair("USDCAD"), -10);
    }

    @Test
    public void newTick() {
        final List<String> allSymbols = symbolIndexMap.getAllSymbols();
        final Map<String, Double> trueValues = createRandomValues(allSymbols);
        final DecouplerMatrix mat = setInitialConditions(trueValues);
        SymbolPair symbolPair = getRandomPair(allSymbols);

        mat.put(symbolPair, 1000);

        assertEquals("Should match", 1000, mat.get(symbolPair), EPSILON);
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

    @NotNull
    private DecouplerMatrix setInitialConditions(Map<String, Double> trueValues) {
        final DecouplerMatrix mat = factory.make();
        final List<String> currencyList = Lists.newArrayList(trueValues.keySet());

        HashSet<String> currencies = new HashSet<>(currencyList);
        cartesianProduct(currencies, currencies).forEach(pair -> {
            String sym1 = pair.get(0);
            String sym2 = pair.get(1);
            if (sym1.equals(sym2)) return;
            SymbolPair symbolPair = new SymbolPair(sym1 + sym2);
            Double val1 = trueValues.get(sym1);
            Double val2 = trueValues.get(sym2);
            double val = val1 / val2;
            mat.put(symbolPair, val);
        });
        return mat;
    }

    @NotNull
    private Map<String, Double> createRandomValues(List<String> currencyList) {
        AtomicReference<Double> sum = new AtomicReference<>((double) 0);
        Map<String, Double> trueValues = Maps.newHashMap();
        currencyList.forEach(key -> {
            double value = 100 * Math.random();
            sum.updateAndGet(v -> v + value);
            trueValues.put(key, value);
        });
        trueValues.forEach((key, value) -> trueValues.put(key, value / sum.get()));
        return trueValues;
    }
}