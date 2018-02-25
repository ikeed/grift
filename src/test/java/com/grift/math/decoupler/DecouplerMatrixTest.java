package com.grift.math.decoupler;

import java.util.List;
import java.util.Map;
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

import static org.apache.commons.math.util.MathUtils.EPSILON;
import static org.junit.Assert.assertEquals;

@RunWith(SpringRunner.class)
@SpringBootTest(classes = {DecouplerMatrix.class})
@ContextConfiguration(name = "fixture", classes = {GriftApplication.class})
public class DecouplerMatrixTest {

    private final double epsilon = 0.000001;
    private SymbolIndexMap symbolIndexMap;
    private Factory factory;

    @Before
    public void setup() {
        symbolIndexMap = new SymbolIndexMap();
        for (String s : Lists.newArrayList("CAD", "USD", "EUR")) {
            symbolIndexMap.addSymbol(s);
        }
        factory = new DecouplerMatrixColtImpl.ColtFactory(symbolIndexMap);
    }

    @Test
    public void decouple() {
        final List<String> allSymbols = symbolIndexMap.getAllSymbols();
        final Map<String, Double> trueValues = createRandomValues(allSymbols);
        final DecouplerMatrix mat = setInitialConditions(trueValues);
        ProbabilityVector result = mat.decouple();

        for (String sym1 : allSymbols) {
            Double expected = trueValues.get(sym1);
            Integer index = symbolIndexMap.get(sym1);
            double actual = result.get(index);
            assertEquals("mismatch", expected, actual, epsilon);
        }
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

        for (int i = 0; i < currencyList.size(); i++) {
            for (int j = 0; j < currencyList.size(); j++) {
                if (i == j) continue;
                final String sym1 = currencyList.get(i);
                final String sym2 = currencyList.get(j);
                SymbolPair pair = new SymbolPair(sym1 + sym2);
                Double val1 = trueValues.get(sym1);
                Double val2 = trueValues.get(sym2);
                double val = val1 / val2;
                mat.put(pair, val);
//                mat.put(new SymbolPair(sym1 + sym2), 1 / val);
            }
        }
        return mat;
    }

    @NotNull
    private Map<String, Double> createRandomValues(List<String> currencyList) {
        double sum = 0;
        Map<String, Double> trueValues = Maps.newHashMap();
        for (String key : currencyList) {
            double value = 1000 * Math.random();
            sum += value;
            trueValues.put(key, value);
        }
        for (Map.Entry<String, Double> entry : trueValues.entrySet()) {
            trueValues.put(entry.getKey(), entry.getValue() / sum);
        }
        return trueValues;
    }
}