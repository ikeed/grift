package com.grift.forex.symbol;

import com.google.common.collect.Lists;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import java.util.ArrayList;

import static org.junit.Assert.assertEquals;

@RunWith(MockitoJUnitRunner.class)
public class ProbabilityVectorTest {

    private static final ArrayList<String> SYMBOLS = Lists.newArrayList("CAD", "USD", "GBP");
    private final double EPSILON = 0.000001;
    private ImmutableSymbolIndexMap symbolIndexMap;
    private ProbabilityVector probabilityVector;

    @Before
    public void setUp() {
        symbolIndexMap = new SymbolIndexMap()
                .addSymbols(SYMBOLS)
                .getImmutablecopy();
    }

    @Test
    public void oneArgConstructor() {
        probabilityVector = new ProbabilityVector(symbolIndexMap);
        for (int i = 0; i < symbolIndexMap.size(); i++) {
            assertEquals("should start at zero", 0, probabilityVector.get(i), EPSILON);
        }
        for (String symbol : symbolIndexMap.keySet()) {
            assertEquals("should start at zero", 0, probabilityVector.get(symbol), EPSILON);
        }
    }

    @Test
    public void twoArgConstructor() {
        double[] expected = new double[]{
                /*CAD*/ 10,
                /*USD*/ 20,
                /*GBP*/ 30
        };
        probabilityVector = new ProbabilityVector(symbolIndexMap, expected);
        for (int i = 0; i < SYMBOLS.size(); i++) {
            assertEquals("value mismatch", expected[i] / 60, probabilityVector.get(i), EPSILON);
        }
        for (String symbol : SYMBOLS) {
            assertEquals("value mismatch", expected[symbolIndexMap.get(symbol)] / 60, probabilityVector.get(symbol), EPSILON);
        }
    }


    @Test
    public void putGetSymbol() {
        double[] expected = new double[]{
                /*CAD*/ 10,
                /*USD*/ 20,
                /*GBP*/ 30
        };
        probabilityVector = new ProbabilityVector(symbolIndexMap, expected);

        probabilityVector.put("USD", probabilityVector.get("USD") * 2);
        double actual = probabilityVector.get("USD");

        assertEquals(0.5, actual, EPSILON);
    }

    @Test
    public void putGetIndex() {
        double[] expected = new double[]{
                /*CAD*/ 10,
                /*USD*/ 20,
                /*GBP*/ 30
        };
        probabilityVector = new ProbabilityVector(symbolIndexMap, expected);

        probabilityVector.put(1, probabilityVector.get(1) * 2);
        double actual = probabilityVector.get(1);

        assertEquals(0.5, actual, EPSILON);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void getIndexTooLarge() {
        double[] expected = new double[]{
                /*CAD*/ 10,
                /*USD*/ 20,
                /*GBP*/ 30
        };
        probabilityVector = new ProbabilityVector(symbolIndexMap, expected);
        probabilityVector.get(3);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void getIndexTooSmall() {
        double[] expected = new double[]{
                /*CAD*/ 10,
                /*USD*/ 20,
                /*GBP*/ 30
        };
        probabilityVector = new ProbabilityVector(symbolIndexMap, expected);
        probabilityVector.get(-1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void putNegativeValue() {
        probabilityVector = new ProbabilityVector(symbolIndexMap);
        probabilityVector.put("CAD", -10);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void putIndexTooLarge() {
        probabilityVector = new ProbabilityVector(symbolIndexMap);
        probabilityVector.put(3, 10);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void putIndexTooSmall() {
        probabilityVector = new ProbabilityVector(symbolIndexMap);
        probabilityVector.put(-1, 10);
    }

    @Test
    public void putZeroAfterNonZero() {
        probabilityVector = new ProbabilityVector(symbolIndexMap);
        probabilityVector.put("USD", 10);
        probabilityVector.put("USD", 0);
        assertEquals(0, probabilityVector.get("USD"), EPSILON);
    }

    @Test(expected = IllegalArgumentException.class)
    public void putBogusSymbol() {
        probabilityVector = new ProbabilityVector(symbolIndexMap);
        probabilityVector.put("POO", 10);
    }

    @Test(expected = IllegalArgumentException.class)
    public void getBogusSymbol() {
        probabilityVector = new ProbabilityVector(symbolIndexMap);
        probabilityVector.get("POO");
    }
}
