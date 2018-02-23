package com.grift.forex.symbol;

import java.util.ArrayList;
import java.util.HashSet;
import com.google.common.collect.Lists;
import com.grift.math.ProbabilityVector;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;

@RunWith(MockitoJUnitRunner.class)
public class ProbabilityVectorTest {

    private static final ArrayList<String> SYMBOLS = Lists.newArrayList("CAD", "USD", "GBP");
    private final double EPSILON = 0.000001;
    private ImmutableSymbolIndexMap symbolIndexMap;

    @Before
    public void setUp() {
        symbolIndexMap = new SymbolIndexMap()
                .addSymbols(SYMBOLS)
                .getImmutablecopy();
    }

    @Test
    public void oneArgConstructor() {
        ProbabilityVector probabilityVector = new ProbabilityVector(symbolIndexMap);
        for (int i = 0; i < symbolIndexMap.size(); i++) {
            assertEquals("should start at zero", 0, probabilityVector.get(i), EPSILON);
        }
        for (String symbol : symbolIndexMap.keySet()) {
            assertEquals("should start at zero", 0, probabilityVector.get(symbol), EPSILON);
        }
        assertEquals("dimension should be set", symbolIndexMap.size(), probabilityVector.getDimension());
        assertNotNull("symbol map should be initialized", probabilityVector.getSymbolIndexMap());
    }

    @Test
    public void twoArgConstructor() {
        double[] expected = new double[]{
                /*CAD*/ 10,
                /*USD*/ 20,
                /*GBP*/ 30
        };
        ProbabilityVector probabilityVector = new ProbabilityVector(symbolIndexMap, expected);
        for (int i = 0; i < SYMBOLS.size(); i++) {
            assertEquals("value mismatch", expected[i] / 60, probabilityVector.get(i), EPSILON);
        }
        for (String symbol : SYMBOLS) {
            assertEquals("value mismatch", expected[symbolIndexMap.get(symbol)] / 60, probabilityVector.get(symbol), EPSILON);
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void twoArgConstructorWeirdLength() {
        assertNotEquals("You need to update this test", 4, symbolIndexMap.size());
        double[] values = new double[] {
                10, 20, 30, 40
        };
        new ProbabilityVector(symbolIndexMap, values);
    }

    @Test
    public void putGetSymbol() {
        double[] expected = new double[]{
                /*CAD*/ 10,
                /*USD*/ 20,
                /*GBP*/ 30
        };
        ProbabilityVector probabilityVector = new ProbabilityVector(symbolIndexMap, expected);

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
        ProbabilityVector probabilityVector = new ProbabilityVector(symbolIndexMap, expected);

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
        ProbabilityVector probabilityVector = new ProbabilityVector(symbolIndexMap, expected);
        probabilityVector.get(3);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void getIndexTooSmall() {
        double[] expected = new double[]{
                /*CAD*/ 10,
                /*USD*/ 20,
                /*GBP*/ 30
        };
        ProbabilityVector probabilityVector = new ProbabilityVector(symbolIndexMap, expected);
        probabilityVector.get(-1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void putNegativeValue() {
        ProbabilityVector probabilityVector = new ProbabilityVector(symbolIndexMap);
        probabilityVector.put("CAD", -10);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void putIndexTooLarge() {
        ProbabilityVector probabilityVector = new ProbabilityVector(symbolIndexMap);
        probabilityVector.put(3, 10);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void putIndexTooSmall() {
        ProbabilityVector probabilityVector = new ProbabilityVector(symbolIndexMap);
        probabilityVector.put(-1, 10);
    }

    @Test
    public void putZeroAfterNonZero() {
        ProbabilityVector probabilityVector = new ProbabilityVector(symbolIndexMap);
        probabilityVector.put("USD", 10);
        probabilityVector.put("USD", 0);
        assertEquals(0, probabilityVector.get("USD"), EPSILON);
    }

    @Test(expected = IllegalArgumentException.class)
    public void putBogusSymbol() {
        ProbabilityVector probabilityVector = new ProbabilityVector(symbolIndexMap);
        probabilityVector.put("POO", 10);
    }

    @Test(expected = IllegalArgumentException.class)
    public void getBogusSymbol() {
        ProbabilityVector probabilityVector = new ProbabilityVector(symbolIndexMap);
        probabilityVector.get("POO");
    }

    @Test
    public void equalsNull() {
        ProbabilityVector probabilityVector = new ProbabilityVector(symbolIndexMap);
        //noinspection ObjectEqualsNull
        assertFalse(probabilityVector.equals(null));
    }

    @Test
    public void equalsWTF() {
        ProbabilityVector probabilityVector = new ProbabilityVector(symbolIndexMap);
        //noinspection EqualsBetweenInconvertibleTypes
        assertFalse(probabilityVector.equals(new HashSet<>()));
    }

    @Test
    public void toStringNormalizes() {
        ProbabilityVector probabilityVector = new ProbabilityVector(symbolIndexMap, new double[] {1, 1, 2});
        assertEquals("toString should normalize", "<0.25, 0.25, 0.5>", probabilityVector.toString());
    }

}
