package com.grift.math;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.stream.IntStream;
import com.google.common.collect.Lists;
import com.grift.forex.symbol.SymbolIndexMap;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.apache.commons.math.util.MathUtils.EPSILON;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;

@RunWith(MockitoJUnitRunner.class)
public class ProbabilityVectorTest {

    private static final ArrayList<String> SYMBOLS = Lists.newArrayList("CAD", "USD", "GBP");
    private SymbolIndexMap symbolIndexMap;
    private ProbabilityVector.Factory vectorFactory;

    @Before
    public void setUp() {
        symbolIndexMap = new SymbolIndexMap()
                .addSymbols(SYMBOLS)
                .getImmutableCopy();
        vectorFactory = new ProbabilityVector.Factory(symbolIndexMap.getImmutableCopy());
    }

    @Test
    public void oneArgConstructor() {
        ProbabilityVector probabilityVector = vectorFactory.create();
        IntStream.range(0, symbolIndexMap.size()).forEach(i -> assertEquals("should start at zero", 0, probabilityVector.get(i), EPSILON));
        symbolIndexMap.keySet().forEach(symbol -> assertEquals("should start at zero", 0, probabilityVector.get(symbol), EPSILON));
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
        ProbabilityVector probabilityVector = vectorFactory.create( expected);
        IntStream.range(0, SYMBOLS.size()).forEach(i -> assertEquals("value mismatch", expected[i] / 60, probabilityVector.get(i), EPSILON));
        SYMBOLS.forEach(symbol -> assertEquals("value mismatch", expected[symbolIndexMap.get(symbol)] / 60, probabilityVector.get(symbol), EPSILON));
    }

    @Test(expected = IllegalArgumentException.class)
    public void twoArgConstructorWeirdLength() {
        assertNotEquals("You need to update this test", 4, symbolIndexMap.size());
        double[] values = new double[] {
                10, 20, 30, 40
        };
        vectorFactory.create( values);
    }

    @Test
    public void putGetSymbol() {
        double[] expected = new double[]{
                /*CAD*/ 10,
                /*USD*/ 20,
                /*GBP*/ 30
        };
        ProbabilityVector probabilityVector = vectorFactory.create( expected);

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
        ProbabilityVector probabilityVector = vectorFactory.create( expected);

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
        ProbabilityVector probabilityVector = vectorFactory.create( expected);
        probabilityVector.get(3);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void getIndexTooSmall() {
        double[] expected = new double[]{
                /*CAD*/ 10,
                /*USD*/ 20,
                /*GBP*/ 30
        };
        ProbabilityVector probabilityVector = vectorFactory.create( expected);
        probabilityVector.get(-1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void putNegativeValue() {
        ProbabilityVector probabilityVector = vectorFactory.create();
        probabilityVector.put("CAD", -10);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void putIndexTooLarge() {
        ProbabilityVector probabilityVector = vectorFactory.create();
        probabilityVector.put(3, 10);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void putIndexTooSmall() {
        ProbabilityVector probabilityVector = vectorFactory.create();
        probabilityVector.put(-1, 10);
    }

    @Test
    public void putZeroAfterNonZero() {
        ProbabilityVector probabilityVector = vectorFactory.create();
        probabilityVector.put("USD", 10);
        probabilityVector.put("USD", 0);
        assertEquals(0, probabilityVector.get("USD"), EPSILON);
    }

    @Test(expected = IllegalArgumentException.class)
    public void putBogusSymbol() {
        ProbabilityVector probabilityVector = vectorFactory.create();
        probabilityVector.put("POO", 10);
    }

    @Test(expected = IllegalArgumentException.class)
    public void getBogusSymbol() {
        ProbabilityVector probabilityVector = vectorFactory.create();
        probabilityVector.get("POO");
    }

    @Test
    public void equalsNull() {
        ProbabilityVector probabilityVector = vectorFactory.create();
        //noinspection ObjectEqualsNull
        assertFalse(probabilityVector.equals(null));
    }

    @Test
    public void equalsWTF() {
        ProbabilityVector probabilityVector = vectorFactory.create();
        //noinspection EqualsBetweenInconvertibleTypes
        assertFalse(probabilityVector.equals(new HashSet<>()));
    }

    @Test
    public void toStringNormalizes() {
        ProbabilityVector probabilityVector = vectorFactory.create(1, 1, 2);
        assertEquals("toString should normalize", "[0.25, 0.25, 0.5]", probabilityVector.toString());
    }

    @Test
    public void hashCodeMatches() {
        ProbabilityVector probabilityVector = vectorFactory.create(1, 1, 2);
        ProbabilityVector probabilityVector2 = vectorFactory.create(1, 1, 2);
        assertEquals(probabilityVector.hashCode(), probabilityVector2.hashCode());
    }


    @Test
    public void hashCodeChanges() {
        ProbabilityVector probabilityVector = vectorFactory.create(1, 1, 2);
        ProbabilityVector probabilityVector2 = vectorFactory.create(1, 2, 2);
        assertNotEquals(probabilityVector.hashCode(), probabilityVector2.hashCode());
    }

    @Test
    public void testNormalize() {
        double[] arr = new double[] { 10, 20, 70};
        double[] result = ProbabilityVector.normalize(arr);
        assertEquals("[0.1, 0.2, 0.7]", Arrays.toString(result));
    }
}
