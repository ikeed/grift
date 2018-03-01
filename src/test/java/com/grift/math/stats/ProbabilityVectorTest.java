package com.grift.math.stats;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.stream.IntStream;
import com.google.common.collect.Lists;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.math.real.Real;
import org.jetbrains.annotations.NotNull;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static com.grift.math.real.Real.ZERO;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

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
        IntStream.range(0, symbolIndexMap.size()).forEach(i -> assertEquals("should start at zero", ZERO, probabilityVector.get(i)));
        for (String symbol : symbolIndexMap.keySet()) {
            assertEquals("should start at zero", ZERO, probabilityVector.get(symbol));
        }
        assertEquals("dimension should be set", symbolIndexMap.size(), probabilityVector.getDimension());
        assertNotNull("symbol map should be initialized", probabilityVector.getSymbolIndexMap());
    }

    @Test
    public void twoArgConstructor() {
        Real[] expected = getReals(10, 20, 30);
        ProbabilityVector probabilityVector = vectorFactory.create(expected);
        IntStream.range(0, SYMBOLS.size()).forEach(i -> assertEquals("value mismatch", expected[i].divide(Real.valueOf(60)), probabilityVector.get(i)));
        SYMBOLS.forEach(symbol -> assertEquals("value mismatch", expected[symbolIndexMap.get(symbol)].divide(Real.valueOf(60)), probabilityVector.get(symbol)));
    }

    @NotNull
    public Real[] getReals(double... vals) {
        Real[] arr = new Real[vals.length];
        for (int i = 0; i < vals.length; i++) {
            arr[i] = new Real(vals[i]);
        }
        return arr;
    }

    @Test(expected = IllegalArgumentException.class)
    public void twoArgConstructorWeirdLength() {
        assertNotEquals("You need to update this test", 4, symbolIndexMap.size());
        vectorFactory.create(getReals(10, 20, 30, 40));
    }

    @Test
    public void putGetSymbol() {
        Real[] expected = getReals(10, 20, 30);
        ProbabilityVector probabilityVector = vectorFactory.create(expected);

        probabilityVector.put("USD", probabilityVector.get("USD").times(Real.valueOf(2)));
        Real actual = probabilityVector.get("USD");

        assertEquals(new Real(0.5).setDigitsPrecision(2), actual.setDigitsPrecision(2));
    }

    @Test
    public void putGetIndex() {
        Real[] expected = getReals(10, 20, 30);
        ProbabilityVector probabilityVector = vectorFactory.create(expected);

        probabilityVector.put(1, probabilityVector.get(1).times(Real.valueOf(2)));
        Real actual = probabilityVector.get(1);

        assertEquals(expected[0].divide(expected[1].setDigitsPrecision(2)), actual.setDigitsPrecision(2));
    }


    @Test
    public void testPutIncremental() {
        ProbabilityVector vec = vectorFactory.create();
        vec.put(0, Real.valueOf(10));
        vec.put(1, Real.valueOf(20));
        vec.put(2, Real.valueOf(70));
        assertEquals("[0.1, 0.2, 0.7]", vec.toString());
    }


    @Test(expected = IndexOutOfBoundsException.class)
    public void getIndexTooLarge() {
        Real[] expected = getReals(10, 20, 30);

        ProbabilityVector probabilityVector = vectorFactory.create(expected);
        probabilityVector.get(3);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void getIndexTooSmall() {
        ProbabilityVector probabilityVector = vectorFactory.create(getReals(10, 20, 30));
        probabilityVector.get(-1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void putNegativeValue() {
        ProbabilityVector probabilityVector = vectorFactory.create();
        probabilityVector.put("CAD", Real.valueOf(-10));
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void putIndexTooLarge() {
        ProbabilityVector probabilityVector = vectorFactory.create();
        probabilityVector.put(3, Real.valueOf(10));
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void putIndexTooSmall() {
        ProbabilityVector probabilityVector = vectorFactory.create();
        probabilityVector.put(-1, Real.valueOf(10));
    }

    @Test(expected = IllegalArgumentException.class)
    public void putBogusSymbol() {
        ProbabilityVector probabilityVector = vectorFactory.create();
        probabilityVector.put("POO", Real.valueOf(10));
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
    public void equalsTrue() {
        ProbabilityVector probabilityVector = vectorFactory.create(1, 2, 3);
        ProbabilityVector probabilityVector2 = vectorFactory.create(1, 2, 3);
        assertTrue(probabilityVector.equals(probabilityVector2));
    }

    @Test
    public void equalsFalse() {
        ProbabilityVector probabilityVector = vectorFactory.create(1, 2, 3);
        ProbabilityVector probabilityVector2 = vectorFactory.create(1, 2, 3.1);
        assertFalse(probabilityVector.equals(probabilityVector2));
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
        Real[] arr = getReals(10, 20, 70);
        Real[] result = ProbabilityVector.normalize(arr);
        assertEquals("[0.1, 0.2, 0.7]", Arrays.toString(result));
    }

    @Test
    public void testMultiplePut() {
        ProbabilityVector probabilityVector = vectorFactory.create(1, 2, 7);

        assertEquals("[0.1, 0.2, 0.7]", probabilityVector.toString());
    }

    @Test
    public void testPutBackToZero() {
        Real[] inputs = getReals(10, 20, 30);
        ProbabilityVector probabilityVector = vectorFactory.create(inputs);
        probabilityVector.put(1, ZERO);
        Real[] expected = getReals(0.25, 0, 0.75);
        for (int i = 0; i < 3; i++) {
            assertEquals(expected[i].setDigitsPrecision(2), probabilityVector.get(i).setDigitsPrecision(2));
        }
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNegativeElement() {
        vectorFactory.create(10, -20, 30);
    }
}
