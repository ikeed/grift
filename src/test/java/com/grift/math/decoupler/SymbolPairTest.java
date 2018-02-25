package com.grift.math.decoupler;


import com.grift.forex.symbol.SymbolPair;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

@SuppressWarnings("EqualsBetweenInconvertibleTypes")
@RunWith(MockitoJUnitRunner.class)
public class SymbolPairTest {
    @Test
    public void happyPath() {
        final SymbolPair symbolPair = new SymbolPair("CADUSD");

        assertEquals("first should match", "CAD", symbolPair.getFirst());
        assertEquals("second should match", "USD", symbolPair.getSecond());
    }

    @Test(expected = IllegalArgumentException.class)
    public void tooLong() {
        new SymbolPair("CAD/USD");
    }

    @Test(expected = IllegalArgumentException.class)
    public void tooShort() {
        new SymbolPair("CAUSD");
    }

    @Test(expected = IllegalArgumentException.class)
    public void wrongCharacters() {
        new SymbolPair("CA/USD");
    }

    @SuppressWarnings("ConstantConditions")
    @Test(expected = IllegalArgumentException.class)
    public void nullArgument() {
        new SymbolPair(null);
    }

    @Test @SuppressWarnings("all")
    public void equalsNullFalse() {
        assertFalse(new SymbolPair("CADUSD").equals(null));
    }

    @Test
    public void equalsStringFalse() {
        assertFalse(new SymbolPair("CADUSD").equals("CADPOO"));
    }

    @Test
    public void equalsStringTrue() {
        assertTrue(new SymbolPair("CADUSD").equals("CADUSD"));
    }

    @Test
    public void equalsObjectFalse() {
        assertFalse(new SymbolPair("CADUSD").equals(new SymbolPair("CADPOO")));
    }

    @Test
    public void equalsObjectTrue() {
        assertTrue(new SymbolPair("CADUSD").equals(new SymbolPair("CADUSD")));
    }

    @Test
    public void equalsObjectWTF() {
        assertFalse(new SymbolPair("CADUSD").equals(-911));
    }

}