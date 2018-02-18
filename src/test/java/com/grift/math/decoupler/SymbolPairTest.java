package com.grift.math.decoupler;


import com.grift.forex.symbol.SymbolPair;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertEquals;

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

    @Test(expected = NullPointerException.class)
    public void nullArgument() {
        new SymbolPair(null);
    }
}