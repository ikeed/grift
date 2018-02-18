package com.grift.forex.symbol;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.runners.MockitoJUnitRunner;

import static org.junit.Assert.assertTrue;

@RunWith(MockitoJUnitRunner.class)
public class ImmutableSymbolIndexMapTest {

    private SymbolIndexMap symbolIndexMap;

    @Before
    public void setup() {
        SymbolIndexMap map = new SymbolIndexMap();
        map.addSymbol("CAD");
        map.addSymbol("USD");
        map.addSymbol("EUR");
        symbolIndexMap = new ImmutableSymbolIndexMap(map);
    }

    @Test
    public void addSymbolExists() {
        assertTrue(symbolIndexMap.addSymbol("CAD"));
    }

    @Test
    public void addSymbolPairExists() {
        symbolIndexMap.addSymbolPair(new SymbolPair("CADUSD"));
    }


    @Test(expected = UnsupportedOperationException.class)
    public void addSymbolNotExists() {
        symbolIndexMap.addSymbol("BRF");
    }

    @Test(expected = UnsupportedOperationException.class)
    public void addSymbolPairNotExists() {
        symbolIndexMap.addSymbolPair(new SymbolPair("BARFED"));
    }
}