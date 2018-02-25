package com.grift.math.decoupler;

import java.util.List;
import java.util.stream.IntStream;
import com.google.common.collect.Lists;
import com.grift.forex.symbol.SymbolIndexMap;
import com.grift.forex.symbol.SymbolPair;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.test.context.junit4.SpringRunner;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

@RunWith(SpringRunner.class)
public class SymbolIndexMapTest {
    private SymbolIndexMap symbolIndexMap;

    @Before
    public void setup() {
        symbolIndexMap = new SymbolIndexMap();
    }

    @Test
    public void addSymbol() {
        assertTrue("should add successfully", symbolIndexMap.addSymbol("CAD"));
    }

    @Test
    public void addSymbolDupe() {
        assertTrue("should add successfully", symbolIndexMap.addSymbol("CAD"));
        assertFalse("should not add dupe", symbolIndexMap.addSymbol("CAD"));
    }

    @Test
    public void addSymbolPair() {
        symbolIndexMap.addSymbolPair(new SymbolPair("USDEUR")); //[0,1]
        Integer[] indices = symbolIndexMap.addSymbolPair(new SymbolPair("EURCAD")); //[1,2]
        assertEquals("indices[0] wrong", 1, (int) indices[0]);
        assertEquals("indices[1] wrong", 2, (int) indices[1]);
    }

    @Test
    public void getIndecesForSymbolPair() {
        assertTrue("Should add successfully", symbolIndexMap.addSymbol("CAD")); //0
        assertTrue("Should add successfully", symbolIndexMap.addSymbol("USD")); //1
        assertTrue("Should add successfully", symbolIndexMap.addSymbol("EUR")); //2
        Integer[] indices = symbolIndexMap.getIndicesForSymbolPair(new SymbolPair("EURUSD"));
        assertNotNull("not nul", indices);
        assertEquals("returned array should have two elements", 2, indices.length);
        assertEquals("indices[0] wrong", 2, (int) indices[0]);
        assertEquals("indices[1] wrong", 1, (int) indices[1]);
    }

    @Test
    public void size() {
        List<String> symbols = Lists.newArrayList("CAD", "AUD", "EUR", "NZD", "USD");
        IntStream.range(0, symbols.size() - 1).forEach(i -> {
            assertEquals("size should match", i, symbolIndexMap.size());
            symbolIndexMap.addSymbol(symbols.get(i));
        });
    }

    @Test
    public void getSymbols() {
        List<String> symbols = Lists.newArrayList("CAD", "AUD", "EUR", "NZD", "USD");
        IntStream.range(0, symbols.size() - 1).forEach(i -> symbolIndexMap.addSymbol(symbols.get(i)));
        List<String> arr = symbolIndexMap.getAllSymbols();
        IntStream.range(0, arr.size() - 1).forEach(i -> {
            Integer[] indices = symbolIndexMap.getIndicesForSymbolPair(new SymbolPair(arr.get(i) + arr.get(i + 1)));
            assertEquals(i, (int) indices[0]);
            assertEquals(i + 1, (int) indices[1]);
        });
    }

    @Test
    public void addSymbols() {
        List<String> symbols = Lists.newArrayList("CAD", "AUD", "EUR", "NZD", "USD");
        symbolIndexMap.addSymbols(symbols);
        List<String> actual = symbolIndexMap.getAllSymbols();
        assertEquals("size should match", symbols.size(), actual.size());
        assertArrayEquals("Lists don't match", symbols.toArray(), actual.toArray());
    }
}